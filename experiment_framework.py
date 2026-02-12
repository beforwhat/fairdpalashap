# experiment_framework.py (更新版)
import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime

# 导入基线方法工厂
from baselines import BaselineFactory

class ExperimentFramework:
    """实验框架"""
    
    def __init__(self, args, data_distributions=None):
        self.args = args
        self.device = args.device
        self.data_distributions = data_distributions
        
        # 实验记录
        self.results = {
            'method': args.method,
            'dataset': args.dataset,
            'num_clients': args.num_clients,
            'global_epochs': args.global_epochs,
            'round_accuracies': [],  # 每轮全局准确率
            'client_accuracies': [],  # 每轮客户端准确率
            'fairness_variances': [],  # 公平性方差
            'communication_costs': [],  # 通信成本
            'training_losses': [],  # 训练损失
            'gradient_norms': []  # 梯度范数
        }
        
        # 创建结果目录
        os.makedirs(args.log_dir, exist_ok=True)
    
    def initialize_clients_and_server(self, model_class, train_loaders, 
                                     test_loaders, train_data_list=None):
        """
        初始化客户端和服务器
        
        Args:
            model_class: 模型类
            train_loaders: 训练数据加载器列表
            test_loaders: 测试数据加载器列表
            train_data_list: 训练数据列表（用于ALA）
            
        Returns:
            tuple: (clients, server)
        """
        # 创建全局模型
        global_model = model_class().to(self.device)
        
        # 获取方法特定参数
        method_params = BaselineFactory.get_method_params(self.args.method, self.args)
        
        # 创建服务器
        server_kwargs = {
            'num_clients': self.args.num_clients,
            'data_distributions': self.data_distributions
        }
        server_kwargs.update(method_params)
        
        server = BaselineFactory.create_server(
            self.args.method, global_model, self.device, **server_kwargs
        )
        
        # 创建客户端
        clients = []
        for client_id in range(self.args.num_clients):
            # 客户端特定参数
            client_kwargs = method_params.copy()
            need_train_data = (
            self.args.method == 'fedala' or
            (self.args.method in ['our_method', 'our_method_no_dp'] and self.args.use_ala)
            )
            if need_train_data:
              if train_data_list is None:
                raise ValueError(f"方法 {self.args.method} 需要提供 train_data_list")
              client_kwargs['train_data'] = train_data_list[client_id]
            if self.args.method == 'fedala' or (self.args.method in ['our_method', 'our_method_no_dp'] and self.args.use_ala):
                client_kwargs['train_data'] = train_data_list[client_id] if train_data_list else None
            
            if self.args.method in ['our_method', 'our_method_no_dp']:
                client_kwargs['data_distribution'] = self.data_distributions.get(client_id, []) if self.data_distributions else []
            
            # 创建客户端
            client = BaselineFactory.create_client(
                self.args.method, client_id, model_class(),
                train_loaders[client_id], test_loaders[client_id],
                self.device, **client_kwargs
            )
            clients.append(client)
        
        return clients, server
    
    def run_experiment(self, clients, server, test_loader):
        """运行实验"""
        print(f"\n{'='*60}")
        print(f"实验: {self.args.method}")
        print(f"数据集: {self.args.dataset}")
        print(f"客户端数: {self.args.num_clients}")
        print(f"全局轮数: {self.args.global_epochs}")
        print(f"本地轮数: {self.args.local_epochs}")
        print(f"学习率: {self.args.lr}")
        print(f"{'='*60}")
        
        # 如果是DP方法，计算噪声尺度
        sigma = 0
        if 'dp' in self.args.method or self.args.method == 'our_method':
            if hasattr(server, 'compute_noise_scale'):
                sigma = server.compute_noise_scale(
                    self.args.target_epsilon,
                    self.args.target_delta,
                    self.args.global_epochs,
                    self.args.num_selected,
                    self.args.num_clients
                )
                print(f"DP噪声尺度: {sigma:.4f}")
                print(f"隐私预算: ε={self.args.target_epsilon}, δ={self.args.target_delta}")
        
        # 进度条
        pbar = tqdm(range(self.args.global_epochs), desc="全局训练轮次")
        
        for round_idx in pbar:
            current_global_lr = server.get_current_lr()
            # 选择客户端
            if self.args.method in ['our_method', 'our_method_no_dp']:
                selected_clients = server.select_clients(
                    self.args.num_selected,
                    self.args.shapley_weight,
                    self.args.diversity_weight,
                    self.args.participation_weight
                )
            else:
                # 其他方法随机选择
                selected_clients = np.random.choice(
                    list(range(self.args.num_clients)),
                    self.args.num_selected,
                    replace=False
                ).tolist()
            
            # 客户端本地训练
            client_updates = {}
            client_accuracies = []
            client_losses = []
            grad_norms = []
            
            for client_id in selected_clients:
                client = clients[client_id]
                
                # 准备训练参数
                train_params = {
                    'global_model': server.global_model,
                    'local_epochs': self.args.local_epochs,
                    'lr': current_global_lr,
                    'momentum': self.args.momentum,
                    'weight_decay': self.args.weight_decay
                }
                
                # 方法特定参数
                if self.args.method == 'dp_fedavg':
                    train_params['clip_norm'] = self.args.clip_init
                    train_params['sigma'] = sigma
                
                elif self.args.method == 'ditto':
                    train_params['lambda_param'] = self.args.ditto_lambda
                
                elif self.args.method == 'fedaddp':
                    train_params.update({
                        'fisher_threshold': self.args.fisher_threshold,
                        'lambda_1': self.args.lambda_1,
                        'lambda_2': self.args.lambda_2,
                        'beta': self.args.beta,
                        'sigma0': sigma,
                        'clipping_bound': self.args.clip_init,
                        'no_clip': False,
                        'no_noise': False
                    })
                
                elif self.args.method in ['our_method', 'our_method_no_dp']:
                    add_dp = (self.args.method == 'our_method')
                    shapley_value = getattr(server, 'shapley_values', {}).get(client_id, 0.1)
                    
                    train_params.update({
                        'shapley_value': shapley_value,
                        'round_idx': round_idx,
                        'use_pseudo': self.args.use_pseudo,
                        'use_adaptive_clip': self.args.use_adaptive_clip,
                        'add_dp_noise': add_dp,
                        'sigma': sigma if add_dp else 0,
                        'f_param': self.args.f_param,
                        'u_param': self.args.u_param
                    })
                
                # 本地训练
                result = client.local_train(**train_params)
                client_updates[client_id] = result
                
                # 记录结果
                client_accuracies.append(result['accuracy'])
                client_losses.append(result['loss'])
                grad_norms.append(result.get('grad_norm', 0))
                
            # 聚合客户端更新
            if self.args.method in ['our_method', 'our_method_no_dp']:
                # 计算聚合权重
                test_accuracies = {cid: info['accuracy'] for cid, info in client_updates.items()}
                aggregation_weights = server.compute_aggregation_weights(
                    selected_clients,
                    test_accuracies,
                    self.args.shapley_weight,
                    self.args.diversity_weight,
                    self.args.participation_weight
                )
                
                server.aggregate(client_updates, aggregation_weights,selected_clients)
            else:
                # 平均聚合
                server.aggregate(client_updates)
            
            # 测试全局模型
            global_acc = server.test_global_model(test_loader)
            
            # 记录结果
            self.results['round_accuracies'].append(global_acc)
            self.results['client_accuracies'].append(client_accuracies)
            self.results['training_losses'].append(np.mean(client_losses) if client_losses else 0)
            self.results['gradient_norms'].append(np.mean(grad_norms) if grad_norms else 0)
            
            # 计算公平性方差
            if client_accuracies:
                fairness_var = np.var(client_accuracies)
                self.results['fairness_variances'].append(fairness_var)
            
            # 更新进度条
            if (round_idx + 1) % 5 == 0 or round_idx == 0:
                pbar.set_postfix({
                    '全局准确率': f'{global_acc:.2f}%',
                    '客户端准确率': f'{np.mean(client_accuracies):.2f}%',
                    '学习率': f'{current_global_lr:.4f}',
                    '公平性方差': f'{fairness_var:.4f}' if client_accuracies else '0.0000'
                })
            
            # 详细日志
            if (round_idx + 1) % 10 == 0:
                print(f"\n轮次 {round_idx+1}/{self.args.global_epochs}")
                print(f"全局准确率: {global_acc:.2f}%")
                print(f"客户端准确率: 平均={np.mean(client_accuracies):.2f}%, "
                      f"范围=[{min(client_accuracies):.2f}%, {max(client_accuracies):.2f}%]")
                print(f"公平性方差: {fairness_var:.4f}")
                print(f"平均训练损失: {np.mean(client_losses):.4f}")
                print("-" * 50)
        
        pbar.close()
        
        # 最终统计
        self._compute_final_statistics()
        
        return self.results
    
    def _compute_final_statistics(self):
        """计算最终统计信息"""
        if not self.results['round_accuracies']:
            return
        
        self.results['final_global_accuracy'] = self.results['round_accuracies'][-1]
        self.results['best_global_accuracy'] = max(self.results['round_accuracies'])
        self.results['mean_client_accuracy'] = np.mean([
            np.mean(accs) for accs in self.results['client_accuracies']
        ])
        self.results['mean_fairness_variance'] = np.mean(self.results['fairness_variances'])
        self.results['mean_training_loss'] = np.mean(self.results['training_losses'])
        self.results['mean_gradient_norm'] = np.mean(self.results['gradient_norms'])
        
        # 收敛速度（达到95%最大准确率的轮次）
        best_acc = self.results['best_global_accuracy']
        target_acc = best_acc * 0.95
        
        convergence_round = None
        for i, acc in enumerate(self.results['round_accuracies']):
            if acc >= target_acc:
                convergence_round = i + 1
                break
        
        self.results['convergence_round'] = convergence_round
        self.results['convergence_speed'] = convergence_round / self.args.global_epochs if convergence_round else 1.0
    
    def save_results(self, filename=None):
        """保存结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.args.method}_{self.args.dataset}_{timestamp}.json"
        
        filepath = os.path.join(self.args.log_dir, filename)
        
        # 转换为可序列化的格式
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, np.generic):
                serializable_results[key] = value.item()
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                serializable_results[key] = [v.tolist() for v in value]
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"结果已保存到: {filepath}")
        print(f"\n实验摘要:")
        print(f"最终全局准确率: {self.results.get('final_global_accuracy', 0):.2f}%")
        print(f"最佳全局准确率: {self.results.get('best_global_accuracy', 0):.2f}%")
        print(f"平均客户端准确率: {self.results.get('mean_client_accuracy', 0):.2f}%")
        print(f"平均公平性方差: {self.results.get('mean_fairness_variance', 0):.4f}")
        print(f"收敛轮次: {self.results.get('convergence_round', 'N/A')}")
        print(f"{'='*60}")
        
        return filepath
    
    def print_comparison(self, other_results=None):
        """打印比较结果"""
        print(f"\n{'='*60}")
        print(f"{self.args.method} 性能总结:")
        print(f"{'='*60}")
        
        metrics = [
            ('最终全局准确率', 'final_global_accuracy', '%', 2),
            ('最佳全局准确率', 'best_global_accuracy', '%', 2),
            ('平均客户端准确率', 'mean_client_accuracy', '%', 2),
            ('平均公平性方差', 'mean_fairness_variance', '', 4),
            ('收敛速度', 'convergence_speed', '', 3),
            ('平均训练损失', 'mean_training_loss', '', 4)
        ]
        
        for name, key, unit, precision in metrics:
            value = self.results.get(key, 0)
            if unit == '%':
                print(f"{name}: {value:.{precision}f}{unit}")
            else:
                print(f"{name}: {value:.{precision}f}")
        
        if other_results:
            print(f"\n{'='*60}")
            print("与其他方法比较:")
            print(f"{'='*60}")
            
            for method, results in other_results.items():
                print(f"\n{method}:")
                for name, key, unit, precision in metrics:
                    value = results.get(key, 0)
                    if unit == '%':
                        print(f"  {name}: {value:.{precision}f}{unit}")
                    else:
                        print(f"  {name}: {value:.{precision}f}")