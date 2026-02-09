# main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import json
from datetime import datetime
import time
from typing import Dict, List

# 导入自定义模块
from config import get_config
from experiment_config import get_experiment_config
from experiment_framework import ExperimentFramework
from data.data_loader import DatasetLoader
from models.models import ModelFactory
from visualization.visualization import ResultVisualizer


def setup_environment(args):
    """设置实验环境"""
    print("="*80)
    print("联邦学习实验系统")
    print("="*80)
    
    # 创建必要的目录
    os.makedirs(args.log_dir, exist_ok=True)
    if args.visualize or args.save_plots:
        os.makedirs(args.plot_dir, exist_ok=True)
    
    # 打印配置信息
    print(f"实验模式: {args.mode}")
    print(f"数据集: {args.dataset}")
    print(f"客户端数量: {args.num_clients}")
    print(f"每轮选择客户端数: {args.num_selected}")
    print(f"全局训练轮数: {args.global_epochs}")
    print(f"本地训练轮数: {args.local_epochs}")
    print(f"数据分布: {'IID' if args.iid else 'Non-IID (α=' + str(args.dir_alpha) + ')'}")
    print(f"设备: {args.device}")
    print(f"随机种子: {args.seed}")
    print("-"*80)


def load_data_and_models(args):
    """加载数据和模型"""
    print("正在加载数据和模型...")
    
    # 创建数据加载器
    data_loader = DatasetLoader(args.dataset)
    dataset_params = data_loader.get_dataset_params()
    
    # 加载完整数据集
    print(f"加载 {args.dataset} 数据集...")
    data, labels = data_loader.load_full_dataset()
    
    print(f"数据集信息:")
    print(f"  总样本数: {len(data)}")
    print(f"  输入维度: {dataset_params['input_dim']}")
    print(f"  类别数: {dataset_params['num_classes']}")
    print(f"  通道数: {dataset_params['channels']}")
    
    # 分配数据给客户端
    print(f"\n分配数据给 {args.num_clients} 个客户端...")
    if args.iid:
        train_loaders, test_loaders, data_distributions = data_loader.distribute_data_iid(
            data, labels, args.num_clients, args.samples_per_client
        )
        print("数据分配方式: IID (独立同分布)")
    else:
        train_loaders, test_loaders, data_distributions = data_loader.distribute_data_non_iid(
            data, labels, args.num_clients, args.samples_per_client,
            args.dir_alpha, args.data_skew_type
        )
        print(f"数据分配方式: Non-IID (Dirichlet α={args.dir_alpha}, 类型={args.data_skew_type})")
    
    # 打印数据分布统计
    total_samples = 0
    class_counts_total = [0] * dataset_params['num_classes']
    
    for client_id, distribution in data_distributions.items():
        total_samples += sum(distribution)
        for class_id, count in enumerate(distribution):
            class_counts_total[class_id] += count
    
    print(f"数据分配统计:")
    print(f"  总样本数: {total_samples}")
    print(f"  每个客户端平均样本数: {total_samples / args.num_clients:.1f}")
    print(f"  类别分布: {class_counts_total}")
    
    # 创建全局测试集
    print("\n创建全局测试集...")
    global_test_loader = data_loader.create_global_test_loader()
    
    # 获取模型类
    def create_model():
        return ModelFactory.get_default_model(args.dataset)
    
    return {
        'train_loaders': train_loaders,
        'test_loaders': test_loaders,
        'global_test_loader': global_test_loader,
        'data_distributions': data_distributions,
        'create_model': create_model,
        'dataset_params': dataset_params
    }


def run_quick_test(args):
    """运行快速测试"""
    print("\n" + "="*80)
    print("快速测试模式")
    print("="*80)
    
    # 加载数据
    data_resources = load_data_and_models(args)
    
    # 创建实验框架
    framework = ExperimentFramework(args, data_resources['data_distributions'])
    
    # 初始化客户端和服务器
    clients, server = framework.initialize_clients_and_server(
        model_class=data_resources['create_model'],
        train_loaders=data_resources['train_loaders'],
        test_loaders=data_resources['test_loaders']
    )
    
    # 运行实验
    start_time = time.time()
    results = framework.run_experiment(
        clients, server, data_resources['global_test_loader']
    )
    elapsed_time = time.time() - start_time
    
    # 保存结果
    results_file = framework.save_results()
    
    print(f"\n快速测试完成!")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"平均每轮耗时: {elapsed_time/args.global_epochs:.2f} 秒")
    print(f"结果文件: {results_file}")
    
    # 可视化（如果启用）
    if args.visualize:
        visualizer = ResultVisualizer(args.plot_dir)
        results_dict = {args.method: results}
        
        visualizer.plot_accuracy_curves(results_dict, 'quick_test_accuracy.png')
        visualizer.generate_summary_table(results_dict, 'quick_test_summary.csv')
    
    return results


def run_ablation_study(args):
    """运行消融实验"""
    print("\n" + "="*80)
    print("消融实验")
    print("="*80)
    
    # 获取消融实验配置
    experiments = get_experiment_config('ablation_study')
    
    # 加载数据（所有实验共享数据）
    data_resources = load_data_and_models(args)
    
    # 运行所有实验
    all_results = {}
    
    for exp_name, exp_args in experiments:
        print(f"\n正在运行实验: {exp_name}")
        print(f"配置: ALA={exp_args.use_ala}, 伪标签={exp_args.use_pseudo}, "
              f"自适应裁剪={exp_args.use_adaptive_clip}")
        
        try:
            # 创建实验框架
            framework = ExperimentFramework(exp_args, data_resources['data_distributions'])
            
            # 初始化
            clients, server = framework.initialize_clients_and_server(
                model_class=data_resources['create_model'],
                train_loaders=data_resources['train_loaders'],
                test_loaders=data_resources['test_loaders']
            )
            
            # 运行实验
            results = framework.run_experiment(
                clients, server, data_resources['global_test_loader']
            )
            
            all_results[exp_name] = results
            
            # 保存结果
            results_file = framework.save_results()
            print(f"实验完成，结果保存至: {results_file}")
            
        except Exception as e:
            print(f"实验 {exp_name} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 间隔一下，避免输出混乱
        time.sleep(1)
    
    # 可视化比较
    if args.visualize and all_results:
        visualizer = ResultVisualizer(args.plot_dir)
        
        visualizer.plot_accuracy_curves(all_results, 'ablation_accuracy.png')
        visualizer.plot_fairness_comparison(all_results, 'ablation_fairness.png')
        visualizer.plot_convergence_speed(all_results, 'ablation_convergence.png')
        visualizer.generate_summary_table(all_results, 'ablation_summary.csv')
    
    return all_results


def run_comparison_experiment(args, experiment_type):
    """运行比较实验"""
    print(f"\n" + "="*80)
    print(f"{experiment_type.replace('_', ' ').title()} 实验")
    print("="*80)
    
    # 获取实验配置
    experiments = get_experiment_config(experiment_type)
    
    # 加载数据（所有实验共享数据）
    data_resources = load_data_and_models(args)
    
    # 运行所有实验
    all_results = {}
    
    for exp_name, exp_args in experiments:
        print(f"\n正在运行方法: {exp_name}")
        
        try:
            # 创建实验框架
            framework = ExperimentFramework(exp_args, data_resources['data_distributions'])
            
            # 初始化
            clients, server = framework.initialize_clients_and_server(
                model_class=data_resources['create_model'],
                train_loaders=data_resources['train_loaders'],
                test_loaders=data_resources['test_loaders']
            )
            
            # 运行实验
            results = framework.run_experiment(
                clients, server, data_resources['global_test_loader']
            )
            
            all_results[exp_name] = results
            
            # 保存结果
            results_file = framework.save_results()
            print(f"方法 {exp_name} 完成，结果保存至: {results_file}")
            
        except Exception as e:
            print(f"方法 {exp_name} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 间隔一下
        time.sleep(1)
    
    # 可视化比较
    if args.visualize and all_results:
        visualizer = ResultVisualizer(args.plot_dir)
        
        visualizer.plot_accuracy_curves(all_results, f'{experiment_type}_accuracy.png')
        visualizer.plot_fairness_comparison(all_results, f'{experiment_type}_fairness.png')
        visualizer.plot_convergence_speed(all_results, f'{experiment_type}_convergence.png')
        visualizer.plot_comprehensive_comparison(all_results, f'{experiment_type}_radar.png')
        visualizer.generate_summary_table(all_results, f'{experiment_type}_summary.csv')
        
        # 绘制数据分布
        if data_resources['data_distributions']:
            visualizer.plot_data_distribution(
                data_resources['data_distributions'],
                num_clients_to_show=min(20, args.num_clients),
                save_name='data_distribution.png'
            )
    
    return all_results


def main():
    """主函数"""
    # 获取配置
    args = get_config()
    
    # 设置环境
    setup_environment(args)
    
    # 根据模式运行实验
    if args.mode == 'quick_test':
        # 快速测试单个方法
        results = run_quick_test(args)
        all_results = {args.method: results}
    
    elif args.mode == 'ablation':
        # 消融实验
        all_results = run_ablation_study(args)
    
    elif args.mode == 'comparison':
        # 比较实验
        all_results = run_comparison_experiment(args, args.experiment_type)
    
    elif args.mode == 'full_experiment':
        # 完整实验：先消融，再对比
        print("\n" + "="*80)
        print("完整实验流程")
        print("="*80)
        
        # 保存原始参数
        original_args = args
        
        # 1. 消融实验
        print("\n阶段1: 消融实验")
        ablation_results = run_ablation_study(args)
        
        # 2. 最终对比实验
        print("\n阶段2: 最终对比实验")
        args.experiment_type = 'final_comparison'
        comparison_results = run_comparison_experiment(args, 'final_comparison')
        
        # 合并结果
        all_results = {**ablation_results, **comparison_results}
    
    else:
        raise ValueError(f"不支持的实验模式: {args.mode}")
    
    # 最终总结
    if all_results:
        print("\n" + "="*80)
        print("所有实验完成!")
        print("="*80)
        
        # 打印最佳方法
        best_method = None
        best_accuracy = 0
        
        for method_name, results in all_results.items():
            accuracy = results.get('final_global_accuracy', 0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_method = method_name
        
        if best_method:
            print(f"最佳方法: {best_method} (准确率: {best_accuracy:.2f}%)")
        
        # 生成最终报告
        if args.visualize:
            visualizer = ResultVisualizer(args.plot_dir)
            
            # 生成完整比较图表
            visualizer.plot_accuracy_curves(all_results, 'final_accuracy_comparison.png')
            visualizer.generate_summary_table(all_results, 'final_results_summary.csv')
            
            print(f"\n所有图表已保存至: {args.plot_dir}")
            print(f"所有结果已保存至: {args.log_dir}")
    
    print("\n实验系统执行完毕!")
    return all_results


if __name__ == "__main__":
    # 运行主程序
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n实验被用户中断")
    except Exception as e:
        print(f"\n实验执行出错: {str(e)}")
        import traceback
        traceback.print_exc()