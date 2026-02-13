# config.py
import argparse
import torch
from typing import Dict, Any, Optional
import random
import numpy as np

class Config:
    """配置类，避免默认值冲突"""
    
    def __init__(self):
        self._original_argv = None
        
    def parse_args(self, args_list: Optional[list] = None):
        """解析参数，可以传入参数列表或使用命令行参数"""
        parser = argparse.ArgumentParser(description="联邦学习实验配置")
        
        # ==================== 实验模式 ====================
        parser.add_argument('--mode', type=str, default='full_experiment',
                           choices=['quick_test', 'ablation', 'comparison', 'full_experiment'],
                           help='实验模式')
        parser.add_argument('--experiment_type', type=str, default=None,
                           choices=['ablation_study', 'adaptive_clip_comparison', 
                                   'aggregation_comparison', 'final_comparison'],
                           help='实验类型')
        parser.add_argument('--method', type=str, default=None,
                           choices=['fedavg', 'dp_fedavg', 'ditto', 'fedaddp', 
                                   'fedala', 'our_method', 'our_method_no_dp'],
                           help='联邦学习方法')
        
        # ==================== 数据参数 ====================
        parser.add_argument('--dataset', type=str, default='MNIST',
                           choices=['CIFAR10', 'FEMNIST', 'SVHN', 'Synthetic', 'MNIST'],
                           help='数据集')
        parser.add_argument('--num_clients', type=int, default=20, help='客户端总数')
        parser.add_argument('--samples_per_client', type=int, default=1000, help='每个客户端样本数')
        parser.add_argument('--test_samples_per_client', type=int, default=500, 
                           help='每个客户端测试样本数')
        
        # 非IID参数
        parser.add_argument('--iid', action='store_true', help='是否使用IID数据')
        parser.add_argument('--dir_alpha', type=float, default=0.3, 
                           help='Dirichlet分布参数，越小非IID程度越高')
        parser.add_argument('--data_skew_type', type=str, default='label_distribution',
                           choices=['label_distribution', 'quantity', 'feature'],
                           help='数据倾斜类型')
        
        # ==================== 联邦学习参数 ====================
        parser.add_argument('--num_selected', type=int, default=10, help='每轮选择的客户端数')
        parser.add_argument('--global_epochs', type=int, default=100, help='全局训练轮数')
        parser.add_argument('--local_epochs', type=int, default=5, help='本地训练轮数')
        parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
        
        # 优化器参数
        parser.add_argument('--lr', type=float, default=0.01, help='学习率')
        parser.add_argument('--momentum', type=float, default=0.9, help='动量')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
        
        # ==================== 隐私参数 ====================
        parser.add_argument('--target_epsilon', type=float, default=2.0, help='总隐私预算ε')
        parser.add_argument('--target_delta', type=float, default=1e-5, help='隐私参数δ')
        parser.add_argument('--clip_init', type=float, default=1.0, help='初始裁剪阈值')
        
        # ==================== 你的方法专用参数 ====================
        parser.add_argument('--shapley_weight', type=float, default=0.5, 
                           help='Shapley值在客户端选择中的权重')
        parser.add_argument('--diversity_weight', type=float, default=0.3,
                           help='数据多样性在客户端选择中的权重')
        parser.add_argument('--participation_weight', type=float, default=0.2,
                           help='参与频率在客户端选择中的权重')
        parser.add_argument('--f_param', type=float, default=0.8,
                           help='自适应裁剪的f参数')
        parser.add_argument('--u_param', type=float, default=0.5,
                           help='自适应裁剪的u参数')
        parser.add_argument('--use_ala', action='store_true', 
                           help='是否使用ALA模块')
        parser.add_argument('--use_pseudo', action='store_true',
                           help='是否使用伪标签训练')
        parser.add_argument('--use_adaptive_clip', action='store_true',
                           help='是否使用自适应裁剪')
        
        # ==================== FedADDP参数 ====================
        parser.add_argument('--fisher_threshold', type=float, default=0.1,
                           help='Fisher阈值')
        parser.add_argument('--lambda_1', type=float, default=0.01,
                           help='FedADDP λ1正则化参数')
        parser.add_argument('--lambda_2', type=float, default=0.01,
                           help='FedADDP λ2正则化参数')
        parser.add_argument('--beta', type=float, default=0.1,
                           help='FedADDP β参数')
        
        # ==================== FedALA参数 ====================
        parser.add_argument('--rand_percent', type=int, default=50,
                           help='随机采样百分比')
        parser.add_argument('--layer_idx', type=int, default=1,
                           help='层索引')
        parser.add_argument('--eta', type=float, default=1.0,
                           help='权重学习率')
        parser.add_argument('--threshold', type=float, default=0.1,
                           help='训练阈值')
        parser.add_argument('--num_pre_loss', type=int, default=10,
                           help='预损失数量')
        
        # ==================== Ditto参数 ====================
        parser.add_argument('--ditto_lambda', type=float, default=0.1,
                           help='Ditto正则化系数')
        
        # ==================== 设备参数 ====================
        parser.add_argument('--device', type=str, default='cuda',
                           choices=['cuda', 'cpu', 'cuda:0', 'cuda:1'],
                           help='设备')
        parser.add_argument('--seed', type=int, default=42,
                           help='随机种子')
        
        # ==================== 实验记录 ====================
        parser.add_argument('--log_dir', type=str, default='./logs',
                           help='日志目录')
        parser.add_argument('--save_model', action='store_true',
                           help='是否保存模型')
        parser.add_argument('--verbose', type=int, default=1,
                           choices=[0, 1, 2],
                           help='日志详细程度 0:静默 1:正常 2:详细')
        parser.add_argument('--visualize', action='store_true',
                           help='是否生成可视化图表')
        parser.add_argument('--save_plots', action='store_true',
                           help='是否保存图表')
        parser.add_argument('--plot_dir', type=str, default='./plots',
                           help='图表保存目录')
        
        # ==================== 快速测试参数 ====================
        parser.add_argument('--quick_test_clients', type=int, default=10,
                           help='快速测试客户端数')
        parser.add_argument('--quick_test_epochs', type=int, default=10,
                           help='快速测试全局轮数')
        parser.add_argument('--quick_test_samples', type=int, default=100,
                           help='快速测试每个客户端样本数')
        
        # 清空命令行参数，使用传入的参数列表
        if args_list is not None:
            import sys
            self._original_argv = sys.argv
            sys.argv = [sys.argv[0]] + args_list
            
        args = parser.parse_args()
        
        # 恢复原始命令行参数
        if self._original_argv is not None:
            sys.argv = self._original_argv
        
        # 设置设备
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            print("警告: CUDA不可用，使用CPU")
            args.device = torch.device("cpu")
        else:
            args.device = torch.device(args.device)
        
        # 设置随机种子
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        
        # 快速测试模式调整参数
        if args.mode == 'quick_test':
            args.num_clients = args.quick_test_clients
            args.global_epochs = args.quick_test_epochs
            args.samples_per_client = args.quick_test_samples
            args.num_selected = max(2, args.num_clients // 5)
            args.verbose = 2
        
        return args
    
    def create_from_dict(self, config_dict: Dict[str, Any]):
        """从字典创建配置"""
        args_list = []
        for key, value in config_dict.items():
            if key.startswith('_'):
                continue
            if isinstance(value, bool):
                if value:
                    args_list.append(f'--{key}')
            elif value is not None:
                args_list.append(f'--{key}')
                args_list.append(str(value))
        return self.parse_args(args_list)
    
    def get_base_config(self):
        """获取基础配置（不包含方法特定的默认值）"""
        base_config = {
            'mode': 'full_experiment',
            'dataset': 'MNIST',
            'num_clients': 20,
            'samples_per_client': 1000,
            'test_samples_per_client': 500,
            'iid': False,
            'dir_alpha': 0.3,
            'data_skew_type': 'label_distribution',
            'num_selected': 10,
            'global_epochs': 100,
            'local_epochs': 10,
            'batch_size': 32,
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'target_epsilon': 2.0,
            'target_delta': 1e-5,
            'clip_init': 1.0,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'seed': 42,
            'log_dir': './logs',
            'save_model': False,
            'verbose': 1,
            'visualize': True,
            'save_plots': True,
            'plot_dir': './plots',
            'quick_test_clients': 10,
            'quick_test_epochs': 10,
            'quick_test_samples': 100,
        }
        return base_config


# 创建全局配置实例
_config = Config()

def get_config():
    """获取命令行配置（向后兼容）"""
    return _config.parse_args()

def create_config_from_dict(config_dict: Dict[str, Any]):
    """从字典创建配置（用于实验配置）"""
    return _config.create_from_dict(config_dict)