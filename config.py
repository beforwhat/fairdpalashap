# config.py
import argparse
import torch

def get_config():
    """获取统一的实验配置参数"""
    parser = argparse.ArgumentParser(description="联邦学习实验配置")
    
    # ==================== 实验模式 ====================
    parser.add_argument('--mode', type=str, default='full_experiment',
                       choices=['quick_test', 'ablation', 'comparison', 'full_experiment'],
                       help='实验模式')
    parser.add_argument('--experiment_type', type=str, default='final_comparison',
                       choices=['ablation_study', 'adaptive_clip_comparison', 
                               'aggregation_comparison', 'final_comparison'],
                       help='实验类型')
    parser.add_argument('--method', type=str, default='our_method',
                       choices=['fedavg', 'dp_fedavg', 'ditto', 'fedaddp', 
                               'fedala', 'our_method', 'our_method_no_dp'],
                       help='联邦学习方法')
    
    # ==================== 数据参数 ====================
    parser.add_argument('--dataset', type=str, default='Synthetic',
                       choices=['CIFAR10', 'FEMNIST', 'SVHN', 'Synthetic', 'MNIST'],
                       help='数据集')
    parser.add_argument('--num_clients', type=int, default=100, help='客户端总数')
    parser.add_argument('--samples_per_client', type=int, default=500, help='每个客户端样本数')
    parser.add_argument('--test_samples_per_client', type=int, default=100, 
                       help='每个客户端测试样本数')
    
    # 非IID参数
    parser.add_argument('--iid', action='store_true', help='是否使用IID数据')
    parser.add_argument('--dir_alpha', type=float, default=0.5, 
                       help='Dirichlet分布参数，越小非IID程度越高')
    parser.add_argument('--data_skew_type', type=str, default='label_distribution',
                       choices=['label_distribution', 'quantity', 'feature'],
                       help='数据倾斜类型')
    
    # ==================== 联邦学习参数 ====================
    parser.add_argument('--num_selected', type=int, default=10, help='每轮选择的客户端数')
    parser.add_argument('--global_epochs', type=int, default=50, help='全局训练轮数')
    parser.add_argument('--local_epochs', type=int, default=5, help='本地训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    
    # 优化器参数
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    
    # ==================== 隐私参数 ====================
    parser.add_argument('--target_epsilon', type=float, default=1.0, help='总隐私预算ε')
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
    
    args = parser.parse_args()
    
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
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 快速测试模式调整参数
    if args.mode == 'quick_test':
        args.num_clients = args.quick_test_clients
        args.global_epochs = args.quick_test_epochs
        args.samples_per_client = args.quick_test_samples
        args.num_selected = max(2, args.num_clients // 5)
        args.verbose = 2  # 详细输出
        print(f"快速测试模式: {args.quick_test_clients}客户端, {args.quick_test_epochs}轮")
    
    return args