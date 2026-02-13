# baselines.py
"""
基线方法集成模块
统一导入所有基线方法，便于experiment调用
"""
import torch.nn as nn
# 导入各个基线方法
from fedavg import FedAvgClient, FedAvgServer
from dp_fedavg import DPFedAvgClient, DPFedAvgServer
from ditto import DittoClient, DittoServer
from fedaddp_module import FedADDPClient,FedADDPServer
from fedala_module import FedALAClient
from ala import ALA
from my_ala import MYALA
# 导入你的方法
from my_method import OurMethodClient, OurMethodServer

class BaselineFactory:
    """基线方法工厂类"""
    
    @staticmethod
    def create_client(method: str, client_id: int, model, train_loader, 
                     test_loader, device, **kwargs):
        """
        创建客户端实例
        
        Args:
            method: 方法名称
            client_id: 客户端ID
            model: 模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            device: 设备
            **kwargs: 方法特定参数
            
        Returns:
            client: 客户端实例
        """
        if method == 'fedavg':
            return FedAvgClient(client_id, model, train_loader, test_loader, device)
        
        elif method == 'dp_fedavg':
            return DPFedAvgClient(client_id, model, train_loader, test_loader, device)
        
        elif method == 'ditto':
            return DittoClient(client_id, model, train_loader, test_loader, device)
        
        elif method == 'fedaddp':
            # FedADDP需要额外参数
            return FedADDPClient(client_id, model, train_loader, test_loader, device)
        
        elif method == 'fedala':
            train_data = kwargs.get('train_data')
            if train_data is None:
                raise ValueError(f"[FedALA] 客户端 {client_id} 必须提供 train_data")
            # FedALA需要额外参数
            return FedALAClient(
                client_id=client_id,
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                train_data=kwargs.get('train_data'),
                batch_size=kwargs.get('batch_size', 32),
                rand_percent=kwargs.get('rand_percent', 50),
                layer_idx=kwargs.get('layer_idx', 1),
                eta=kwargs.get('eta', 1.0),
                threshold=kwargs.get('threshold', 0.1),
                num_pre_loss=kwargs.get('num_pre_loss', 10)
            )
        
        elif method in ['our_method', 'our_method_no_dp']:
            # 你的方法
            client = OurMethodClient(
                client_id, model, train_loader, test_loader, 
                kwargs.get('data_distribution', []), device
            )
            
            # 如果需要，设置ALA模块
            if kwargs.get('use_ala', False):
                train_data = kwargs.get('train_data')
                if train_data is None:
                    raise ValueError(f"[OurMethod] 客户端 {client_id} 需要 train_data 以使用ALA")
                ala_module = MYALA(
                    client_id=client_id,
                    loss_function=nn.CrossEntropyLoss(),
                    train_data=kwargs.get('train_data'),
                    batch_size=kwargs.get('batch_size', 32),
                    rand_percent=kwargs.get('rand_percent', 50),
                    layer_idx=kwargs.get('layer_idx', 1),
                    eta=kwargs.get('eta', 1.0),
                    device=device,
                    threshold=kwargs.get('threshold', 0.1),
                    num_pre_loss=kwargs.get('num_pre_loss', 10),
                    use_pseudo=kwargs.get('use_pseudo', False)
                )
                client.set_ala_module(ala_module)
            
            return client
        
        else:
            raise ValueError(f"未知的方法: {method}")
    
    @staticmethod
    def create_server(method: str, global_model, device, **kwargs):
        """
        创建服务器实例
        
        Args:
            method: 方法名称
            global_model: 全局模型
            device: 设备
            **kwargs: 方法特定参数
            
        Returns:
            server: 服务器实例
        """
        if method == 'fedavg':
            return FedAvgServer(global_model, device,kwargs.get('client_data_sizes', None))
        
        elif method == 'dp_fedavg':
            return DPFedAvgServer(global_model, device,kwargs.get('client_data_sizes', None))
        
        elif method == 'ditto':
            return DittoServer(global_model, device,kwargs.get('client_data_sizes', None))
        
        elif method =='fedala':
            # 对于这些方法，使用FedAvgServer作为基础
            return FedAvgServer(global_model, device,kwargs.get('client_data_sizes', None))
        elif method == 'fedaddp':
            # FedADDP需要额外参数
            return FedADDPServer(global_model, device)
        elif method in ['our_method', 'our_method_no_dp']:
            # 你的方法
            server = OurMethodServer(
                global_model=global_model,
                num_clients=kwargs.get('num_clients', 100),
                device=device
            )
            
            # 设置数据分布
            if 'data_distributions' in kwargs:
                server.set_data_distributions(kwargs['data_distributions'])
            
            return server
        
        else:
            raise ValueError(f"未知的方法: {method}")
    
    @staticmethod
    def get_method_params(method: str, args,sigma: float = 0,global_lr: float = None):
        """
        获取方法特定参数
        
        Args:
            method: 方法名称
            args: 配置参数
            
        Returns:
            dict: 方法参数
        """
        params = { 'local_epochs': args.local_epochs,
            'lr': global_lr if global_lr is not None else args.lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay}
        
        if method == 'dp_fedavg':
            params.update({
                'clip_norm': args.clip_init,
                'sigma': sigma  # 将在运行时计算
            })
        
        elif method == 'ditto':
            params.update({
                'lambda_param': args.ditto_lambda
            })
        
        elif method == 'fedaddp':
            params.update({
                'fisher_threshold': args.fisher_threshold,
                'lambda_1': args.lambda_1,
                'lambda_2': args.lambda_2,
                'beta': args.beta,
                'sigma0':  sigma,  # 将在运行时计算
                'clipping_bound': args.clip_init,
                'no_clip': False,
                'no_noise': False
            })
        
        elif method == 'fedala':
            params.update({
                'rand_percent': args.rand_percent,
                'layer_idx': args.layer_idx,
                'eta': args.eta,
                'threshold': args.threshold,
                'num_pre_loss': args.num_pre_loss
            })
        
        elif method in ['our_method', 'our_method_no_dp']:
            params.update({
                'shapley_weight': args.shapley_weight,
                'diversity_weight': args.diversity_weight,
                'participation_weight': args.participation_weight,
                'f_param': args.f_param,
                'u_param': args.u_param,
                'use_ala': args.use_ala,
                'use_pseudo': args.use_pseudo,
                'use_adaptive_clip': args.use_adaptive_clip
            })
            if method == 'our_method':
                params.update({
                    'add_dp_noise': True,
                    'sigma': sigma
                })
            else:
                params.update({
                    'add_dp_noise': False,
                    'sigma': 0
                })
        
        return params