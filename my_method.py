# our_method.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import copy
from utils import Utils
from ala import ALA
import numpy as np
class OurMethodClient:
    """你的方法：客户端"""
    
    def __init__(self, client_id: int, 
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 data_distribution: List[int],
                 device: torch.device):
        """
        初始化客户端
        
        Args:
            client_id: 客户端ID
            model: 模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            data_distribution: 数据分布
            device: 设备
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.data_distribution = data_distribution
        self.device = device
        
        # 历史信息
        self.last_grad_norm = 0.0
        self.clip_norm = 1.0  # 初始裁剪阈值
        self.selected_count = 0
        
        # ALA模块
        self.ala_module = None
    
    def set_ala_module(self, ala_module: ALA):
        """设置ALA模块"""
        self.ala_module = ala_module
    
    def local_train(self, 
                   global_model: nn.Module,
                   shapley_value: float,
                   round_idx: int,
                   use_pseudo: bool,
                   use_adaptive_clip: bool,
                   add_dp_noise: bool,
                   sigma: float,
                   f_param: float,
                   u_param: float,
                   local_epochs: int,
                   lr: float,
                   momentum: float,
                   weight_decay: float) -> Tuple[Dict[str, torch.Tensor], float, float]:
        """
        本地训练
        
        Returns:
            model_state: 模型状态字典
            accuracy: 测试准确率
            grad_norm: 梯度范数
        """
        # 加载全局模型参数
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(self.device)
        self.model.train()
        
        # 优化器
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # 记录梯度
        grad_norms = []
        
        for epoch in range(local_epochs):
            epoch_loss = 0
            for batch_idx, (data, labels) in enumerate(self.train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                
                # 伪标签训练（如果启用）
                pseudo_ratio = 0
                if use_pseudo and batch_idx % 2 == 0:  # 每隔一个批次使用伪标签
                    pseudo_ratio = Utils.pseudo_label_training(
                        self.model, self.train_loader, self.device
                    )
                
                # 反向传播
                loss.backward()
                
                # 计算梯度范数
                grad_norm = Utils.clip_gradients(self.model, self.clip_norm)
                grad_norms.append(grad_norm)
                
                # 自适应裁剪（如果启用）
                if use_adaptive_clip and len(grad_norms) >= 2:
                    last_grad = self.last_grad_norm if self.last_grad_norm > 0 else grad_norm
                    self.clip_norm = Utils.adaptive_clipping(
                        grad_norm, last_grad, shapley_value, 
                        self.clip_norm, f_param, u_param
                    )
                
                # 添加DP噪声（如果启用）
                if add_dp_noise:
                    Utils.add_dp_noise(self.model, self.clip_norm, sigma)
                
                # 更新参数
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 更新最后梯度范数
            if grad_norms:
                self.last_grad_norm = grad_norms[-1]
        
        # ALA模块（如果启用）
        if self.ala_module is not None:
            self.ala_module.adaptive_local_aggregation(global_model, self.model)
        
        # 计算测试准确率
        accuracy = self.test()
        
        # 计算模型更新
        model_update = {}
        for name, param in self.model.state_dict().items():
            model_update[name] = param - global_model.state_dict()[name]
        
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
        
        return model_update, accuracy, avg_grad_norm
    
    def test(self) -> float:
        """测试模型"""
        self.model.eval()
        self.model.to(self.device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0
        return accuracy


class OurMethodServer:
    """你的方法：服务器"""
    
    def __init__(self, 
                 global_model: nn.Module,
                 num_clients: int,
                 device: torch.device):
        """
        初始化服务器
        
        Args:
            global_model: 全局模型
            num_clients: 客户端数量
            device: 设备
        """
        self.global_model = global_model
        self.num_clients = num_clients
        self.device = device
        
        # 记录信息
        self.shapley_values = {i: 1.0/num_clients for i in range(num_clients)}
        self.data_diversities = {i: 0.0 for i in range(num_clients)}
        self.participation_counts = {i: 0 for i in range(num_clients)}
        self.client_grad_norms = {i: 0.0 for i in range(num_clients)}
        
        # 数据分布信息（需要在初始化后设置）
        self.data_distributions = None
    
    def set_data_distributions(self, data_distributions: Dict[int, List[int]]):
        """设置数据分布"""
        self.data_distributions = data_distributions
        # 计算初始数据多样性
        self.data_diversities = Utils.compute_data_diversity(data_distributions)
    
    def select_clients(self, 
                      num_select: int,
                      shapley_weight: float,
                      diversity_weight: float,
                      participation_weight: float) -> List[int]:
        """
        选择客户端
        
        Args:
            num_select: 选择数量
            weights: 权重分配
            
        Returns:
            selected_clients: 选中的客户端
        """
        # 计算参与频率
        frequencies = Utils.compute_participation_frequency(
            self.participation_counts, 
            sum(self.participation_counts.values())
        )
        
        # 战略选择客户端
        selected_clients = Utils.select_clients_strategic(
            self.shapley_values,
            self.data_diversities,
            frequencies,
            num_select,
            shapley_weight,
            diversity_weight,
            participation_weight
        )
        
        # 更新参与次数
        for client_id in selected_clients:
            self.participation_counts[client_id] += 1
        
        return selected_clients
    
    def compute_aggregation_weights(self,
                                  selected_clients: List[int],
                                  test_accuracies: Dict[int, float],
                                  shapley_weight: float,
                                  diversity_weight: float,
                                  participation_weight: float) -> Dict[int, float]:
        """
        计算聚合权重
        
        Args:
            selected_clients: 选中的客户端
            test_accuracies: 测试准确率
            weights: 权重分配
            
        Returns:
            aggregation_weights: 聚合权重
        """
        # 更新Shapley值（基于最新准确率）
        self.shapley_values = Utils.compute_shapley_contributions(
            {cid: [torch.zeros(1)] for cid in selected_clients},  # 这里简化计算
            test_accuracies
        )
        
        # 计算参与频率
        frequencies = Utils.compute_participation_frequency(
            self.participation_counts,
            sum(self.participation_counts.values())
        )
        
        # 计算聚合权重
        aggregation_weights = Utils.compute_aggregation_weights(
            self.shapley_values,
            self.data_diversities,
            frequencies,
            selected_clients,
            shapley_weight,
            diversity_weight,
            participation_weight
        )
        
        return aggregation_weights
    
    def aggregate(self,
                 client_updates: Dict[int, Dict[str, torch.Tensor]],
                 aggregation_weights: Dict[int, float]) -> None:
        """
        聚合客户端更新
        
        Args:
            client_updates: 客户端更新
            aggregation_weights: 聚合权重
        """
        # 初始化全局模型更新
        global_update = {}
        for name in self.global_model.state_dict().keys():
            global_update[name] = torch.zeros_like(self.global_model.state_dict()[name])
        
        # 加权聚合
        for client_id, update in client_updates.items():
            weight = aggregation_weights.get(client_id, 0)
            if weight > 0:
                for name in update.keys():
                    if update[name] is not None:
                        global_update[name] += weight * update[name]
        
        # 更新全局模型
        current_state = self.global_model.state_dict()
        new_state = {}
        for name in current_state.keys():
            new_state[name] = current_state[name] + global_update[name]
        
        self.global_model.load_state_dict(new_state)
    
    def compute_noise_scale(self, 
                          target_epsilon: float,
                          target_delta: float,
                          num_rounds: int,
                          num_selected: int) -> float:
        """
        计算噪声尺度
        
        Args:
            target_epsilon: 目标隐私预算
            target_delta: 目标δ
            num_rounds: 总轮数
            num_selected: 每轮选择客户端数
            
        Returns:
            sigma: 噪声尺度
        """
        # 每轮分配的隐私预算
        epsilon_per_round = target_epsilon / num_rounds
        
        # 简化计算，实际应根据隐私会计计算
        q = num_selected / self.num_clients  # 采样率
        sigma = (q * np.sqrt(2 * np.log(1.25 / target_delta))) / epsilon_per_round
        
        return sigma