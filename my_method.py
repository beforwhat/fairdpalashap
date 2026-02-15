# our_method.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import copy
from utils import Utils
from ala import ALA
import numpy as np
from lr_scheduler import CosineAnnealingLR
from opacus.accountants.utils import get_noise_multiplier
from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer 
from adaptiveclip import DualEMAClipper  
class OurMethodClient:
    """你的方法：客户端"""
    
    def __init__(self, client_id: int, 
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 data_distribution: List[int],
                 device: torch.device,use_pseudo:bool,local_epochs:int):
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
        self.use_pseudo = use_pseudo
        self.sensitivity = 1.0                     
        # 历史信息
        
        self.clip_norm = 0.5  # 初始裁剪阈值
        
        self.ala_module = None
        self.DualEMAClipper = DualEMAClipper(alpha_fast=0.5, alpha_slow=0.99, init_val=1.0)
    def set_ala_module(self, ala_module: ALA):
        """设置ALA模块"""
        self.ala_module = ala_module
         # my_method.py

    def local_train(self,
                global_model: nn.Module,
                shapley_value: float,
                use_adaptive_clip: bool,
                add_dp_noise: bool,
                sigma: float,
                f_param: float,
                u_param: float,
                local_epochs: int,
                lr: float,
                momentum: float,
                weight_decay: float) -> dict:

    # 加载全局模型参数
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(self.device)
        self.model.train()
       
        optimizer = torch.optim.Adam(
          self.model.parameters(),
          lr=lr
       )
        criterion = nn.CrossEntropyLoss()

        epoch_losses = []
        grad_norms = []

        for epoch in range(local_epochs):
           epoch_loss = 0
           batch_count = 0
           for batch_idx, (data, labels) in enumerate(self.train_loader):
               data, labels = data.to(self.device), labels.to(self.device)
               
               optimizer.zero_grad()
               outputs = self.model(data)
               loss = criterion(outputs, labels)
               loss.backward()
               grad_norm = Utils.compute_norm(self.model)
               grad_norms.append(grad_norm)
               use_adaptive_clip=True
               if use_adaptive_clip and len(grad_norms) >= 2:
                  clip_val = self.DualEMAClipper.update(grad_norm)
                  self.clip_norm = Utils.clip_gradients_by_value(
                     self.model, clip_val
                  )
                #   self.last_grad_norm = grad_norms[-1]
               if add_dp_noise:
                   
                   Utils.add_dp_noise(self.model, self.clip_norm, sigma,batch_size=data.size(0),device=self.device)
               optimizer.step()
               epoch_loss += loss.item()
               batch_count += 1

           if batch_count > 0:
              epoch_losses.append(epoch_loss / batch_count)

    # 更新梯度历史
        if grad_norms:
          self.last_grad_norm = grad_norms[-1]

    # ALA 模块（如果启用）
        if self.ala_module is not None:
          self.ala_module.adaptive_local_aggregation(global_model, self.model)

    # 测试准确率
        accuracy = self.test()

    # 计算模型更新（全局模型 → 本地模型的变化）
        model_update = {}
        global_state = global_model.state_dict()
        local_state = self.model.state_dict()
        for name in global_state.keys():
          model_update[name] = local_state[name] - global_state[name]
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0

        try:
          return {
          'update': model_update,
          'accuracy': accuracy,
          'loss': avg_loss,
          'grad_norm': avg_grad_norm
        }
        except Exception as e:
          print(f"⚠️ 生成客户端更新时发生错误: {e}")
          print(f"客户端 {self.client_id} 训练异常: {e}")
          return {
          'update':  {name: torch.zeros_like(p) for name, p in global_model.state_dict().items()},
          'accuracy': accuracy,
          'loss': avg_loss,
          'grad_norm': avg_grad_norm
        }
    
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
                 device: torch.device,samples_per_client: int,batch_size: int,local_epochs: int):
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
        self.lr_scheduler = CosineAnnealingLR(
            initial_lr=1e-3,      # 默认初始学习率
            total_epochs=100,     # 默认总轮数
            warmup_epochs=5       # 默认预热5轮
        )
        # 记录信息
        self.shapley_values = {i: 1.0/num_clients for i in range(num_clients)}
        self.data_diversities = {i: 0.0 for i in range(num_clients)}
        self.participation_counts = {i: 0 for i in range(num_clients)}
        self.client_grad_norms = {i: 0.0 for i in range(num_clients)}
        self.global_accuracies = []
        # 数据分布信息（需要在初始化后设置）
        self.data_distributions = None
        self.samples_per_client = samples_per_client
        self.batch_size = batch_size
        self.local_epochs = local_epochs
    def get_current_lr(self) -> float:
        """获取当前学习率（供客户端使用）"""
        return self.lr_scheduler.get_lr()
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
        
        full_test_accuracies = {i: test_accuracies.get(i, 0.0) for i in range(self.num_clients)}
        # 更新Shapley值（基于最新准确率）
        self.shapley_values = Utils.compute_shapley_contributions(
            {cid: [torch.zeros(1)] for cid in selected_clients},  
            full_test_accuracies
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
                 client_updates,
                 aggregation_weights: Dict[int, float],selected_clients: List[int]) -> None:
        """
        聚合客户端更新
        
        Args:
            client_updates: 客户端更新
            aggregation_weights: 聚合权重
        """
        
        # 初始化全局模型更新
        missing = [cid for cid in client_updates if 'update' not in client_updates[cid]]
        if missing:
           print(f"⚠️ 本轮缺失 update 的客户端: {missing}")
        global_update = {}
        for name in self.global_model.state_dict().keys():
            global_update[name] = torch.zeros_like(self.global_model.state_dict()[name])
        
        # 加权聚合
        for client_id, update in client_updates.items():
            weight = aggregation_weights.get(client_id, 0)
            if weight > 0:
                for name,param_update in update['update'].items():
                    if name in global_update:
                       global_update[name] += weight * param_update
                    else:
                      print(f"警告: 参数 '{name}' 不在全局模型中，已跳过 (客户端 {client_id})")
        
        # 更新全局模型
        current_state = self.global_model.state_dict()
        current_lr = self.lr_scheduler.step()
        new_state = {}
        for name in current_state.keys():
            new_state[name] = current_state[name] + global_update[name]
        
        self.global_model.load_state_dict(new_state)
    
    def compute_noise_scale(self, target_epsilon: float, 
                           target_delta: float, 
                           num_rounds: int,
                           num_selected: int,
                           total_clients: int) -> float:
        """
        计算噪声尺度
        
        Args:
            target_epsilon: 总隐私预算
            target_delta: 目标δ
            num_rounds: 总轮数
            num_selected: 每轮选择客户端数
            total_clients: 总客户端数
            
        Returns:
            sigma: 噪声尺度
        """
        # 每轮分配的隐私预算
        q = num_selected / total_clients
        
        # 计算每个客户端的步数（假设所有客户端样本数相同）
        steps_per_epoch = max(1, self.samples_per_client // self.batch_size)
        steps_per_round = self.local_epochs * steps_per_epoch
        total_steps = num_rounds * num_selected * steps_per_round

        # 使用 RDP accountant 求解 sigma
        sigma = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=q,
            steps=total_steps,
            accountant='rdp'   # 使用 RDP 会计
        )
        
        self.noise_scale = sigma
        return sigma
    def test_global_model(self, test_loader) -> float:
        """测试全局模型"""
        self.global_model.eval()
        self.global_model.to(self.device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0
        self.global_accuracies.append(accuracy)
        
        return accuracy