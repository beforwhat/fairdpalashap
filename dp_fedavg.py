# dp_fedavg.py
import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Dict
from utils import Utils
from fedavg import FedAvgServer
from opacus.accountants.utils import get_noise_multiplier
class DPFedAvgClient:
    """DP-FedAvg客户端（带差分隐私）"""
    
    def __init__(self, client_id: int, model: nn.Module,
                 train_loader, test_loader, device,client_data_size):
        """
        初始化DP-FedAvg客户端
        
        Args:
            client_id: 客户端ID
            model: 模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            device: 设备
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.client_data_size = client_data_size
        # 历史信息
        self.grad_norms = []
        self.clip_norm = 1.0  # 初始裁剪阈值
        self.sensitivity = 1.0                     # 初始敏感度

    def local_train(self, global_model: nn.Module,
                    local_epochs: int, lr: float,
                    momentum: float, weight_decay: float,
                    clip_norm: float, sigma: float) -> Dict:
        """
        本地训练（带差分隐私）
        
        Args:
            clip_norm: 梯度裁剪阈值
            sigma: 噪声尺度
            
        Returns:
            dict: 包含模型更新、准确率和梯度范数
        """
        # 设置裁剪阈值
        
        # 加载全局模型参数
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(self.device)
        self.model.train()
        
        # 优化器
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # 训练
        epoch_losses = []
        grad_norms = []
        
        for epoch in range(local_epochs):
            epoch_loss = 0
            batch_count = 0
            
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                grad_norm = Utils.compute_norm(self.model)
                Utils.clip_gradients_by_value(
                    self.model, 
                    clip_val=self.clip_norm    # 使用指定分位数作为裁剪阈值
                )
                grad_norms.append(grad_norm)
               
                Utils.add_dp_noise(self.model, self.sensitivity, sigma,batch_size=data.size(0),device=self.device)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                epoch_losses.append(avg_loss)
        
        # 计算测试准确率
        accuracy = self.test()
        
        # 计算模型更新
        model_update = {}
        global_state = global_model.state_dict()
        local_state = self.model.state_dict()
        
        for name in global_state.keys():
            model_update[name] = local_state[name] - global_state[name]
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
        self.grad_norms.append(avg_grad_norm)
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        return {
            'update': model_update,
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


class DPFedAvgServer(FedAvgServer):
    """DP-FedAvg服务器"""
    
    def __init__(self, global_model: nn.Module, device: torch.device,samples_per_client,client_data_sizes: Dict[int, int]  ,batch_size: int,local_epochs: int):
        super().__init__(global_model, device,client_data_sizes)
        
        # DP相关参数
        self.privacy_spent = 0
        self.noise_scale = 0
        self.samples_per_client = samples_per_client
        self.local_epochs = local_epochs
        self.batch_size = batch_size
    def compute_noise_scale(self, target_epsilon: float, 
                           target_delta: float, 
                           global_epoch: int,
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
        alpha = np.ceil(np.log2(1 / target_delta) / target_epsilon + 1)

        temp = 2 * (target_epsilon + np.log(target_delta) / (alpha - 1))
        sigma_0 = np.sqrt(global_epoch * alpha / temp)
        
        self.noise_scale = sigma
        return sigma