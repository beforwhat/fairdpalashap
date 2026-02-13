# ditto.py
import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Dict
from fedavg import FedAvgServer
class DittoClient:
    """Ditto客户端（个性化联邦学习）"""
    
    def __init__(self, client_id: int, model: nn.Module,
                 train_loader, test_loader, device):
        """
        初始化Ditto客户端
        
        Args:
            client_id: 客户端ID
            model: 模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            device: 设备
        """
        self.client_id = client_id
        self.global_model = copy.deepcopy(model)  # 全局模型副本
        self.personal_model = copy.deepcopy(model)  # 个性化模型
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # 历史信息
        self.global_losses = []
        self.personal_losses = []
    
    def local_train(self, global_model: nn.Module,
                    local_epochs: int, lr: float,
                    momentum: float, weight_decay: float,
                    lambda_param: float) -> Dict:
        """
        Ditto本地训练
        
        Args:
            lambda_param: 正则化系数
            
        Returns:
            dict: 包含全局模型更新和准确率
        """
        # 更新全局模型副本
        self.global_model.load_state_dict(global_model.state_dict())
        self.global_model.to(self.device)
        
        # 从全局模型初始化个性化模型
        self.personal_model.load_state_dict(global_model.state_dict())
        self.personal_model.to(self.device)
        self.personal_model.train()
        
        # 优化器（只优化个性化模型）
        optimizer = torch.optim.SGD(
            self.personal_model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # 训练个性化模型
        epoch_losses = []
        
        for epoch in range(local_epochs):
            epoch_loss = 0
            batch_count = 0
            
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.personal_model(data)
                
                # Ditto损失：本地损失 + λ * 正则化项
                loss_local = criterion(outputs, labels)
                
                # 正则化项：个性化模型与全局模型的差异
                reg_loss = 0
                for p_personal, p_global in zip(self.personal_model.parameters(),
                                               self.global_model.parameters()):
                    reg_loss += torch.norm(p_personal - p_global) ** 2
                
                loss = loss_local + lambda_param * reg_loss
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                epoch_losses.append(avg_loss)
        
        self.personal_losses.extend(epoch_losses)
        
        # 计算测试准确率（使用个性化模型）
        accuracy = self.test_personal()
        
        # 计算全局模型更新（个性化模型与全局模型的差异）
        model_update = {}
        global_state = global_model.state_dict()
        personal_state = self.personal_model.state_dict()
        
        for name in global_state.keys():
            model_update[name] = personal_state[name] - global_state[name]
        
        return {
            'update': model_update,
            'accuracy': accuracy,
            'loss': np.mean(epoch_losses) if epoch_losses else 0,
            'grad_norm': 0,
            'personal_accuracy': accuracy  # 个性化模型准确率
        }
    
    def test_personal(self) -> float:
        """测试个性化模型"""
        self.personal_model.eval()
        self.personal_model.to(self.device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.personal_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0
        return accuracy
    
    def test_global(self) -> float:
        """测试全局模型副本"""
        self.global_model.eval()
        self.global_model.to(self.device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0
        return accuracy


class DittoServer(FedAvgServer):
    """Ditto服务器"""
    
    def __init__(self, global_model: nn.Module, device: torch.device,client_data_sizes: Dict[int, int] = None):
        super().__init__(global_model, device,client_data_sizes)
        
        # Ditto特定记录
        self.personal_accuracies = []
    
    def aggregate(self, client_updates: Dict[int, Dict], 
                  client_weights: Dict[int, float] = None) -> None:
        """
        Ditto聚合（注意：这里聚合的是个性化模型与全局模型的差异）
        """
        super().aggregate(client_updates, client_weights)
        
        # 记录个性化模型准确率
        if client_updates:
            personal_accs = [info.get('personal_accuracy', 0) 
                           for info in client_updates.values()]
            if personal_accs:
                self.personal_accuracies.append(np.mean(personal_accs))