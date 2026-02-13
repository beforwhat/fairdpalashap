# fedavg.py
import torch
import torch.nn as nn
import copy
from typing import Dict, List
import numpy as np
from lr_scheduler import CosineAnnealingLR
class FedAvgClient:
    """FedAvg客户端"""
    
    def __init__(self, client_id: int, model: nn.Module, 
                 train_loader, test_loader, device):
        """
        初始化FedAvg客户端
        
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
        
        # 历史信息
        self.train_losses = []
        self.test_accuracies = []
    
    def local_train(self, global_model: nn.Module, 
                    local_epochs: int, lr: float, 
                    momentum: float, weight_decay: float) -> Dict:
        """
        本地训练
        
        Returns:
            dict: 包含模型更新和准确率
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
        
        # 训练
        epoch_losses = []
        for epoch in range(local_epochs):
            epoch_loss = 0
            batch_count = 0
            
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            if batch_count > 0: 
                avg_loss = epoch_loss / batch_count
                epoch_losses.append(avg_loss)
        
        self.train_losses.extend(epoch_losses)
        
        # 计算测试准确率
        accuracy = self.test()
        self.test_accuracies.append(accuracy)
        
        # 计算模型更新
        model_update = {}
        global_state = global_model.state_dict()
        local_state = self.model.state_dict()
        
        for name in global_state.keys():
            model_update[name] = local_state[name] - global_state[name]
        
        return {
            'update': model_update,
            'accuracy': accuracy,
            'loss': np.mean(epoch_losses) if epoch_losses else 0,
            'grad_norm': 0  # FedAvg不记录梯度范数
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


class FedAvgServer:
    """FedAvg服务器"""
    
    def __init__(self, global_model: nn.Module, device: torch.device,client_data_sizes: Dict[int, int] = None):
        """
        初始化FedAvg服务器
        
        Args:
            global_model: 全局模型
            device: 设备
        """
        self.global_model = global_model
        self.device = device
        self.lr_scheduler = CosineAnnealingLR(
            initial_lr=0.01,      # 默认初始学习率
            total_epochs=100,     # 默认总轮数
            warmup_epochs=5       # 默认预热5轮
        )
        # 记录信息
        self.global_accuracies = []
        self.communication_rounds = 0
        self.client_data_sizes = client_data_sizes or {} 
    def get_current_lr(self) -> float:
        """获取当前学习率"""
        return self.lr_scheduler.get_lr()
    def aggregate(self, client_updates: Dict[int, Dict], 
                  client_weights: Dict[int, float] = None) -> None:
        """
        聚合客户端更新
        
        Args:
            client_updates: 客户端更新字典 {client_id: {'update': ...}}
            client_weights: 客户端权重，如果为None则平均
        """
        self.communication_rounds += 1
        
        if not client_updates:
            return
        
        selected_clients = list(client_updates.keys())
        
        # 如果没有指定权重，则平均
        if client_weights is None:
        # 获取选中客户端的数据量
           sizes = [self.client_data_sizes.get(cid, 1.0) for cid in selected_clients]
           total = sum(sizes)
           client_weights = {cid: size / total for cid, size in zip(selected_clients, sizes)}
        
        # 初始化全局更新
        global_update = {}
        for name in self.global_model.state_dict().keys():
            global_update[name] = torch.zeros_like(
                self.global_model.state_dict()[name]
            )
        
        # 加权聚合
        for client_id in selected_clients:
            if client_id in client_updates:
                update = client_updates[client_id].get('update', {})
                weight = client_weights.get(client_id, 0)
                
                if update and weight > 0:
                    for name, param_update in update.items():
                        if name in global_update:
                            global_update[name] += weight * param_update
        
        # 更新全局模型
        current_state = self.global_model.state_dict()
        new_state = {}
        for name in current_state.keys():
            if name in global_update:
                new_state[name] = current_state[name] + global_update[name]
            else:
                new_state[name] = current_state[name]
        
        self.global_model.load_state_dict(new_state)
        
        current_lr = self.lr_scheduler.step()
        # 每10轮打印一次学习率信息
        if self.communication_rounds % 10 == 0:
            print(f"轮次 {self.communication_rounds}: 全局学习率更新为 {current_lr:.6f}")
    
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