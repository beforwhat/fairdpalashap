# fedala_module.py
import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Dict
from ala import ALA
from utils import Utils
class FedALAClient:
    """FedALA客户端"""
    
    def __init__(self, client_id: int, model: nn.Module,
                 train_loader, test_loader, device,
                 train_data, batch_size, rand_percent,
                 layer_idx, eta, threshold, num_pre_loss):
        """
        初始化FedALA客户端
        
        Args:
            client_id: 客户端ID
            model: 模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            device: 设备
            train_data: 训练数据（用于ALA）
            其他参数同ALA模块
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # ALA模块
        self.ala_module = ALA(
            cid=client_id,
            loss_function=nn.CrossEntropyLoss(),
            train_data=train_data,
            batch_size=batch_size,
            rand_percent=rand_percent,
            layer_idx=layer_idx,
            eta=eta,
            device=device,
            threshold=threshold,
            num_pre_loss=num_pre_loss
        )
        self.clip_norm = 1.0  # 初始裁剪阈值
        # 历史信息
        self.ala_weights_history = []
    
    def local_train(self, global_model: nn.Module,
                    local_epochs: int, lr: float,
                    momentum: float, weight_decay: float) -> Dict:
        """
        FedALA本地训练
        
        Returns:
            dict: 训练结果
        """
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
        
        # 本地训练
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
                Utils.clip_gradients_by_value(
                    self.model, 
                    clip_val=self.clip_norm    # 使用指定分位数作为裁剪阈值
                )
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                epoch_losses.append(avg_loss)
        
        # 应用ALA模块
        self.ala_module.adaptive_local_aggregation(global_model, self.model)
        
        # 记录ALA权重
        if self.ala_module.weights is not None:
            weight_mean = np.mean([w.mean().item() for w in self.ala_module.weights])
            self.ala_weights_history.append(weight_mean)
        
        # 计算测试准确率
        accuracy = self.test()
        
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
            'grad_norm': 0,
            'ala_weight': self.ala_weights_history[-1] if self.ala_weights_history else 0
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