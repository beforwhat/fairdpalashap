import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Tuple, List
from ala import ALA

class FedShapleyClient:
    def __init__(self,
                 client_id: int,
                 local_model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 data_distribution: List[float],
                 device: torch.device,
                 local_epochs: int,
                 lr: float,
                 clip_threshold: float,
                 sigma: float,
                 target_epsilon: float,
                 target_delta: float):
        
        self.client_id = client_id
        self.local_model = local_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.data_distribution = data_distribution
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.clip_threshold = clip_threshold
        self.sigma = sigma
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        
        # 记录历史信息
        self.prev_gradient_norm = 0.0
        self.clip_history = []
        
        # ALA模块
        self.ala_module = None
        
    def local_train(self,
                   global_model: torch.nn.Module,
                   round_idx: int) -> Tuple[torch.nn.Module, float, float]:
        """
        本地训练，包含自适应裁剪和DP噪声
        """
        # 复制全局模型
        self.local_model.load_state_dict(global_model.state_dict())
        self.local_model.to(self.device)
        
        # 初始化ALA模块
        if self.ala_module is None:
            self.ala_module = ALA(
                cid=self.client_id,
                loss=nn.CrossEntropyLoss(),
                train_data=list(self.train_loader.dataset),
                batch_size=self.train_loader.batch_size,
                rand_percent=50,
                layer_idx=1,
                eta=1.0,
                device=self.device,
                threshold=0.1,
                num_pre_loss=10
            )
        
        # 自适应裁剪阈值计算
        if round_idx > 0:
            # 计算梯度变化趋势
            if self.prev_gradient_norm > 0:
                n_it = (self.prev_gradient_norm - self.prev_gradient_norm_prev) / self.prev_gradient_norm_prev
            else:
                n_it = 0
            
            # 自适应裁剪阈值公式
            f = 0.8
            u = 0.5
            # 这里SV_i暂时用上一次的梯度范数模拟
            SV_i = self.prev_gradient_norm if self.prev_gradient_norm > 0 else 1.0
            new_clip_threshold = self.clip_threshold * (1 + f * SV_i - u * n_it)
            self.clip_threshold = max(new_clip_threshold, 0.1)  # 设置最小值
        
        # 训练
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        gradient_norm = 0.0
        for epoch in range(self.local_epochs):
            for batch_idx, (data, labels) in enumerate(self.train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.local_model(data)
                
                # 计算真实损失
                true_loss = criterion(outputs, labels)
                
                # 伪标签训练（参考论文）
                with torch.no_grad():
                    pseudo_labels = torch.argmax(outputs, dim=1)
                pseudo_loss = criterion(outputs, pseudo_labels)
                
                # 总损失
                total_loss = true_loss + 0.5 * pseudo_loss
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪和添加噪声
                clipped_grads = self._clip_and_add_noise(optimizer)
                gradient_norm = torch.norm(torch.cat([g.view(-1) for g in clipped_grads]))
                
                # 更新参数
                for param, grad in zip(self.local_model.parameters(), clipped_grads):
                    param.data.add_(-self.lr * grad)
        
        # 保存梯度范数用于下一轮
        self.prev_gradient_norm_prev = self.prev_gradient_norm
        self.prev_gradient_norm = gradient_norm
        
        # ALA自适应聚合
        self.ala_module.adaptive_local_aggregation(global_model, self.local_model)
        
        # 计算准确率
        accuracy = self.test()
        
        return self.local_model, gradient_norm, accuracy
    
    def _clip_and_add_noise(self, optimizer) -> List[torch.Tensor]:
        """
        梯度裁剪和添加DP噪声
        """
        clipped_grads = []
        
        for param in self.local_model.parameters():
            if param.grad is not None:
                # 梯度裁剪
                grad_norm = torch.norm(param.grad)
                if grad_norm > self.clip_threshold:
                    scale_factor = self.clip_threshold / (grad_norm + 1e-10)
                    clipped_grad = param.grad * scale_factor
                else:
                    clipped_grad = param.grad.clone()
                
                # 添加高斯噪声
                if self.sigma > 0:
                    noise = torch.randn_like(clipped_grad) * self.sigma * self.clip_threshold
                    clipped_grad += noise
                
                clipped_grads.append(clipped_grad)
        
        return clipped_grads
    
    def test(self) -> float:
        """测试本地模型"""
        self.local_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.local_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        self.local_model.train()
        return accuracy
    
    def get_data_distribution(self) -> List[float]:
        """获取数据分布"""
        return self.data_distribution