import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
from typing import List, Tuple, Any

class ALA:
    def __init__(self,
                cid: int,
                loss: nn.Module,
                train_data: List[Tuple[Any, Any]], 
                batch_size: int, 
                rand_percent: int, 
                layer_idx: int,
                eta: float,
                device: torch.device, 
                threshold: float,
                num_pre_loss: int) -> None:
        """
        Initialize ALA module
        
        Args:
            cid: Client ID
            loss: The loss function
            train_data: The reference of the local training data
            batch_size: Weight learning batch size
            rand_percent: The percent of the local training data to sample
            layer_idx: Control the weight range
            eta: Weight learning rate
            device: Using cuda or cpu
            threshold: Train the weight until the standard deviation of the recorded losses is less than a given threshold
            num_pre_loss: The number of the recorded losses to be considered to calculate the standard deviation
        """
        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None  # Learnable local aggregation weights
        self.start_phase = True
        self.loss_history = []  # 记录损失历史

    def adaptive_local_aggregation(self, 
                                  global_model: nn.Module,
                                  local_model: nn.Module) -> nn.Module:
        """
        Adaptive local aggregation with weight learning
        
        Args:
            global_model: The received global/aggregated model
            local_model: The trained local model
            
        Returns:
            Aggregated local model
        """
        # 如果没有足够的训练数据，直接返回
        if len(self.train_data) < self.batch_size:
            return local_model
        
        # 随机采样部分本地训练数据
        rand_ratio = self.rand_percent / 100.0
        rand_num = max(1, int(rand_ratio * len(self.train_data)))
        rand_num = min(rand_num, len(self.train_data))
        
        # 确保采样不会超出范围
        if rand_num >= len(self.train_data):
            rand_loader = DataLoader(self.train_data, self.batch_size, shuffle=True, drop_last=False)
        else:
            rand_idx = random.sample(range(len(self.train_data)), rand_num)
            rand_subset = [self.train_data[i] for i in rand_idx]
            rand_loader = DataLoader(rand_subset, self.batch_size, shuffle=True, drop_last=False)

        # 获取参数引用
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())
        
        # 第一轮通信时跳过（模型相同）
        if len(params_g) > 0 and len(params) > 0:
            if torch.sum(torch.abs(params_g[0].data - params[0].data)) < 1e-10:
                return local_model

        # 保留低层的更新
        if self.layer_idx > 0 and len(params) > self.layer_idx:
            for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
                param.data = param_g.data.clone()

        # 用于权重学习的临时模型
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # 只考虑高层
        if self.layer_idx > 0 and len(params) >= self.layer_idx:
            params_p = params[-self.layer_idx:]
            params_gp = params_g[-self.layer_idx:]
            params_tp = params_t[-self.layer_idx:]
            
            # 冻结低层以减少计算
            for param in params_t[:-self.layer_idx]:
                param.requires_grad = False

            # 初始化权重（如果第一次）
            if self.weights is None:
                self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

            # 初始化高层
            for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                param_t.data = param + (param_g - param) * weight

            # 权重学习
            optimizer = torch.optim.SGD(params_tp, lr=0)  # lr=0，因为只更新权重
            
            cnt = 0
            max_iterations = 10  # 最大迭代次数
            
            while cnt < max_iterations:
                epoch_losses = []
                for x, y in rand_loader:
                    # 移动数据到设备
                    if isinstance(x, list):
                        x = [xi.to(self.device) for xi in x]
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model_t(x)
                    loss_value = self.loss(output, y)
                    loss_value.backward()

                    # 更新权重
                    for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                        grad_weight = -self.eta * torch.sum(param_t.grad * (param_g - param))
                        weight.data = torch.clamp(weight + grad_weight, 0, 1)

                    # 更新临时模型
                    for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                        param_t.data = param + (param_g - param) * weight
                    
                    epoch_losses.append(loss_value.item())
                
                self.loss_history.append(np.mean(epoch_losses))
                cnt += 1
                
                # 检查收敛条件
                if not self.start_phase:
                    break
                    
                if len(self.loss_history) > self.num_pre_loss:
                    recent_losses = self.loss_history[-self.num_pre_loss:]
                    if np.std(recent_losses) < self.threshold:
                        break
            
            self.start_phase = False
            
            # 更新本地模型的高层
            for param, param_t in zip(params_p, params_tp):
                param.data = param_t.data.clone()
        
        return local_model

    def get_loss_history(self) -> List[float]:
        """获取损失历史"""
        return self.loss_history