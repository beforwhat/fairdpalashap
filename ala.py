# ala.py
import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional

class ALA:
    def __init__(self,
                client_id: int,
                loss_function: nn.Module,
                train_data: List[Tuple],
                batch_size: int,
                rand_percent: int,
                layer_idx: int,
                eta: float,
                device: torch.device,
                threshold: float,
                num_pre_loss: int):
        """
        修改后的ALA模块
        
        Args:
            client_id: 客户端ID
            loss_function: 损失函数
            train_data: 训练数据
            batch_size: 批次大小
            rand_percent: 随机采样百分比
            layer_idx: 层索引
            eta: 权重学习率
            device: 设备
            threshold: 训练阈值
            num_pre_loss: 预损失数量
        """
        self.client_id = client_id
        self.loss = loss_function
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None  # 可学习的本地聚合权重
        self.start_phase = True

    def adaptive_local_aggregation(self, 
                                 global_model: nn.Module,
                                 local_model: nn.Module) -> None:
        """
        自适应本地聚合
        
        Args:
            global_model: 全局模型
            local_model: 本地模型
        """
        # 如果没有训练数据，直接返回
        if len(self.train_data) == 0:
            return

        # 随机采样部分本地训练数据
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(self.train_data))
        
        if rand_num == 0:
            return
            
        rand_idx = random.randint(0, max(0, len(self.train_data) - rand_num))
        rand_data = self.train_data[rand_idx:rand_idx + rand_num]
        
        # 转换为DataLoader
        if len(rand_data) > 0:
            rand_loader = DataLoader(rand_data, self.batch_size, drop_last=False)
        else:
            return

        # 获取参数引用
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # 在第一轮通信迭代中停用ALA
        if len(params_g) == 0 or len(params) == 0:
            return
            
        if torch.sum(params_g[0].data - params[0].data) == 0:
            return

        # 保留较低层的更新
        preserve_layers = min(len(params), len(params_g)) - self.layer_idx
        if preserve_layers > 0:
            for param, param_g in zip(params[:preserve_layers], params_g[:preserve_layers]):
                param.data = param_g.data.clone()

        # 仅用于权重学习的临时本地模型
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # 仅考虑较高层
        params_p = params[-self.layer_idx:] if self.layer_idx > 0 else []
        params_gp = params_g[-self.layer_idx:] if self.layer_idx > 0 else []
        params_tp = params_t[-self.layer_idx:] if self.layer_idx > 0 else []

        if not params_p or not params_gp or not params_tp:
            return

        # 冻结较低层以减少计算成本
        for param in params_t[:-self.layer_idx] if self.layer_idx > 0 else params_t:
            param.requires_grad = False

        # 用于获取较高层的梯度
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # 在开始时将权重初始化为全1
        if self.weights is None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # 初始化临时本地模型中的较高层
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
            param_t.data = param + (param_g - param) * weight

        # 权重学习
        losses = []  # 记录损失
        cnt = 0  # 权重训练迭代计数器
        
        while True:
            epoch_loss = 0
            batch_count = 0
            
            for x, y in rand_loader:
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
                    grad_contribution = param_t.grad * (param_g - param)
                    if grad_contribution is not None:
                        weight_update = self.eta * grad_contribution
                        weight.data = torch.clamp(weight - weight_update, 0, 1)

                # 更新临时本地模型
                for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight
                
                epoch_loss += loss_value.item()
                batch_count += 1

            if batch_count > 0:
                losses.append(epoch_loss / batch_count)
            cnt += 1

            # 在后续迭代中仅训练一个epoch
            if not self.start_phase:
                break

            # 训练权重直到收敛
            if (len(losses) > self.num_pre_loss and 
                np.std(losses[-self.num_pre_loss:]) < self.threshold):
                if self.client_id < 10:  # 只打印前10个客户端的日志
                    print(f'Client: {self.client_id}, Std: {np.std(losses[-self.num_pre_loss:]):.4f}, ALA epochs: {cnt}')
                break

        self.start_phase = False

        # 获取初始化的本地模型
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()