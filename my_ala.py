# ala.py（完整文件，已集成伪标签，无本地训练干扰）
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from torch.utils.data import DataLoader
from typing import List, Tuple

class MYALA:
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
                num_pre_loss: int,
                # ---------- 伪标签集成参数 ----------
                use_pseudo,
                pseudo_threshold: float = 0.8,
                pseudo_weight: float = 0.5,
                max_iter: int = 30):      # 最大迭代保护
        """
        Args:
            client_id: 客户端ID
            loss_function: 损失函数
            train_data: 训练数据（list of tuples）
            batch_size: 批次大小
            rand_percent: 随机采样百分比
            layer_idx: 层索引
            eta: 权重学习率
            device: 设备
            threshold: 收敛阈值
            num_pre_loss: 用于计算标准差的损失数量
            use_pseudo: 是否在ALA权重学习中使用伪标签
            pseudo_threshold: 伪标签置信度阈值
            pseudo_weight: 伪标签损失权重
            max_iter: 最大迭代次数（防止卡死）
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
        self.use_pseudo = use_pseudo
        self.pseudo_threshold = pseudo_threshold
        self.pseudo_weight = pseudo_weight
        self.max_iter = max_iter

        self.weights = None
        self.start_phase = True

    def adaptive_local_aggregation(self, 
                                 global_model: nn.Module,
                                 local_model: nn.Module) -> None:
        """自适应本地聚合（完全接管伪标签）"""
        # ---------- 1. 健壮性检查 ----------
        if self.train_data is None or len(self.train_data) == 0:
            return
        rand_num = int((self.rand_percent / 100) * len(self.train_data))
        if rand_num == 0:
            return
            
        # ---------- 2. 随机采样 ----------
        rand_idx = random.randint(0, max(0, len(self.train_data) - rand_num))
        rand_data = self.train_data[rand_idx:rand_idx + rand_num]
        rand_loader = DataLoader(rand_data, self.batch_size, drop_last=False)

        # ---------- 3. 参数准备 ----------
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        if len(params_g) == 0 or len(params) == 0:
            return
        if torch.sum(params_g[0].data - params[0].data) == 0:
            return

        # 保留低层更新
        preserve_layers = min(len(params), len(params_g)) - self.layer_idx
        if preserve_layers > 0:
            for param, param_g in zip(params[:preserve_layers], params_g[:preserve_layers]):
                param.data = param_g.data.clone()

        # 临时模型
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        params_p = params[-self.layer_idx:] if self.layer_idx > 0 else []
        params_gp = params_g[-self.layer_idx:] if self.layer_idx > 0 else []
        params_tp = params_t[-self.layer_idx:] if self.layer_idx > 0 else []

        if not params_p or not params_gp or not params_tp:
            return

        # 冻结低层
        for param in params_t[:-self.layer_idx] if self.layer_idx > 0 else params_t:
            param.requires_grad = False

        optimizer = torch.optim.SGD(params_tp, lr=0)

        if self.weights is None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # 初始化临时模型
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
            param_t.data = param + (param_g - param) * weight

        # ---------- 4. 权重学习（集成伪标签）----------
        losses = []
        cnt = 0

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
                
                # --- 有监督损失 ---
                sup_loss = self.loss(output, y)
                total_loss = sup_loss

                # --- 伪标签损失（集成，仅在启用时）---
                if self.use_pseudo:
                    with torch.no_grad():
                        probs = F.softmax(output, dim=1)
                        max_probs, pseudo_labels = torch.max(probs, dim=1)
                        mask = max_probs > self.pseudo_threshold
                    if mask.sum() > 0:
                        pseudo_loss = self.loss(output[mask], pseudo_labels[mask])
                        total_loss = total_loss + self.pseudo_weight * pseudo_loss
                # --------------------------------

                total_loss.backward()

                # 更新权重
                for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                    grad_contribution = param_t.grad * (param_g - param)
                    if grad_contribution is not None:
                        weight_update = self.eta * grad_contribution
                        weight.data = torch.clamp(weight - weight_update, 0, 1)

                # 更新临时模型
                for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight
                
                epoch_loss += total_loss.item()
                batch_count += 1

            if batch_count > 0:
                losses.append(epoch_loss / batch_count)
            cnt += 1

            # 退出条件
            if not self.start_phase:
                break
            if len(losses) > self.num_pre_loss and \
               np.std(losses[-self.num_pre_loss:]) < self.threshold:
                if self.client_id < 10:
                    print(f'Client {self.client_id}: ALA收敛于{cnt}轮，Std={np.std(losses[-self.num_pre_loss:]):.4f}')
                break
            if cnt >= self.max_iter:
                if self.client_id < 10:
                    print(f'Client {self.client_id}: ALA达到最大迭代{self.max_iter}，强制退出')
                break

        self.start_phase = False

        # 更新本地模型的高层参数
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()