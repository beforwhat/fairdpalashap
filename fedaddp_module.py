# fedaddp_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, List
from utils import Utils
from lr_scheduler import CosineAnnealingLR
class FedADDPClient:
    """FedADDP客户端（重新实现以保持接口一致）"""
    
    def __init__(self, client_id: int, model: nn.Module,
                 train_loader, test_loader, device):
        """
        初始化FedADDP客户端
        
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
        
        # FedADDP特定参数
        self.u_params = None
        self.v_params = None
        self.fisher_diag = None
        self.grad_norms = []
        self.clipping_bound = 1.0
    
    def local_train(self, global_model: nn.Module,
                    local_epochs: int, lr: float,
                    momentum: float, weight_decay: float,
                    fisher_threshold: float, lambda_1: float,
                    lambda_2: float, beta: float,
                    sigma0: float, no_clip: bool = False,
                    no_noise: bool = False) -> Dict:
        """
        FedADDP本地训练（重新实现，支持多epoch和标准优化器）
        
        Args:
            local_epochs: 本地训练轮数（之前未使用）
            lr: 学习率
            fisher_threshold: Fisher阈值
            lambda_1: R1正则化系数
            lambda_2: R2正则化系数
            beta: β参数
            sigma0: 噪声尺度
            no_clip: 是否禁用裁剪
            no_noise: 是否禁用噪声
            
        Returns:
            dict: 训练结果
        """
        # 将模型移动到设备
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(self.device)
        global_model.to(self.device)
        
        # 计算Fisher信息矩阵对角线
        self.fisher_diag = Utils.compute_fisher_diag(
            self.model, self.train_loader, self.device
        )
        
        # 获取全局参数
        w_glob = [param.clone().detach() for param in global_model.parameters()]
        
        # 根据Fisher阈值分割参数为u（重要）和v（不重要）
        self.u_params = []
        self.v_params = []
        for param, fisher_value in zip(self.model.parameters(), self.fisher_diag):
            # u参数：Fisher值大于阈值的部分
            u_mask = (fisher_value > fisher_threshold).float()
            u_param = (param * u_mask).clone().detach()
            
            # v参数：Fisher值小于等于阈值的部分
            v_mask = (fisher_value <= fisher_threshold).float()
            v_param = (param * v_mask).clone().detach()
            
            self.u_params.append(u_param)
            self.v_params.append(v_param)
        
        # 初始化模型参数：u部分使用本地，v部分使用全局
        u_glob = []
        v_glob = []
        for global_param, fisher_value in zip(global_model.parameters(), self.fisher_diag):
            u_mask = (fisher_value > fisher_threshold).float()
            v_mask = (fisher_value <= fisher_threshold).float()
            
            u_glob.append((global_param * u_mask).clone().detach())
            v_glob.append((global_param * v_mask).clone().detach())
        
        # 合并参数：u部分使用本地，v部分使用全局
        with torch.no_grad():
            for u_param, v_g_param, model_param in zip(self.u_params, v_glob, self.model.parameters()):
                model_param.data = u_param + v_g_param
        
        # 保存初始模型参数
        w_0 = [param.clone().detach() for param in self.model.parameters()]
        
        # 定义FedADDP的自定义损失函数
        def fedaddp_loss(outputs, labels, model_params, global_params, reg_type, clipping_bound):
            ce_loss = F.cross_entropy(outputs, labels)
            
            if reg_type == "R1":
                # R1正则化：保持参数接近全局参数
                reg_loss = 0
                for model_param, global_param in zip(model_params, global_params):
                    reg_loss += torch.norm(model_param - global_param) ** 2
                reg_loss = (lambda_1 / 2) * reg_loss
            
            elif reg_type == "R2":
                # R2正则化：控制更新幅度
                total_diff = 0
                for model_param, global_param in zip(model_params, global_params):
                    total_diff += torch.norm(model_param - global_param) ** 2
                total_diff = torch.sqrt(total_diff)
                reg_loss = (lambda_2 / 2) * torch.norm(total_diff - clipping_bound) ** 2
            
            else:
                raise ValueError("Invalid regularization type")
            
            return ce_loss + reg_loss
        
        # 优化器（与其他方法一致）
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # 记录训练信息
        epoch_losses = []
        grad_norms = []
        
        # 多epoch训练
        for epoch in range(local_epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, (data, labels) in enumerate(self.train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                
                # 第一阶段：使用R1正则化训练u参数
                optimizer.zero_grad()
                outputs = self.model(data)
                
                # 获取当前参数
                current_params = list(self.model.parameters())
                
                # 计算R1损失（只对u参数部分）
                loss_r1 = fedaddp_loss(outputs, labels, current_params, w_glob, "R1", 0)
                loss_r1.backward()
                
                # 只保留u参数的梯度
                with torch.no_grad():
                    for param, u_param in zip(self.model.parameters(), self.u_params):
                        param.grad *= (u_param != 0).float()
                
                optimizer.step()
                
                # 第二阶段：使用R2正则化训练所有参数
                optimizer.zero_grad()
                outputs = self.model(data)
                
                # 计算R2损失
                loss_r2 = fedaddp_loss(outputs, labels, current_params, w_glob, "R2", self.clipping_bound)
                loss_r2.backward()
                
                # 记录梯度范数
                grad_norm = Utils.clip_gradients(self.model, self.clipping_bound)
                grad_norms.append(grad_norm)
                
                # 添加DP噪声（如果不禁用）
                if not no_noise and sigma0 > 0:
                    Utils.add_dp_noise(self.model, self.clipping_bound, sigma0)
                
                optimizer.step()
                
                # 记录损失
                total_loss = loss_r1.item() + loss_r2.item()
                epoch_loss += total_loss
                batch_count += 1
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                epoch_losses.append(avg_loss)
        
        # 应用FedADDP的裁剪和加噪机制（如果不禁用）
        if not no_clip:
            # 计算参数变化范围
            M = len(list(self.model.parameters()))
            new_clipping_bound = 0
            
            # 获取训练后的参数
            trained_params = list(self.model.parameters())
            
            for idx, (trained_param, w0_param) in enumerate(zip(trained_params, w_0)):
                # 计算参数变化
                param_change = trained_param.data - w0_param.data
                
                # 计算裁剪边界
                q = (beta / 2) * torch.abs(param_change)
                
                # 裁剪参数
                trained_param.data = torch.clamp(
                    trained_param.data,
                    min=w0_param.data - q,
                    max=w0_param.data + q
                )
                
                # 更新裁剪边界
                delta_f = 2 * q
                new_clipping_bound += torch.norm(delta_f).item()
                
                # 添加噪声（如果不禁用）
                if not no_noise and sigma0 > 0:
                    omega_m = np.sqrt(M) * sigma0 * delta_f
                    noise = torch.randn_like(trained_param.data) * omega_m
                    trained_param.data += noise
            
            self.clipping_bound = new_clipping_bound
        else:
            self.clipping_bound = 0
        
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
        
        return {
            'update': model_update,
            'accuracy': accuracy,
            'loss': np.mean(epoch_losses) if epoch_losses else 0,
            'grad_norm': avg_grad_norm,
            'clipping_bound': self.clipping_bound,
            'fisher_diag_mean': np.mean([f.mean().item() for f in self.fisher_diag]) if self.fisher_diag else 0
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


class FedADDPServer:
    """FedADDP服务器"""
    
    def __init__(self, global_model: nn.Module, device: torch.device):
        """
        初始化FedADDP服务器
        
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
        self.client_clipping_bounds = {}
    def get_current_lr(self) -> float:
        """获取当前学习率（供客户端使用）"""
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
            client_weights = {cid: 1.0/len(selected_clients) for cid in selected_clients}
        
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
                
                # 记录客户端的裁剪边界
                clipping_bound = client_updates[client_id].get('clipping_bound', 0)
                self.client_clipping_bounds[client_id] = clipping_bound
        
        # 更新全局模型
        current_state = self.global_model.state_dict()
        new_state = {}
        current_lr = self.lr_scheduler.step()
        for name in current_state.keys():
            if name in global_update:
                new_state[name] = current_state[name] + global_update[name]
            else:
                new_state[name] = current_state[name]
        
        self.global_model.load_state_dict(new_state)
    
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