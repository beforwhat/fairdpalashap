# fedaddp_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, List
from utils import Utils

class FedADDPClient:
    """FedADDP客户端"""
    
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
        self.u_loc = None
        self.v_loc = None
        self.fisher_diag = None
    
    def local_train(self, global_model: nn.Module,
                    local_epochs: int, lr: float,
                    momentum: float, weight_decay: float,
                    fisher_threshold: float, lambda_1: float,
                    lambda_2: float, beta: float,
                    sigma0: float, clipping_bound: float,
                    no_clip: bool, no_noise: bool) -> Dict:
        """
        FedADDP本地训练
        
        Args:
            fisher_threshold: Fisher阈值
            lambda_1: R1正则化系数
            lambda_2: R2正则化系数
            beta: β参数
            sigma0: 噪声尺度
            clipping_bound: 裁剪边界
            no_clip: 是否禁用裁剪
            no_noise: 是否禁用噪声
            
        Returns:
            dict: 训练结果
        """
        # 将模型移动到设备
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(self.device)
        global_model.to(self.device)
        
        # 计算Fisher对角线
        self.fisher_diag = Utils.compute_fisher_diag(
            self.model, self.train_loader, self.device
        )
        
        # 获取全局参数
        w_glob = [param.clone().detach() for param in global_model.parameters()]
        
        # 分割u和v
        self.u_loc = []
        self.v_loc = []
        for param, fisher_value in zip(self.model.parameters(), self.fisher_diag):
            u_param = (param * (fisher_value > fisher_threshold)).clone().detach()
            v_param = (param * (fisher_value <= fisher_threshold)).clone().detach()
            self.u_loc.append(u_param)
            self.v_loc.append(v_param)
        
        # 分割全局u和v
        u_glob = []
        v_glob = []
        for global_param, fisher_value in zip(global_model.parameters(), self.fisher_diag):
            u_param = (global_param * (fisher_value > fisher_threshold)).clone().detach()
            v_param = (global_param * (fisher_value <= fisher_threshold)).clone().detach()
            u_glob.append(u_param)
            v_glob.append(v_param)
        
        # 合并u和v
        for u_param, v_param, model_param in zip(self.u_loc, v_glob, self.model.parameters()):
            model_param.data = u_param + v_param
        
        w_0 = [param.clone().detach() for param in self.model.parameters()]
        
        # 自定义损失函数
        def custom_loss(outputs, labels, param_diffs, reg_type, v_clipping_bound):
            ce_loss = F.cross_entropy(outputs, labels)
            
            if reg_type == "R1":
                reg_loss = (lambda_1 / 2) * torch.sum(
                    torch.stack([torch.norm(diff) for diff in param_diffs])
                )
            elif reg_type == "R2":
                norm_diff = torch.sum(
                    torch.stack([torch.norm(diff) for diff in param_diffs])
                )
                reg_loss = (lambda_2 / 2) * torch.norm(norm_diff - v_clipping_bound)
            else:
                raise ValueError("Invalid regularization type")
            
            return ce_loss + reg_loss
        
        # 优化器
        optimizer1 = torch.optim.Adam(self.model.parameters(), lr=lr)
        optimizer2 = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # 梯度记录
        u_gradients = [torch.zeros_like(param) for param in self.model.parameters()]
        v_gradients = [torch.zeros_like(param) for param in self.model.parameters()]
        
        last_client_model = None
        last_client_model_update = [torch.zeros_like(param) for param in self.model.parameters()]
        
        # 训练循环（简化版，原代码中的batch循环）
        for data, labels in self.train_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            
            # 记录上一个模型
            last_client_model = copy.deepcopy(self.model)
            
            # 第一阶段训练（R1正则化）
            optimizer1.zero_grad()
            outputs = self.model(data)
            param_diffs = [u_new - u_old for u_new, u_old in zip(self.model.parameters(), w_glob)]
            loss = custom_loss(outputs, labels, param_diffs, "R1", 0)
            loss.backward()
            
            # 记录u梯度
            with torch.no_grad():
                for grad, model_param, u_param in zip(u_gradients, self.model.parameters(), self.u_loc):
                    model_param.grad *= (u_param != 0)
                    grad.copy_(model_param.grad)
            
            optimizer1.step()
            
            # 第二阶段训练（R2正则化）
            optimizer2.zero_grad()
            outputs = self.model(data)
            param_diffs = [model_param - w_old for model_param, w_old in zip(self.model.parameters(), w_glob)]
            loss = custom_loss(outputs, labels, param_diffs, "R2", clipping_bound)
            loss.backward()
            
            # 记录v梯度
            with torch.no_grad():
                for grad, model_param, v_param in zip(v_gradients, self.model.parameters(), self.v_loc):
                    model_param.grad *= (v_param != 0)
                    grad.copy_(model_param.grad)
            
            # 记录模型更新
            last_client_model_update = [
                lr * (u_grad + v_grad) for u_grad, v_grad in zip(u_gradients, v_gradients)
            ]
            
            with torch.no_grad():
                for model_param, v_param in zip(self.model.parameters(), self.v_loc):
                    model_param.grad *= (v_param != 0)
            
            optimizer2.step()
        
        # 创建新模型（加噪后）
        new_model = copy.deepcopy(self.model)
        
        if not no_clip:
            M = len(list(new_model.parameters()))
            new_clipping_bound = 0
            
            for (client_param, last_param, last_update, w0_param) in zip(
                new_model.parameters(), last_client_model.parameters(),
                last_client_model_update, w_0):
                
                q = (beta / 2) * torch.abs(last_param - last_update)
                client_param.data = torch.clamp(
                    client_param.data,
                    min=w0_param.data - q.data,
                    max=w0_param.data + q.data
                )
                
                delta_f = 2 * q
                new_clipping_bound += torch.norm(delta_f).item()
                
                if not no_noise:
                    omega_m = np.sqrt(M) * sigma0 * delta_f
                    noise = torch.randn_like(client_param.data) * omega_m
                    client_param.data += noise
            
            clipping_bound = new_clipping_bound
        else:
            clipping_bound = 0
        
        # 计算准确率
        accuracy = self.test()
        
        # 计算模型更新
        model_update = {}
        global_state = global_model.state_dict()
        new_state = new_model.state_dict()
        
        for name in global_state.keys():
            model_update[name] = new_state[name] - global_state[name]
        
        return {
            'update': model_update,
            'accuracy': accuracy,
            'clipping_bound': clipping_bound,
            'grad_norm': 0
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