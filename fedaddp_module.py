# fedaddp_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from typing import Dict
from utils import Utils
from fedavg import FedAvgServer
class FedADDPClient:
    """FedADDP客户端（严格遵循原始论文实现）"""
    
    def __init__(self, client_id: int, model: nn.Module,
                 train_loader, test_loader, device):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
         # 初始裁剪边界，将由服务器传入或从0开始

    def local_train(self,
                    global_model: nn.Module,
                    local_epochs: int,
                    lr: float,
                    momentum: float,
                    weight_decay: float,
                    fisher_threshold: float,
                    lambda_1: float,
                    lambda_2: float,
                    beta: float,
                    sigma0: float,
                    clipping_bound: float,
                    no_clip: bool = False,
                    no_noise: bool = False) -> Dict:
        """
        FedADDP本地训练（完全按照原始论文 local_update 函数）
        """
        # ---------- 1. 准备 ----------
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(self.device)
        global_model.to(self.device)

        # ---------- 2. 计算 Fisher 对角线 ----------
        fisher_diag = Utils.compute_fisher_diag(self.model, self.train_loader, self.device)

        # ---------- 3. 获取全局模型参数 ----------
        w_glob = [param.clone().detach() for param in global_model.parameters()]

        # ---------- 4. 分割 u 和 v ----------
        u_loc, v_loc = [], []
        for param, fisher_val in zip(self.model.parameters(), fisher_diag):
            u_param = (param * (fisher_val > fisher_threshold)).clone().detach()
            v_param = (param * (fisher_val <= fisher_threshold)).clone().detach()
            u_loc.append(u_param)
            v_loc.append(v_param)

        u_glob, v_glob = [], []
        for global_param, fisher_val in zip(global_model.parameters(), fisher_diag):
            u_param = (global_param * (fisher_val > fisher_threshold)).clone().detach()
            v_param = (global_param * (fisher_val <= fisher_threshold)).clone().detach()
            u_glob.append(u_param)
            v_glob.append(v_param)

        # ---------- 5. 初始化模型参数为 u_loc + v_glob ----------
        for u_p, v_p, model_param in zip(u_loc, v_glob, self.model.parameters()):
            model_param.data = u_p + v_p

        w_0 = [param.clone().detach() for param in self.model.parameters()]  # 保存初始参数

        # ---------- 6. 定义自定义损失函数 ----------
        def custom_loss(outputs, labels, param_diffs, reg_type, v_clip_bound):
            ce_loss = F.cross_entropy(outputs, labels)
            if reg_type == "R1":
                reg_loss = (lambda_1 / 2) * torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))
            elif reg_type == "R2":
                norm_diff = torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))
                reg_loss = (lambda_2 / 2) * torch.norm(norm_diff - v_clip_bound)
            else:
                raise ValueError("Invalid regularization type")
            return ce_loss + reg_loss

        # ---------- 7. 准备优化器和梯度记录 ----------
        optimizer1 = torch.optim.Adam(self.model.parameters(), lr=lr)
        optimizer2 = torch.optim.Adam(self.model.parameters(), lr=lr)
        u_gradients = [torch.zeros_like(p) for p in self.model.parameters()]
        v_gradients = [torch.zeros_like(p) for p in self.model.parameters()]

        last_client_model = None
        last_client_model_update = [torch.zeros_like(p) for p in self.model.parameters()]

        epoch_losses = []  # 用于记录损失（可选）
        
        # ---------- 8. 遍历所有 batch ----------
        for batch_id, (data, labels) in enumerate(self.train_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            batch_size = data.size(0)

            # 记录上一个模型状态（用于最后的裁剪）
            last_client_model = copy.deepcopy(self.model)

            # ---------- 第一阶段：R1 正则，更新 u 部分 ----------
            optimizer1.zero_grad()
            outputs = self.model(data)
            param_diffs = [new - old for new, old in zip(self.model.parameters(), w_glob)]
            loss1 = custom_loss(outputs, labels, param_diffs, "R1", 0)
            loss1.backward()

            # 记录 u 的梯度（掩码后）
            with torch.no_grad():
                for grad, model_param, u_p in zip(u_gradients, self.model.parameters(), u_loc):
                    model_param.grad *= (u_p != 0).float()
                    grad.copy_(model_param.grad)
            optimizer1.step()

            # ---------- 第二阶段：R2 正则，更新 v 部分 ----------
            optimizer2.zero_grad()
            outputs = self.model(data)
            param_diffs = [new - old for new, old in zip(self.model.parameters(), w_glob)]
            loss2 = custom_loss(outputs, labels, param_diffs, "R2", clipping_bound)
            loss2.backward()

            # 记录 v 的梯度（掩码后）
            with torch.no_grad():
                for grad, model_param, v_p in zip(v_gradients, self.model.parameters(), v_glob):
                    model_param.grad *= (v_p != 0).float()
                    grad.copy_(model_param.grad)

            # 计算本轮更新量（用于后续裁剪）
            with torch.no_grad():
                last_client_model_update = [lr * (u_g + v_g) for u_g, v_g in zip(u_gradients, v_gradients)]

            # 再次掩码后执行 optimizer2 步进（原始代码中在记录 last_client_model_update 后做了这个）
            with torch.no_grad():
                for model_param, v_p in zip(self.model.parameters(), v_glob):
                    model_param.grad *= (v_p != 0).float()
            optimizer2.step()

            # 记录总损失（用于返回）
            epoch_losses.append((loss1.item() + loss2.item()) / 2.0)

        # ---------- 9. 所有 batch 处理完毕，得到训练后的模型 ----------
        new_model = copy.deepcopy(self.model)  # 不加噪的模型副本，稍后将加噪

        # ---------- 10. 裁剪和加噪（按照原始代码） ----------
        if no_clip:
            new_clipping_bound = 0
        else:
            M = len(list(new_model.parameters()))
            new_clipping_bound = 0.0
            # 获取 last_client_model 的参数列表（用于计算 q）
            last_params = list(last_client_model.parameters()) if last_client_model else list(self.model.parameters())
            # 确保 last_client_model_update 长度一致
            for (client_param, last_param, last_update, w0_param) in zip(
                    new_model.parameters(), last_params, last_client_model_update, w_0):
                q = (beta / 2) * torch.abs(last_param - last_update)
                # 裁剪
                client_param.data = torch.clamp(
                    client_param.data,
                    min=w0_param.data - q,
                    max=w0_param.data + q
                )
                delta_f = 2 * q  # 原始论文中裁剪边界是 2*q
                new_clipping_bound += torch.norm(delta_f).item()
                if not no_noise and sigma0 > 0:
                    omega_m = math.sqrt(M) * sigma0 * delta_f
                    noise = torch.randn_like(client_param.data) * omega_m
                    client_param.data += noise

        # ---------- 11. 计算测试准确率 ----------
        accuracy = self._test_model(new_model)

        # ---------- 12. 计算模型更新（用于服务器聚合）----------
        model_update = {}
        global_state = global_model.state_dict()
        local_state = new_model.state_dict()
        for name in global_state.keys():
            model_update[name] = local_state[name] - global_state[name]

        # ---------- 13. 返回结果 ----------
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        return {
            'update': model_update,
            'accuracy': accuracy,
            'loss': avg_loss,
            'grad_norm': 0.0,  # 原始论文未计算梯度范数
            'clipping_bound': new_clipping_bound
        }

    def _test_model(self, model):
        """测试模型准确率"""
        model.eval()
        model.to(self.device)
        correct = total = 0
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = model(data)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        return 100.0 * correct / total if total > 0 else 0


class FedADDPServer(FedAvgServer):
    """FedADDP服务器（维护每个客户端的裁剪边界）"""
    def __init__(self, global_model: nn.Module, device: torch.device,
                 num_clients: int, client_data_sizes: Dict[int, int] = None):
        super().__init__(global_model, device, client_data_sizes)
        self.num_clients = num_clients
        # 为每个客户端初始化裁剪边界为 0
        self.client_clipping_bounds = {i: 0.0 for i in range(num_clients)}

    def get_clipping_bound(self, client_id: int) -> float:
        """获取指定客户端的当前裁剪边界"""
        return self.client_clipping_bounds.get(client_id, 0.0)

    def update_clipping_bound(self, client_id: int, new_bound: float):
        """更新指定客户端的裁剪边界"""
        self.client_clipping_bounds[client_id] = new_bound

    def aggregate(self, client_updates: Dict[int, Dict],
                  client_weights: Dict[int, float] = None) -> None:
        """
        聚合客户端更新，同时更新裁剪边界
        """
        # 先更新裁剪边界
        for client_id, update_info in client_updates.items():
            if 'clipping_bound' in update_info:
                
                self.update_clipping_bound(client_id, update_info['clipping_bound'])
        # 调用父类的聚合方法（数据量加权平均）
        super().aggregate(client_updates, client_weights)