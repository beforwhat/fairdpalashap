# utils.py
import torch
import numpy as np
from scipy.special import comb
from itertools import chain, combinations
import torch.nn.functional as F
from typing import List, Dict, Tuple
import copy

class Utils:
    """工具函数类"""
    
    @staticmethod
    def compute_shapley_contributions(model_updates: Dict[int, List[torch.Tensor]], 
                                     test_accuracies: Dict[int, float]) -> Dict[int, float]:
        """
        计算近似Shapley值贡献
        
        Args:
            model_updates: 客户端模型更新 {client_id: [param_updates]}
            test_accuracies: 客户端测试准确率 {client_id: accuracy}
            
        Returns:
            shapley_values: Shapley值 {client_id: value}
        """
        n_clients = len(model_updates)
        shapley_values = {i: 0.0 for i in range(n_clients)}
        
        if n_clients <= 10:  # 客户端较少时精确计算
            for client_id in range(n_clients):
                total_contrib = 0
                for subset in Utils.powersettool(range(n_clients)):
                    if client_id not in subset:
                        continue
                    
                    subset_with = list(subset)
                    subset_without = [c for c in subset if c != client_id]
                    
                    # 计算边际贡献
                    if len(subset_with) > 0:
                        acc_with = np.mean([test_accuracies[c] for c in subset_with])
                    else:
                        acc_with = 0
                    
                    if len(subset_without) > 0:
                        acc_without = np.mean([test_accuracies[c] for c in subset_without])
                    else:
                        acc_without = 0
                    
                    marginal = acc_with - acc_without
                    weight = 1 / (comb(n_clients - 1, len(subset_with) - 1) * n_clients)
                    total_contrib += marginal * weight
                
                shapley_values[client_id] = total_contrib
        
        else:  # 客户端较多时使用蒙特卡洛近似
            n_samples = min(1000, 2**n_clients)
            for _ in range(n_samples):
                # 随机排列
                perm = np.random.permutation(n_clients)
                
                for i, client_id in enumerate(perm):
                    # 前面的客户端集合
                    prev_clients = perm[:i]
                    
                    # 计算边际贡献
                    if len(prev_clients) > 0:
                        acc_before = np.mean([test_accuracies[c] for c in prev_clients])
                    else:
                        acc_before = 0
                    
                    clients_with = list(prev_clients) + [client_id]
                    acc_after = np.mean([test_accuracies[c] for c in clients_with])
                    
                    marginal = acc_after - acc_before
                    shapley_values[client_id] += marginal / n_samples
        
        # 归一化
        total = sum(shapley_values.values())
        if total > 0:
            shapley_values = {k: v/total for k, v in shapley_values.items()}
        
        return shapley_values
    
    @staticmethod
    def powersettool(iterable):
        """生成幂集"""
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
    @staticmethod
    def compute_data_diversity(data_distributions: Dict[int, List[int]]) -> Dict[int, float]:
        """
        计算客户端数据多样性（使用熵）
        
        Args:
            data_distributions: 客户端数据分布 {client_id: [class_counts]}
            
        Returns:
            diversities: 数据多样性 {client_id: diversity}
        """
        diversities = {}
        for client_id, class_counts in data_distributions.items():
            counts = np.array(class_counts)
            total = counts.sum()
            if total > 0:
                probs = counts / total
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                diversities[client_id] = entropy
            else:
                diversities[client_id] = 0
        
        # 归一化
        max_entropy = max(diversities.values()) if diversities else 1
        diversities = {k: v/max_entropy for k, v in diversities.items()}
        
        return diversities
    
    @staticmethod
    def compute_participation_frequency(selected_counts: Dict[int, int], 
                                       current_round: int) -> Dict[int, float]:
        """
        计算客户端参与频率
        
        Args:
            selected_counts: 客户端被选中的次数
            current_round: 当前轮数
            
        Returns:
            frequencies: 参与频率 {client_id: frequency}
        """
        frequencies = {}
        for client_id, count in selected_counts.items():
            frequencies[client_id] = count / max(current_round, 1)
        
        return frequencies
    
    @staticmethod
    def select_clients_strategic(shapley_values: Dict[int, float],
                               diversities: Dict[int, float],
                               frequencies: Dict[int, float],
                               num_select: int,
                               shapley_weight: float = 0.5,
                               diversity_weight: float = 0.3,
                               participation_weight: float = 0.2) -> List[int]:
        """
        基于Shapley值、数据多样性和参与频率选择客户端
        
        Args:
            shapley_values: Shapley值
            diversities: 数据多样性
            frequencies: 参与频率
            num_select: 选择数量
            weights: 权重分配
            
        Returns:
            selected_clients: 选中的客户端ID列表
        """
        all_clients = list(shapley_values.keys())
        
        # 计算综合得分
        scores = {}
        for client_id in all_clients:
            score = (shapley_weight * shapley_values[client_id] +
                    diversity_weight * diversities[client_id] +
                    participation_weight * (1 - frequencies[client_id]))  # 参与频率低的优先
            scores[client_id] = score
        
        # 基于概率选择
        score_values = np.array(list(scores.values()))
        probs = score_values / score_values.sum()
        
        selected_clients = np.random.choice(
            all_clients,
            size=min(num_select, len(all_clients)),
            replace=False,
            p=probs
        ).tolist()
        
        return selected_clients
    
    @staticmethod
    def compute_aggregation_weights(shapley_values: Dict[int, float],
                                  diversities: Dict[int, float],
                                  frequencies: Dict[int, float],
                                  selected_clients: List[int],
                                  shapley_weight: float = 0.5,
                                  diversity_weight: float = 0.3,
                                  participation_weight: float = 0.2) -> Dict[int, float]:
        """
        计算客户端聚合权重
        
        Args:
            shapley_values: Shapley值
            diversities: 数据多样性
            frequencies: 参与频率
            selected_clients: 选中的客户端
            weights: 权重分配
            
        Returns:
            weights: 聚合权重 {client_id: weight}
        """
        weights = {}
        
        # 计算每个选中客户端的综合得分
        scores = {}
        total_score = 0
        for client_id in selected_clients:
            sv = shapley_values.get(client_id, 0.0)
            dv = diversities.get(client_id, 0.0)
            fq = frequencies.get(client_id, 1.0) 
            score = (shapley_weight * sv +
                    diversity_weight * dv +
                    participation_weight * (1 - fq))
            scores[client_id] = score
            total_score += score
        
        # 归一化为权重
        if total_score > 0:
            weights = {k: v/total_score for k, v in scores.items()}
        else:
            # 平均分配
            equal_weight = 1.0 / len(selected_clients)
            weights = {k: equal_weight for k in selected_clients}
        
        return weights
    
    @staticmethod
    def adaptive_clipping(grad_norm: float, 
                         last_grad_norm: float, 
                         shapley_value: float,
                         last_clip: float,
                         f_param: float = 0.8,
                         u_param: float = 0.5) -> float:
        """
        自适应梯度裁剪
        
        Args:
            grad_norm: 当前梯度范数
            last_grad_norm: 上一轮梯度范数
            shapley_value: Shapley值
            last_clip: 上一轮裁剪阈值
            f_param: f参数
            u_param: u参数
            
        Returns:
            new_clip: 新的裁剪阈值
        """
        if last_grad_norm > 0:
            gradient_change = (grad_norm - last_grad_norm) / last_grad_norm
        else:
            gradient_change = 0
        
        # 计算新的裁剪阈值
        new_clip = last_clip * (1 + f_param * shapley_value - u_param * gradient_change)
        
        # 确保裁剪阈值为正
        new_clip = max(new_clip, 1e-6)
        
        return new_clip
    
    @staticmethod
    def clip_gradients(model: torch.nn.Module, clip_norm: float) -> float:
        """
        裁剪梯度
        
        Args:
            model: 模型
            clip_norm: 裁剪阈值
            
        Returns:
            grad_norm: 裁剪前的梯度范数
        """
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        
        # 裁剪梯度
        clip_coef = clip_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return total_norm
    
    @staticmethod
    def add_dp_noise(model: torch.nn.Module, 
                    clip_norm: float, 
                    sigma: float,
                    sensitivity: float = 1.0) -> None:
        """
        添加差分隐私噪声
        
        Args:
            model: 模型
            clip_norm: 裁剪阈值
            sigma: 噪声尺度
            sensitivity: 敏感度
        """
        for param in model.parameters():
            if param.grad is not None:
                # 计算噪声
                noise_scale = sigma * sensitivity * clip_norm
                noise = torch.randn_like(param.grad.data) * noise_scale
                param.grad.data.add_(noise)
    
    @staticmethod
    # utils.py 中的修改部分

    def pseudo_label_training(model: torch.nn.Module, 
                         dataloader: torch.utils.data.DataLoader,
                         device: torch.device,
                         confidence_threshold: float = 0.9) -> float:

        model.train()
        criterion = torch.nn.CrossEntropyLoss()
    
        pseudo_count = 0
        total_count = 0
    
        for data, _ in dataloader:
           data = data.to(device)
        
        # 1. 生成伪标签：不需要梯度
           with torch.no_grad():
              outputs = model(data)
              probabilities = F.softmax(outputs, dim=1)
              max_probs, pseudo_labels = torch.max(probabilities, dim=1)
              mask = max_probs > confidence_threshold
        
        # 2. 计算损失并反向传播：需要梯度
           if mask.sum() > 0:
              pseudo_count += mask.sum().item()
              total_count += mask.size(0)
              
              pseudo_outputs = model(data[mask])          # 此步会构建计算图
              pseudo_loss = criterion(pseudo_outputs, pseudo_labels[mask])
              pseudo_loss.backward()                     # 现在可以正常反向传播
    
        pseudo_ratio = pseudo_count / max(total_count, 1)
        return pseudo_ratio
    
    @staticmethod
    def compute_fisher_diag(model: torch.nn.Module, 
                          dataloader: torch.utils.data.DataLoader,
                          device: torch.device) -> List[torch.Tensor]:
        """
        计算Fisher信息矩阵对角线（用于FedADDP）
        
        Args:
            model: 模型
            dataloader: 数据加载器
            device: 设备
            
        Returns:
            fisher_diag: Fisher对角线
        """
        model.eval()
        fisher_diag = [torch.zeros_like(param) for param in model.parameters()]
        
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            
            model.zero_grad()
            outputs = model(data)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            
            for i, param in enumerate(model.parameters()):
                if param.grad is not None:
                    fisher_diag[i] += param.grad.data ** 2
        
        # 平均
        n_batches = len(dataloader)
        fisher_diag = [f / n_batches for f in fisher_diag]
        
        return fisher_diag