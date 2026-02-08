import torch
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
import copy
from util import (
    compute_shapley_approximation, 
    compute_entropy, 
    compute_participation_score,
    normalize_dict_values
)

class FedShapleyServer:
    def __init__(self,
                 global_model: torch.nn.Module,
                 num_clients: int,
                 sample_rate: float,
                 device: torch.device,
                 shapley_weight: float = 0.5,
                 diversity_weight: float = 0.3,
                 participation_weight: float = 0.2,
                 f_param: float = 0.8,
                 u_param: float = 0.5,
                 min_clip_threshold: float = 0.1):
        """
        Server for federated learning with Shapley-based client selection
        
        Args:
            global_model: Global model
            num_clients: Number of clients
            sample_rate: Client sampling rate per round
            device: Computation device
            shapley_weight: Weight for Shapley value in client selection
            diversity_weight: Weight for data diversity in client selection
            participation_weight: Weight for participation frequency in client selection
            f_param: Parameter f in adaptive clipping formula
            u_param: Parameter u in adaptive clipping formula
            min_clip_threshold: Minimum clipping threshold
        """
        self.global_model = global_model
        self.num_clients = num_clients
        self.sample_rate = sample_rate
        self.device = device
        
        # 权重参数
        self.shapley_weight = shapley_weight
        self.diversity_weight = diversity_weight
        self.participation_weight = participation_weight
        
        # 自适应裁剪参数
        self.f_param = f_param
        self.u_param = u_param
        self.min_clip_threshold = min_clip_threshold
        
        # 客户端指标
        self.client_shapley_values = {i: 0.0 for i in range(num_clients)}
        self.client_diversity_scores = {i: 0.0 for i in range(num_clients)}
        self.client_participation_counts = {i: 0 for i in range(num_clients)}
        self.client_gradient_norms = {i: [] for i in range(num_clients)}  # 存储历史梯度范数
        self.client_data_sizes = {i: 0 for i in range(num_clients)}
        
        # 历史记录
        self.round_shapley_values = []  # 每轮的Shapley值
        self.round_accuracies = []  # 每轮的准确率
        
        # 裁剪阈值
        self.clip_thresholds = {i: 1.0 for i in range(num_clients)}  # 初始裁剪阈值
        
    def select_clients(self, round_idx: int) -> List[int]:
        """
        基于Shapley值、多样性和参与度选择客户端
        
        Args:
            round_idx: Current round index
            
        Returns:
            List of selected client IDs
        """
        num_selected = max(1, int(self.num_clients * self.sample_rate))
        
        # 归一化指标
        shapley_norm = normalize_dict_values(self.client_shapley_values)
        diversity_norm = normalize_dict_values(self.client_diversity_scores)
        
        # 计算参与度分数（参与越少分数越高）
        max_participation = max(self.client_participation_counts.values()) + 1
        participation_scores = compute_participation_score(
            self.client_participation_counts, max_participation
        )
        participation_norm = normalize_dict_values(participation_scores)
        
        # 计算选择概率
        selection_probs = {}
        for client_id in range(self.num_clients):
            prob = (self.shapley_weight * shapley_norm.get(client_id, 0) +
                   self.diversity_weight * diversity_norm.get(client_id, 0) +
                   self.participation_weight * participation_norm.get(client_id, 0))
            
            # 确保概率非负
            selection_probs[client_id] = max(prob, 0.01)  # 最小概率为0.01
        
        # 归一化概率
        total_prob = sum(selection_probs.values())
        if total_prob > 0:
            selection_probs = {cid: prob/total_prob for cid, prob in selection_probs.items()}
        
        # 根据概率选择客户端
        client_ids = list(selection_probs.keys())
        probs = list(selection_probs.values())
        
        selected_clients = random.choices(client_ids, weights=probs, k=num_selected)
        
        # 更新参与次数
        for client_id in selected_clients:
            self.client_participation_counts[client_id] += 1
        
        return selected_clients
    
    def compute_aggregation_weights(self, selected_clients: List[int]) -> Dict[int, float]:
        """
        计算聚合权重（与选择概率相同的权重配置）
        
        Args:
            selected_clients: List of selected client IDs
            
        Returns:
            Aggregation weights for each selected client
        """
        # 获取选中客户端的指标
        selected_shapley = {cid: self.client_shapley_values[cid] for cid in selected_clients}
        selected_diversity = {cid: self.client_diversity_scores[cid] for cid in selected_clients}
        selected_data_sizes = {cid: self.client_data_sizes[cid] for cid in selected_clients}
        
        # 归一化指标
        shapley_norm = normalize_dict_values(selected_shapley)
        diversity_norm = normalize_dict_values(selected_diversity)
        
        # 计算参与度分数（参与越少权重越高）
        selected_participation = {cid: self.client_participation_counts[cid] for cid in selected_clients}
        max_participation = max(selected_participation.values()) + 1
        participation_scores = compute_participation_score(selected_participation, max_participation)
        participation_norm = normalize_dict_values(participation_scores)
        
        # 计算聚合权重
        agg_weights = {}
        for client_id in selected_clients:
            weight = (self.shapley_weight * shapley_norm.get(client_id, 0) +
                     self.diversity_weight * diversity_norm.get(client_id, 0) +
                     self.participation_weight * participation_norm.get(client_id, 0))
            
            # 考虑数据量
            data_weight = selected_data_sizes[client_id] / sum(selected_data_sizes.values())
            
            # 综合权重
            agg_weights[client_id] = max(weight * data_weight, 0.01)  # 最小权重
        
        # 归一化权重
        total_weight = sum(agg_weights.values())
        if total_weight > 0:
            agg_weights = {cid: w/total_weight for cid, w in agg_weights.items()}
        
        return agg_weights
    
    def compute_adaptive_clip_threshold(self, 
                                       client_id: int, 
                                       current_gradient_norm: float,
                                       round_idx: int) -> float:
        """
        计算自适应裁剪阈值
        
        Args:
            client_id: Client ID
            current_gradient_norm: Current gradient norm
            round_idx: Current round index
            
        Returns:
            Adaptive clipping threshold
        """
        if round_idx == 0:
            # 第一轮使用初始阈值
            return self.clip_thresholds[client_id]
        
        # 获取历史梯度范数
        grad_history = self.client_gradient_norms[client_id]
        if len(grad_history) < 2:
            # 没有足够历史数据
            return self.clip_thresholds[client_id]
        
        # 计算梯度变化趋势 n_it
        prev_norm = grad_history[-1] if grad_history else 1.0
        prev_prev_norm = grad_history[-2] if len(grad_history) >= 2 else prev_norm
        
        if prev_prev_norm < 1e-10:
            n_it = 0.0
        else:
            n_it = (prev_norm - prev_prev_norm) / prev_prev_norm
        
        # 获取Shapley值（归一化到[0, 1]）
        all_shapley = list(self.client_shapley_values.values())
        if all_shapley:
            min_shapley = min(all_shapley)
            max_shapley = max(all_shapley)
            if max_shapley - min_shapley > 1e-10:
                SV_i = (self.client_shapley_values[client_id] - min_shapley) / (max_shapley - min_shapley)
            else:
                SV_i = 0.5
        else:
            SV_i = 0.5
        
        # 自适应裁剪公式: T_it = T_(t-1) * (1 + f*SV_i - u * n_it)
        prev_threshold = self.clip_thresholds[client_id]
        new_threshold = prev_threshold * (1 + self.f_param * SV_i - self.u_param * n_it)
        
        # 确保阈值不小于最小值
        new_threshold = max(new_threshold, self.min_clip_threshold)
        
        return new_threshold
    
    def update_client_metrics(self,
                             client_accuracies: Dict[int, float],
                             client_diversities: Dict[int, float],
                             client_gradient_norms: Dict[int, float],
                             client_data_sizes: Dict[int, int],
                             round_idx: int):
        """
        更新客户端指标
        
        Args:
            client_accuracies: Client accuracies
            client_diversities: Client data diversities
            client_gradient_norms: Client gradient norms
            client_data_sizes: Client data sizes
            round_idx: Current round index
        """
        # 更新数据大小
        for client_id, data_size in client_data_sizes.items():
            self.client_data_sizes[client_id] = data_size
        
        # 更新多样性
        for client_id, diversity in client_diversities.items():
            self.client_diversity_scores[client_id] = diversity
        
        # 更新梯度范数历史
        for client_id, grad_norm in client_gradient_norms.items():
            if client_id in self.client_gradient_norms:
                self.client_gradient_norms[client_id].append(grad_norm)
                # 只保留最近10个值
                if len(self.client_gradient_norms[client_id]) > 10:
                    self.client_gradient_norms[client_id] = self.client_gradient_norms[client_id][-10:]
            else:
                self.client_gradient_norms[client_id] = [grad_norm]
        
        # 记录准确率历史
        self.round_accuracies.append(client_accuracies)
        
        # 计算并更新Shapley值
        shapley_values = compute_shapley_approximation(
            client_accuracies, 
            self.client_data_sizes,
            self.round_accuracies
        )
        
        # 使用指数移动平均更新Shapley值
        alpha = 0.3  # 平滑系数
        for client_id, new_shapley in shapley_values.items():
            old_shapley = self.client_shapley_values.get(client_id, 0.0)
            self.client_shapley_values[client_id] = alpha * new_shapley + (1 - alpha) * old_shapley
        
        # 记录Shapley值历史
        self.round_shapley_values.append(self.client_shapley_values.copy())
    
    def aggregate_models(self,
                        client_models: Dict[int, torch.nn.Module],
                        agg_weights: Dict[int, float]) -> torch.nn.Module:
        """
        聚合客户端模型
        
        Args:
            client_models: Client models
            agg_weights: Aggregation weights
            
        Returns:
            Updated global model
        """
        # 获取全局模型状态
        global_state = self.global_model.state_dict()
        
        # 初始化加权平均
        weighted_state = {}
        for key in global_state.keys():
            weighted_state[key] = torch.zeros_like(global_state[key])
        
        # 加权聚合
        for client_id, client_model in client_models.items():
            weight = agg_weights[client_id]
            client_state = client_model.state_dict()
            
            for key in weighted_state.keys():
                if key in client_state:
                    weighted_state[key] += weight * client_state[key].to(self.device)
        
        # 更新全局模型
        self.global_model.load_state_dict(weighted_state)
        
        return self.global_model
    
    def get_client_metrics(self, client_id: int) -> Dict[str, float]:
        """获取客户端指标"""
        return {
            'shapley': self.client_shapley_values.get(client_id, 0.0),
            'diversity': self.client_diversity_scores.get(client_id, 0.0),
            'participation': self.client_participation_counts.get(client_id, 0),
            'clip_threshold': self.clip_thresholds.get(client_id, 1.0)
        }
    
    def update_clip_threshold(self, client_id: int, threshold: float):
        """更新客户端裁剪阈值"""
        self.clip_thresholds[client_id] = threshold