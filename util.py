import numpy as np
import torch
from typing import Dict, List, Tuple
from scipy.special import comb
from itertools import chain, combinations

def compute_shapley_approximation(client_accuracies: Dict[int, float],
                                 client_data_sizes: Dict[int, int],
                                 round_accuracies: List[Dict[int, float]]) -> Dict[int, float]:
    """
    近似计算Shapley值（基于边际贡献的简化方法）
    
    Args:
        client_accuracies: 当前轮各客户端准确率 {client_id: accuracy}
        client_data_sizes: 客户端数据量 {client_id: data_size}
        round_accuracies: 历史各轮准确率 [{client_id: accuracy}]
    
    Returns:
        近似Shapley值 {client_id: shapley_value}
    """
    num_clients = len(client_accuracies)
    shapley_values = {i: 0.0 for i in client_accuracies.keys()}
    
    if len(round_accuracies) == 0:
        # 第一轮，平均分配
        avg_accuracy = np.mean(list(client_accuracies.values()))
        for client_id in client_accuracies.keys():
            shapley_values[client_id] = client_accuracies[client_id] - avg_accuracy
        return shapley_values
    
    # 计算每个客户端的边际贡献（基于最近几轮的表现）
    recent_rounds = min(5, len(round_accuracies))
    recent_accuracies = round_accuracies[-recent_rounds:]
    
    for client_id in client_accuracies.keys():
        # 计算客户端的历史贡献
        client_history_acc = []
        for round_acc in recent_accuracies:
            if client_id in round_acc:
                client_history_acc.append(round_acc[client_id])
        
        if len(client_history_acc) > 0:
            # 边际贡献 = 当前准确率 - 历史平均准确率
            historical_avg = np.mean(client_history_acc)
            marginal_contribution = max(0, client_accuracies[client_id] - historical_avg)
        else:
            marginal_contribution = client_accuracies[client_id]
        
        # 考虑数据量权重
        data_weight = client_data_sizes[client_id] / sum(client_data_sizes.values())
        
        # 近似Shapley值（考虑边际贡献和数据量）
        shapley_values[client_id] = marginal_contribution * data_weight
    
    return shapley_values

def compute_data_diversity(data_loader) -> List[float]:
    """计算数据多样性（类别分布）"""
    # 统计类别分布
    class_counts = {}
    total_samples = 0
    
    for _, labels in data_loader:
        labels_np = labels.numpy() if torch.is_tensor(labels) else np.array(labels)
        for label in labels_np:
            class_counts[label] = class_counts.get(label, 0) + 1
            total_samples += 1
    
    # 计算分布
    distribution = []
    num_classes = max(class_counts.keys()) + 1 if class_counts else 0
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        distribution.append(count / max(total_samples, 1))
    
    return distribution

def compute_entropy(distribution: List[float]) -> float:
    """计算信息熵作为多样性度量"""
    if not distribution:
        return 0.0
    
    dist_array = np.array(distribution)
    # 归一化
    if dist_array.sum() > 0:
        dist_array = dist_array / dist_array.sum()
    else:
        return 0.0
    
    # 计算熵
    entropy = -np.sum(dist_array * np.log(dist_array + 1e-10))
    return entropy

def compute_participation_score(participation_counts: Dict[int, int], 
                               max_participation: int) -> Dict[int, float]:
    """
    计算参与度分数（参与越少分数越高）
    """
    participation_scores = {}
    for client_id, count in participation_counts.items():
        # 归一化到[0, 1]，参与越少分数越高
        score = 1.0 - (count / max_participation) if max_participation > 0 else 1.0
        participation_scores[client_id] = max(score, 0.1)  # 最小0.1
    return participation_scores

def normalize_dict_values(data_dict: Dict[int, float]) -> Dict[int, float]:
    """归一化字典值到[0, 1]范围"""
    if not data_dict:
        return {}
    
    values = list(data_dict.values())
    min_val = min(values)
    max_val = max(values)
    
    if max_val - min_val < 1e-10:
        # 所有值相等，返回均匀分布
        return {k: 1.0/len(data_dict) for k in data_dict.keys()}
    
    normalized = {}
    for k, v in data_dict.items():
        normalized[k] = (v - min_val) / (max_val - min_val)
    
    return normalized

def compute_noise_scale(target_epsilon: float,
                       target_delta: float,
                       num_rounds: int,
                       sample_rate: float) -> float:
    """
    计算DP噪声尺度（高斯机制）
    
    Args:
        target_epsilon: 总隐私预算ε
        target_delta: 隐私参数δ
        num_rounds: 总通信轮数
        sample_rate: 每轮采样率
    
    Returns:
        噪声尺度sigma
    """
    # 每轮的隐私预算
    epsilon_per_round = target_epsilon / num_rounds
    
    # 计算高斯机制的sigma
    # 对于高斯机制，σ = sqrt(2 * ln(1.25/δ)) / ε * 敏感度
    # 这里假设敏感度为1（经过梯度裁剪）
    sigma = np.sqrt(2 * np.log(1.25 / target_delta)) / epsilon_per_round
    
    # 考虑采样率的影响
    sigma = sigma / sample_rate if sample_rate > 0 else sigma
    
    return sigma

def compute_gradient_trend(current_norm: float, 
                          previous_norm: float, 
                          previous_previous_norm: float = None) -> float:
    """
    计算梯度变化趋势
    n_it = (本轮梯度范数 - 上轮梯度范数) / 上轮梯度范数
    """
    if previous_norm < 1e-10:
        return 0.0
    
    if previous_previous_norm is None:
        # 如果没有前两轮数据，用简单方法
        return (current_norm - previous_norm) / previous_norm
    
    # 使用更平滑的方法（考虑趋势）
    trend = (current_norm - previous_norm) / previous_norm
    return trend