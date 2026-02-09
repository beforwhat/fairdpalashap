# visualization/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os
import json

class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, plot_dir: str = './plots'):
        """
        初始化可视化器
        
        Args:
            plot_dir: 图表保存目录
        """
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = sns.color_palette("husl", 8)
        self.markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    def plot_accuracy_curves(self, results_dict: Dict[str, Dict], 
                           save_name: str = 'accuracy_curves.png'):
        """
        绘制准确率曲线
        
        Args:
            results_dict: 结果字典 {method_name: results}
            save_name: 保存文件名
        """
        plt.figure(figsize=(12, 8))
        
        for i, (method_name, results) in enumerate(results_dict.items()):
            if 'round_accuracies' in results:
                accuracies = results['round_accuracies']
                rounds = range(1, len(accuracies) + 1)
                
                plt.plot(rounds, accuracies, 
                        color=self.colors[i % len(self.colors)],
                        marker=self.markers[i % len(self.markers)],
                        markersize=8,
                        linewidth=2,
                        label=f'{method_name} (最终: {accuracies[-1]:.2f}%)')
        
        plt.xlabel('训练轮次', fontsize=14)
        plt.ylabel('全局准确率 (%)', fontsize=14)
        plt.title('联邦学习方法准确率比较', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # 保存图表
        save_path = os.path.join(self.plot_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"准确率曲线已保存至: {save_path}")
    
    def plot_fairness_comparison(self, results_dict: Dict[str, Dict],
                               save_name: str = 'fairness_comparison.png'):
        """
        绘制公平性比较
        
        Args:
            results_dict: 结果字典
            save_name: 保存文件名
        """
        methods = []
        fairness_variances = []
        
        for method_name, results in results_dict.items():
            if 'fairness_variances' in results and results['fairness_variances']:
                methods.append(method_name)
                fairness_variances.append(np.mean(results['fairness_variances']))
        
        if not methods:
            print("没有公平性数据可绘制")
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, fairness_variances, color=self.colors[:len(methods)])
        
        # 添加数值标签
        for bar, variance in zip(bars, fairness_variances):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{variance:.4f}', ha='center', va='bottom', fontsize=11)
        
        plt.xlabel('方法', fontsize=14)
        plt.ylabel('平均公平性方差', fontsize=14)
        plt.title('联邦学习方法公平性比较', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 保存图表
        save_path = os.path.join(self.plot_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"公平性比较图已保存至: {save_path}")
    
    def plot_data_distribution(self, data_distributions: Dict[int, List[int]],
                             num_clients_to_show: int = 10,
                             save_name: str = 'data_distribution.png'):
        """
        绘制数据分布热图
        
        Args:
            data_distributions: 数据分布字典
            num_clients_to_show: 显示的客户端数量
            save_name: 保存文件名
        """
        if not data_distributions:
            print("没有数据分布数据")
            return
        
        # 选择要显示的客户端
        client_ids = list(data_distributions.keys())[:num_clients_to_show]
        num_classes = len(data_distributions[client_ids[0]])
        
        # 创建数据矩阵
        data_matrix = np.zeros((len(client_ids), num_classes))
        for i, client_id in enumerate(client_ids):
            data_matrix[i, :] = data_distributions[client_id]
        
        # 创建热图
        plt.figure(figsize=(14, 8))
        sns.heatmap(data_matrix, 
                   cmap='YlOrRd',
                   annot=True,
                   fmt='.0f',
                   cbar_kws={'label': '样本数量'},
                   xticklabels=[f'类别{i}' for i in range(num_classes)],
                   yticklabels=[f'客户端{id}' for id in client_ids])
        
        plt.xlabel('类别', fontsize=14)
        plt.ylabel('客户端', fontsize=14)
        plt.title('客户端数据分布热图 (非IID)', fontsize=16, fontweight='bold')
        
        # 保存图表
        save_path = os.path.join(self.plot_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"数据分布热图已保存至: {save_path}")
    
    def plot_convergence_speed(self, results_dict: Dict[str, Dict],
                             save_name: str = 'convergence_speed.png'):
        """
        绘制收敛速度比较
        
        Args:
            results_dict: 结果字典
            save_name: 保存文件名
        """
        methods = []
        convergence_rounds = []
        
        for method_name, results in results_dict.items():
            if 'convergence_round' in results and results['convergence_round']:
                methods.append(method_name)
                convergence_rounds.append(results['convergence_round'])
        
        if not methods:
            print("没有收敛速度数据")
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, convergence_rounds, color=self.colors[:len(methods)])
        
        # 添加数值标签
        for bar, rounds in zip(bars, convergence_rounds):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rounds}', ha='center', va='bottom', fontsize=11)
        
        plt.xlabel('方法', fontsize=14)
        plt.ylabel('收敛轮次', fontsize=14)
        plt.title('联邦学习方法收敛速度比较', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 保存图表
        save_path = os.path.join(self.plot_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"收敛速度图已保存至: {save_path}")
    
    def plot_comprehensive_comparison(self, results_dict: Dict[str, Dict],
                                    save_name: str = 'comprehensive_comparison.png'):
        """
        绘制综合比较雷达图
        
        Args:
            results_dict: 结果字典
            save_name: 保存文件名
        """
        # 提取指标
        metrics = ['准确率', '公平性', '收敛速度', '稳定性']
        methods = list(results_dict.keys())
        
        # 归一化指标
        normalized_data = {}
        for method in methods:
            results = results_dict[method]
            
            # 准确率 (越高越好)
            accuracy = results.get('final_global_accuracy', 0) / 100
            
            # 公平性 (越低越好，取倒数)
            fairness = results.get('mean_fairness_variance', 1)
            fairness = 1 / (fairness + 0.001)  # 避免除零
            
            # 收敛速度 (越快越好，取倒数)
            convergence = results.get('convergence_round', 100)
            convergence = 100 / (convergence + 1)
            
            # 稳定性 (准确率标准差越小越好，取倒数)
            if 'round_accuracies' in results:
                stability = np.std(results['round_accuracies'])
                stability = 100 / (stability + 1)
            else:
                stability = 50
            
            normalized_data[method] = [accuracy, fairness, convergence, stability]
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for i, (method, values) in enumerate(normalized_data.items()):
            values += values[:1]  # 闭合
            ax.plot(angles, values, 'o-', linewidth=2, 
                   color=self.colors[i % len(self.colors)],
                   label=method, markersize=8)
            ax.fill(angles, values, alpha=0.25, color=self.colors[i % len(self.colors)])
        
        # 设置坐标
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=14)
        ax.set_ylim(0, 1)
        
        plt.title('联邦学习方法综合性能比较', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
        
        # 保存图表
        save_path = os.path.join(self.plot_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"综合比较雷达图已保存至: {save_path}")
    
    def generate_summary_table(self, results_dict: Dict[str, Dict],
                             save_name: str = 'results_summary.csv'):
        """
        生成结果摘要表格
        
        Args:
            results_dict: 结果字典
            save_name: 保存文件名
        """
        summary_data = []
        
        for method_name, results in results_dict.items():
            row = {
                '方法': method_name,
                '最终准确率 (%)': results.get('final_global_accuracy', 0),
                '最佳准确率 (%)': results.get('best_global_accuracy', 0),
                '平均客户端准确率 (%)': results.get('mean_client_accuracy', 0),
                '平均公平性方差': results.get('mean_fairness_variance', 0),
                '收敛轮次': results.get('convergence_round', 'N/A'),
                '收敛速度': f"{results.get('convergence_speed', 0):.3f}",
                '平均训练损失': results.get('mean_training_loss', 0),
                '平均梯度范数': results.get('mean_gradient_norm', 0)
            }
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # 保存为CSV
        csv_path = os.path.join(self.plot_dir, save_name)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 打印表格
        print("\n" + "="*80)
        print("实验结果摘要")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        print(f"详细结果已保存至: {csv_path}")
        
        return df