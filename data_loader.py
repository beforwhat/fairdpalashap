# data/data_loader.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Optional
import os

class DatasetLoader:
    """数据集加载器"""
    
    def __init__(self, dataset_name: str, data_dir: str = './data'):
        """
        初始化数据集加载器
        
        Args:
            dataset_name: 数据集名称
            data_dir: 数据目录
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # 数据集参数
        self.dataset_params = {
            'Synthetic': {
                'input_dim': 20,
                'num_classes': 5,
                'channels': 1
            },
            'MNIST': {
                'input_dim': 28 * 28,
                'num_classes': 10,
                'channels': 1
            },
            'CIFAR10': {
                'input_dim': 32 * 32 * 3,
                'num_classes': 10,
                'channels': 3
            },
            'FEMNIST': {
                'input_dim': 28 * 28,
                'num_classes': 62,
                'channels': 1
            },
            'SVHN': {
                'input_dim': 32 * 32 * 3,
                'num_classes': 10,
                'channels': 3
            }
        }
    
    def get_dataset_params(self):
        """获取数据集参数"""
        return self.dataset_params.get(self.dataset_name, {
            'input_dim': 20,
            'num_classes': 5,
            'channels': 1
        })
    
    def load_full_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        加载完整数据集
        
        Returns:
            (data, labels): 数据和标签
        """
        if self.dataset_name == 'Synthetic':
            # 生成合成数据
            num_samples = 10000
            input_dim = self.dataset_params['Synthetic']['input_dim']
            num_classes = self.dataset_params['Synthetic']['num_classes']
            
            data = torch.randn(num_samples, input_dim)
            labels = torch.randint(0, num_classes, (num_samples,))
            
            return data, labels
        
        elif self.dataset_name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            train_dataset = torchvision.datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=True,
                transform=transform
            )
            
            # 转换为tensor
            data = []
            labels = []
            for img, label in train_dataset:
                data.append(img.view(-1))
                labels.append(label)
            
            data = torch.stack(data)
            labels = torch.tensor(labels)
            
            return data, labels
        
        elif self.dataset_name == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=transform
            )
            
            data = []
            labels = []
            for img, label in train_dataset:
                data.append(img.view(-1))
                labels.append(label)
            
            data = torch.stack(data)
            labels = torch.tensor(labels)
            
            return data, labels
        
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
    
    def distribute_data_non_iid(self, 
                               data: torch.Tensor, 
                               labels: torch.Tensor,
                               num_clients: int,
                               samples_per_client: int,
                               dir_alpha: float = 0.5,
                               data_skew_type: str = 'label_distribution') -> Tuple[List[DataLoader], List[DataLoader], Dict[int, List[int]]]:
        """
        非IID数据分配（使用Dirichlet分布）
        
        Args:
            data: 完整数据
            labels: 完整标签
            num_clients: 客户端数量
            samples_per_client: 每个客户端样本数
            dir_alpha: Dirichlet参数
            data_skew_type: 数据倾斜类型
            
        Returns:
            (train_loaders, test_loaders, data_distributions): 训练/测试加载器和数据分布
        """
        num_classes = len(torch.unique(labels))
        
        # 初始化数据分布记录
        data_distributions = {i: [0] * num_classes for i in range(num_clients)}
        
        # 按类别分割数据
        class_indices = {}
        for c in range(num_classes):
            class_indices[c] = torch.where(labels == c)[0].numpy()
        
        # 使用Dirichlet分布分配每个类别的样本
        client_indices = {i: [] for i in range(num_clients)}
        
        if data_skew_type == 'label_distribution':
            # 标签分布倾斜（主要方式）
            for c in range(num_classes):
                # 获取该类别的所有样本
                indices_c = class_indices[c]
                n_samples_c = len(indices_c)
                
                if n_samples_c == 0:
                    continue
                
                # 使用Dirichlet分布生成分配比例
                proportions = np.random.dirichlet(np.repeat(dir_alpha, num_clients))
                allocations = (proportions * samples_per_client).astype(int)
                
                # 确保分配总数正确
                total_allocated = allocations.sum()
                diff = samples_per_client - total_allocated
                if diff > 0:
                    allocations[np.random.choice(num_clients, diff, replace=False)] += 1
                
                # 随机打乱并分配样本
                np.random.shuffle(indices_c)
                pos = 0
                for i in range(num_clients):
                    n_alloc = min(allocations[i], n_samples_c - pos)
                    if n_alloc > 0:
                        assigned_indices = indices_c[pos:pos + n_alloc]
                        client_indices[i].extend(assigned_indices)
                        data_distributions[i][c] += len(assigned_indices)
                        pos += n_alloc
                
                # 如果没有足够样本，用其他数据补充
                if pos < samples_per_client:
                    for i in range(num_clients):
                        remaining = allocations[i] - data_distributions[i][c]
                        if remaining > 0:
                            # 从其他类别中随机抽取
                            other_classes = [oc for oc in range(num_classes) if oc != c]
                            if other_classes:
                                random_class = np.random.choice(other_classes)
                                if len(class_indices[random_class]) > 0:
                                    n_take = min(remaining, len(class_indices[random_class]))
                                    take_indices = np.random.choice(class_indices[random_class], n_take, replace=False)
                                    client_indices[i].extend(take_indices)
                                    data_distributions[i][random_class] += n_take
                                    # 从原列表中移除已取走的索引
                                    class_indices[random_class] = np.setdiff1d(
                                        class_indices[random_class], take_indices
                                    )
        
        elif data_skew_type == 'quantity':
            # 数量倾斜：某些客户端数据多，某些少
            proportions = np.random.dirichlet(np.repeat(dir_alpha, num_clients))
            total_samples = num_clients * samples_per_client
            allocations = (proportions * total_samples).astype(int)
            
            # 确保总数正确
            diff = total_samples - allocations.sum()
            if diff > 0:
                allocations[np.random.choice(num_clients, diff, replace=False)] += 1
            
            # 混合所有样本
            all_indices = torch.arange(len(data)).numpy()
            np.random.shuffle(all_indices)
            
            pos = 0
            for i in range(num_clients):
                n_alloc = allocations[i]
                if n_alloc > 0:
                    assigned_indices = all_indices[pos:pos + n_alloc]
                    client_indices[i].extend(assigned_indices)
                    # 记录分布
                    for idx in assigned_indices:
                        label = labels[idx].item()
                        data_distributions[i][label] += 1
                    pos += n_alloc
        
        # 创建数据加载器
        train_loaders = []
        test_loaders = []
        
        for i in range(num_clients):
            indices = client_indices[i]
            
            if len(indices) == 0:
                # 如果没有数据，创建空数据加载器
                empty_data = torch.zeros(0, data.shape[1])
                empty_labels = torch.zeros(0, dtype=torch.long)
                train_dataset = TensorDataset(empty_data, empty_labels)
                test_dataset = TensorDataset(empty_data, empty_labels)
            else:
                # 划分训练和测试集
                n_train = int(len(indices) * 0.8)
                train_indices = indices[:n_train]
                test_indices = indices[n_train:]
                
                # 创建数据集
                train_data = data[train_indices]
                train_labels = labels[train_indices]
                test_data = data[test_indices]
                test_labels = labels[test_indices]
                
                train_dataset = TensorDataset(train_data, train_labels)
                test_dataset = TensorDataset(test_data, test_labels)
            
            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)
        
        return train_loaders, test_loaders, data_distributions
    
    def distribute_data_iid(self,
                           data: torch.Tensor,
                           labels: torch.Tensor,
                           num_clients: int,
                           samples_per_client: int) -> Tuple[List[DataLoader], List[DataLoader], Dict[int, List[int]]]:
        """
        IID数据分配
        
        Returns:
            (train_loaders, test_loaders, data_distributions)
        """
        num_classes = len(torch.unique(labels))
        
        # 打乱数据
        indices = torch.randperm(len(data))
        data = data[indices]
        labels = labels[indices]
        
        # 初始化
        train_loaders = []
        test_loaders = []
        data_distributions = {i: [0] * num_classes for i in range(num_clients)}
        
        # 为每个客户端分配数据
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client
            
            client_data = data[start_idx:end_idx]
            client_labels = labels[start_idx:end_idx]
            
            # 记录分布
            for label in client_labels:
                data_distributions[i][label.item()] += 1
            
            # 划分训练和测试
            n_train = int(len(client_data) * 0.8)
            train_data = client_data[:n_train]
            train_labels = client_labels[:n_train]
            test_data = client_data[n_train:]
            test_labels = client_labels[n_train:]
            
            # 创建数据加载器
            train_dataset = TensorDataset(train_data, train_labels)
            test_dataset = TensorDataset(test_data, test_labels)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)
        
        return train_loaders, test_loaders, data_distributions
    
    def create_global_test_loader(self, test_size: int = 1000) -> DataLoader:
        """
        创建全局测试集
        
        Args:
            test_size: 测试集大小
            
        Returns:
            test_loader: 测试数据加载器
        """
        if self.dataset_name == 'Synthetic':
            input_dim = self.dataset_params['Synthetic']['input_dim']
            num_classes = self.dataset_params['Synthetic']['num_classes']
            
            test_data = torch.randn(test_size, input_dim)
            test_labels = torch.randint(0, num_classes, (test_size,))
            
            test_dataset = TensorDataset(test_data, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            return test_loader
        
        elif self.dataset_name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            test_dataset = torchvision.datasets.MNIST(
                root=self.data_dir,
                train=False,
                download=True,
                transform=transform
            )
            
            return DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        elif self.dataset_name == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=transform
            )
            
            return DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        else:
            # 对于其他数据集，创建合成测试集
            input_dim = self.get_dataset_params()['input_dim']
            num_classes = self.get_dataset_params()['num_classes']
            
            test_data = torch.randn(test_size, input_dim)
            test_labels = torch.randint(0, num_classes, (test_size,))
            
            test_dataset = TensorDataset(test_data, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            return test_loader