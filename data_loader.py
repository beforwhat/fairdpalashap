# data/data_loader.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from typing import List, Tuple, Dict
import os

class DatasetLoader:
    """数据集加载器"""
    
    def __init__(self, dataset_name: str, data_dir: str = './data'):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.dataset_params = {
            'Synthetic': {'input_dim': 20, 'num_classes': 5, 'channels': 1},
            'MNIST': {'input_dim': 28 * 28, 'num_classes': 10, 'channels': 1},
            'CIFAR10': {'input_dim': 32 * 32 * 3, 'num_classes': 10, 'channels': 3},
            'FEMNIST': {'input_dim': 28 * 28, 'num_classes': 62, 'channels': 1},
            'SVHN': {'input_dim': 32 * 32 * 3, 'num_classes': 10, 'channels': 3}
        }
    
    def get_dataset_params(self):
        return self.dataset_params.get(self.dataset_name, {
            'input_dim': 20, 'num_classes': 5, 'channels': 1
        })
    
    def load_full_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载完整训练集"""
        if self.dataset_name == 'Synthetic':
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
                root=self.data_dir, train=True, download=True, transform=transform
            )
            data = torch.stack([img.view(-1) for img, _ in train_dataset])
            labels = torch.tensor([label for _, label in train_dataset])
            return data, labels
        
        elif self.dataset_name == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, download=True, transform=transform
            )
            data = torch.stack([img.view(-1) for img, _ in train_dataset])
            labels = torch.tensor([label for _, label in train_dataset])
            return data, labels
        
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
    
    def distribute_data_non_iid(self,
                               data: torch.Tensor,
                               labels: torch.Tensor,
                               num_clients: int,
                               samples_per_client: int,
                               dir_alpha: float = 0.5,
                               data_skew_type: str = 'label_distribution'):
        """
        非IID分配训练数据（每个客户端获得 samples_per_client 个样本）
        不再从训练集中划分测试集，全部用于训练。
        
        返回:
            train_loaders: 每个客户端的 DataLoader
            train_datasets: 每个客户端的 TensorDataset（用于 ALA）
            data_distributions: 每个客户端每类的样本数分布
        """
        num_classes = len(torch.unique(labels))
        data_distributions = {i: [0] * num_classes for i in range(num_clients)}
        
        # 按类别索引
        class_indices = {}
        for c in range(num_classes):
            class_indices[c] = torch.where(labels == c)[0].numpy()
        
        client_indices = {i: [] for i in range(num_clients)}
        
        if data_skew_type == 'label_distribution':
            for c in range(num_classes):
                indices_c = class_indices[c]
                n_samples_c = len(indices_c)
                if n_samples_c == 0:
                    continue
                
                proportions = np.random.dirichlet(np.repeat(dir_alpha, num_clients))
                allocations = (proportions * samples_per_client).astype(int)
                total_allocated = allocations.sum()
                diff = samples_per_client - total_allocated
                if diff > 0:
                    allocations[np.random.choice(num_clients, diff, replace=False)] += 1
                
                np.random.shuffle(indices_c)
                pos = 0
                for i in range(num_clients):
                    n_alloc = min(allocations[i], n_samples_c - pos)
                    if n_alloc > 0:
                        assigned_indices = indices_c[pos:pos + n_alloc]
                        client_indices[i].extend(assigned_indices)
                        data_distributions[i][c] += len(assigned_indices)
                        pos += n_alloc
                
                # 若该类样本不足，从其他类补充
                if pos < samples_per_client:
                    for i in range(num_clients):
                        remaining = allocations[i] - data_distributions[i][c]
                        if remaining > 0:
                            other_classes = [oc for oc in range(num_classes) if oc != c]
                            if other_classes:
                                random_class = np.random.choice(other_classes)
                                if len(class_indices[random_class]) > 0:
                                    n_take = min(remaining, len(class_indices[random_class]))
                                    take_indices = np.random.choice(class_indices[random_class], n_take, replace=False)
                                    client_indices[i].extend(take_indices)
                                    data_distributions[i][random_class] += n_take
                                    class_indices[random_class] = np.setdiff1d(
                                        class_indices[random_class], take_indices
                                    )
        
        elif data_skew_type == 'quantity':
            proportions = np.random.dirichlet(np.repeat(dir_alpha, num_clients))
            total_samples = num_clients * samples_per_client
            allocations = (proportions * total_samples).astype(int)
            diff = total_samples - allocations.sum()
            if diff > 0:
                allocations[np.random.choice(num_clients, diff, replace=False)] += 1
            
            all_indices = torch.arange(len(data)).numpy()
            np.random.shuffle(all_indices)
            pos = 0
            for i in range(num_clients):
                n_alloc = allocations[i]
                if n_alloc > 0:
                    assigned_indices = all_indices[pos:pos + n_alloc]
                    client_indices[i].extend(assigned_indices)
                    for idx in assigned_indices:
                        label = labels[idx].item()
                        data_distributions[i][label] += 1
                    pos += n_alloc
        
        # 创建训练 DataLoader 和数据集（不再划分测试集）
        train_loaders = []
        train_datasets = []
        for i in range(num_clients):
            indices = client_indices[i]
            if len(indices) == 0:
                # 极少情况：若客户端未分配到任何样本，创建一个空数据集
                empty_data = torch.zeros(0, data.shape[1])
                empty_labels = torch.zeros(0, dtype=torch.long)
                train_dataset = TensorDataset(empty_data, empty_labels)
            else:
                train_dataset = TensorDataset(data[indices], labels[indices])
            
            train_datasets.append(train_dataset)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            train_loaders.append(train_loader)
        
        return train_loaders, train_datasets, data_distributions
    
    def distribute_iid_test_data(self,
                                test_data: torch.Tensor,
                                test_labels: torch.Tensor,
                                num_clients: int,
                                samples_per_client: int):
        """
        将全局 IID 测试集均匀、随机分配给每个客户端
        返回: 每个客户端的测试 DataLoader 列表
        """
        test_loaders = []
        total_needed = num_clients * samples_per_client
        
        # 若全局测试集不足，进行重复采样
        if len(test_data) < total_needed:
            repeat_times = (total_needed // len(test_data)) + 1
            test_data = test_data.repeat(repeat_times, 1)[:total_needed]
            test_labels = test_labels.repeat(repeat_times)[:total_needed]
        
        # 随机打乱
        indices = torch.randperm(len(test_data))
        for i in range(num_clients):
            start = i * samples_per_client
            end = (i + 1) * samples_per_client
            client_test_data = test_data[indices[start:end]]
            client_test_labels = test_labels[indices[start:end]]
            test_dataset = TensorDataset(client_test_data, client_test_labels)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            test_loaders.append(test_loader)
        
        return test_loaders
    
    def distribute_data_iid(self,
                           data: torch.Tensor,
                           labels: torch.Tensor,
                           num_clients: int,
                           samples_per_client: int):
        """
        IID数据分配（保留，但本实验中未使用）
        """
        num_classes = len(torch.unique(labels))
        indices = torch.randperm(len(data))
        data = data[indices]
        labels = labels[indices]
        
        train_datasets = []
        train_loaders = []
        data_distributions = {i: [0] * num_classes for i in range(num_clients)}
        
        for i in range(num_clients):
            start = i * samples_per_client
            end = (i + 1) * samples_per_client
            client_data = data[start:end]
            client_labels = labels[start:end]
            
            for label in client_labels:
                data_distributions[i][label.item()] += 1
            
            train_dataset = TensorDataset(client_data, client_labels)
            train_datasets.append(train_dataset)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            train_loaders.append(train_loader)
        
        return train_loaders, train_datasets, data_distributions
    
    def create_global_test_loader(self, test_size: int) -> DataLoader:
        """
        创建全局 IID 测试集 DataLoader
        """
        if self.dataset_name == 'Synthetic':
            input_dim = self.dataset_params['Synthetic']['input_dim']
            num_classes = self.dataset_params['Synthetic']['num_classes']
            test_data = torch.randn(test_size, input_dim)
            test_labels = torch.randint(0, num_classes, (test_size,))
            test_dataset = TensorDataset(test_data, test_labels)
            return DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        elif self.dataset_name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_dataset = torchvision.datasets.MNIST(
                root=self.data_dir, train=False, download=True, transform=transform
            )
            return DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        elif self.dataset_name == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, download=True, transform=transform
            )
            return DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        else:
            input_dim = self.get_dataset_params()['input_dim']
            num_classes = self.get_dataset_params()['num_classes']
            test_data = torch.randn(test_size, input_dim)
            test_labels = torch.randint(0, num_classes, (test_size,))
            test_dataset = TensorDataset(test_data, test_labels)
            return DataLoader(test_dataset, batch_size=32, shuffle=False)