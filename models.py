# models/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    """简单的多层感知机"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list = [64, 32]):
        """
        初始化MLP
        
        Args:
            input_dim: 输入维度
            num_classes: 类别数
            hidden_dims: 隐藏层维度列表
        """
        super(SimpleMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.model(x)


class SimpleCNN(nn.Module):
    """简单的卷积神经网络"""
    
    def __init__(self, input_channels: int, num_classes: int):
        """
        初始化CNN
        
        Args:
            input_channels: 输入通道数
            num_classes: 类别数
        """
        super(SimpleCNN, self).__init__()
        
        # 对于MNIST/FEMNIST (1通道)
        if input_channels == 1:
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 28x28 -> 14x14 -> 7x7
            self.fc2 = nn.Linear(128, num_classes)
        
        # 对于CIFAR10/SVHN (3通道)
        elif input_channels == 3:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 4 * 4, 256)  # 32x32 -> 16x16 -> 8x8 -> 4x4
            self.fc2 = nn.Linear(256, num_classes)
        
        else:
            raise ValueError(f"不支持的输入通道数: {input_channels}")
        
        self.dropout = nn.Dropout(0.3)
        self.input_channels = input_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 如果是扁平化的数据，重塑为图像
        if len(x.shape) == 2:
            if self.input_channels == 1:
                # 28x28图像
                x = x.view(-1, 1, 28, 28)
            elif self.input_channels == 3:
                # 32x32图像
                x = x.view(-1, 3, 32, 32)
        
        if self.input_channels == 1:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
        
        elif self.input_channels == 3:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 4 * 4)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
        
        return x


class ModelFactory:
    """模型工厂"""
    
    @staticmethod
    def create_model(model_type: str, dataset_params: dict) -> nn.Module:
        """
        创建模型
        
        Args:
            model_type: 模型类型 ('MLP' 或 'CNN')
            dataset_params: 数据集参数
            
        Returns:
            model: 模型实例
        """
        input_dim = dataset_params.get('input_dim', 20)
        num_classes = dataset_params.get('num_classes', 5)
        channels = dataset_params.get('channels', 1)
        
        if model_type == 'MLP':
            return SimpleMLP(input_dim, num_classes)
        
        elif model_type == 'CNN':
            return SimpleCNN(channels, num_classes)
        
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    @staticmethod
    def get_default_model(dataset_name: str) -> nn.Module:
        """
        根据数据集获取默认模型
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            model: 默认模型
        """
        dataset_mapping = {
            'Synthetic': ('MLP', {'input_dim': 20, 'num_classes': 5, 'channels': 1}),
            'MNIST': ('CNN', {'input_dim': 28*28, 'num_classes': 10, 'channels': 1}),
            'FEMNIST': ('CNN', {'input_dim': 28*28, 'num_classes': 62, 'channels': 1}),
            'CIFAR10': ('CNN', {'input_dim': 32*32*3, 'num_classes': 10, 'channels': 3}),
            'SVHN': ('CNN', {'input_dim': 32*32*3, 'num_classes': 10, 'channels': 3})
        }
        
        if dataset_name not in dataset_mapping:
            dataset_name = 'Synthetic'
        
        model_type, dataset_params = dataset_mapping[dataset_name]
        return ModelFactory.create_model(model_type, dataset_params)