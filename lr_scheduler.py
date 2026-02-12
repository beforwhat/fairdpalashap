# utils/simple_lr_scheduler.py
import math

class CosineAnnealingLR:
    """简洁的余弦退火学习率调度器"""
    
    def __init__(self, initial_lr, total_epochs, warmup_epochs):
        """
        初始化余弦退火学习率调度器
        
        Args:
            initial_lr: 初始学习率 (默认0.01)
            total_epochs: 总训练轮数 (默认100)
            warmup_epochs: 预热轮数 (默认0)
        """
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.current_lr = initial_lr
        self.min_lr = initial_lr * 0.01  # 最小学习率为初始学习率的1%
    
    def step(self):
        """更新学习率到下一轮"""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # 预热阶段：线性增加学习率
            warmup_ratio = self.current_epoch / self.warmup_epochs
            warmup_lr = self.initial_lr * (0.1 + 0.9 * warmup_ratio)
            self.current_lr = warmup_lr
        else:
            # 余弦退火阶段
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)  # 确保不超过1
            
            # 余弦退火公式
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        
        return self.current_lr
    
    def get_lr(self):
        """获取当前学习率"""
        return self.current_lr