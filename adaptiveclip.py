class DualEMAClipper:
    """
    双EMA自适应裁剪阈值管理器
    - fast_ema: 快速响应，跟踪近期梯度尺度
    - slow_ema: 慢速响应，反映长期趋势
    - 当两者偏差过大时，使用 slow_ema 作为阈值，否则使用 fast_ema
    """
    def __init__(self, alpha_fast=0.4, alpha_slow=0.8, init_val=1.0, deviation_thresh=0.7):
        """
        Args:
            alpha_fast: 快速EMA平滑系数 (0~1)，越小响应越快
            alpha_slow: 慢速EMA平滑系数 (0~1)，越大越平滑
            init_val: 初始阈值
            deviation_thresh: 相对偏差阈值，当 |fast - slow| / slow > 此值时，认为异常
        """
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.fast_ema = init_val
        self.slow_ema = init_val
        self.deviation_thresh = deviation_thresh
        self.clip_val = init_val   # 当前应使用的裁剪阈值
        self.clip_Val_history=[]
    def update(self, grad_norm):
        """
        用当前批次的梯度范数更新双EMA，并返回应使用的裁剪阈值
        Args:
            grad_norm: 当前批次的原始梯度范数（可能带噪）
        Returns:
            clip_val: 当前应使用的裁剪阈值
        """
        # 更新EMA
        self.fast_ema = self.alpha_fast * self.fast_ema + (1 - self.alpha_fast) * grad_norm
        self.slow_ema = self.alpha_slow * self.slow_ema + (1 - self.alpha_slow) * grad_norm
        self.clip_Val_history.append(self.clip_val)
        # 判断是否异常
        if self.slow_ema > 1e-8:   # 避免除零
            relative_dev = abs(self.fast_ema - self.slow_ema) / self.slow_ema
        else:
            relative_dev = 0.0

        if relative_dev > self.deviation_thresh:
            # 异常波动，使用慢速EMA（稳定值）
            self.clip_val = self.slow_ema
        else:
            # 正常波动，使用快速EMA（适应变化）
            self.clip_val = self.fast_ema
       
        # 限制阈值范围，防止极端值
        self.clip_val = max(self.clip_val, 1e-6)
        # 可选上限（根据数据集调整）
        self.clip_val = min(self.clip_val, self.clip_Val_history[-1]*1.5)
       
        return self.clip_val