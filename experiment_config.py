# experiment_config.py
from config import get_config

def get_experiment_config(experiment_type: str):
    """获取实验配置"""
    
    # 基础配置
    base_args = get_config()
    
    if experiment_type == 'ablation_study':
        """消融实验配置"""
        experiments = []
        
        # 实验1：完整方法
        args1 = base_args
        args1.method = 'our_method'
        args1.use_ala = True
        args1.use_pseudo = True
        args1.use_adaptive_clip = True
        experiments.append(('our_method_full', args1))
        
        # 实验2：无ALA
        args2 = base_args
        args2.method = 'our_method'
        args2.use_ala = False
        args2.use_pseudo = True
        args2.use_adaptive_clip = True
        experiments.append(('our_method_no_ala', args2))
        
        # 实验3：无伪标签
        args3 = base_args
        args3.method = 'our_method'
        args3.use_ala = True
        args3.use_pseudo = False
        args3.use_adaptive_clip = True
        experiments.append(('our_method_no_pseudo', args3))
        
        # 实验4：无自适应裁剪
        args4 = base_args
        args4.method = 'our_method'
        args4.use_ala = True
        args4.use_pseudo = True
        args4.use_adaptive_clip = False
        experiments.append(('our_method_no_adaptive', args4))
        
        # 实验5：无DP
        args5 = base_args
        args5.method = 'our_method_no_dp'
        args5.use_ala = True
        args5.use_pseudo = True
        args5.use_adaptive_clip = True
        experiments.append(('our_method_no_dp', args5))
        
        return experiments
    
    elif experiment_type == 'adaptive_clip_comparison':
        """自适应裁剪比较实验"""
        experiments = []
        
        # 你的方法（有自适应裁剪）
        args1 = base_args
        args1.method = 'our_method'
        args1.use_adaptive_clip = True
        experiments.append(('our_method_adaptive', args1))
        
        # DP-FedAvg（固定裁剪）
        args2 = base_args
        args2.method = 'dp_fedavg'
        experiments.append(('dp_fedavg', args2))
        
        # FedADDP
        args3 = base_args
        args3.method = 'fedaddp'
        experiments.append(('fedaddp', args3))
        
        return experiments
    
    elif experiment_type == 'aggregation_comparison':
        """聚合方法比较实验"""
        experiments = []
        
        # 你的方法（无噪声）
        args1 = base_args
        args1.method = 'our_method_no_dp'
        experiments.append(('our_method_no_dp', args1))
        
        # Ditto
        args2 = base_args
        args2.method = 'ditto'
        experiments.append(('ditto', args2))
        
        # FedALA
        args3 = base_args
        args3.method = 'fedala'
        experiments.append(('fedala', args3))
        
        return experiments
    
    elif experiment_type == 'final_comparison':
        """最终对比实验"""
        experiments = []
        
        methods = [
            'fedavg',
            'dp_fedavg', 
            'ditto',
            'fedala',
            'fedaddp',
            'our_method_no_dp',
            'our_method'
        ]
        
        for method in methods:
            args = base_args
            args.method = method
            experiments.append((method, args))
        
        return experiments
    
    else:
        raise ValueError(f"未知的实验类型: {experiment_type}")