# experiment_config.py
import copy
from config import create_config_from_dict, _config

def get_experiment_config(experiment_type: str):
    """获取实验配置"""
    
    # 获取基础配置（不包含方法特定默认值）
    base_config = _config.get_base_config()
    
    experiments = []
    
    if experiment_type == 'ablation_study':
        """消融实验配置"""
        
        # 实验1：完整方法
        config1 = copy.deepcopy(base_config)
        config1.update({
            'method': 'our_method',
            'use_ala': True,
            'use_pseudo': True,
            'use_adaptive_clip': True,
        })
        args1 = create_config_from_dict(config1)
        experiments.append(('our_method_full', args1))
        
        # 实验2：无ALA
        config2 = copy.deepcopy(base_config)
        config2.update({
            'method': 'our_method',
            'use_ala': False,
            'use_pseudo': True,
            'use_adaptive_clip': True,
        })
        args2 = create_config_from_dict(config2)
        experiments.append(('our_method_no_ala', args2))
        
        # 实验3：无伪标签
        config3 = copy.deepcopy(base_config)
        config3.update({
            'method': 'our_method',
            'use_ala': True,
            'use_pseudo': False,
            'use_adaptive_clip': True,
        })
        args3 = create_config_from_dict(config3)
        experiments.append(('our_method_no_pseudo', args3))
        
        # 实验4：无自适应裁剪
        config4 = copy.deepcopy(base_config)
        config4.update({
            'method': 'our_method',
            'use_ala': True,
            'use_pseudo': True,
            'use_adaptive_clip': False,
        })
        args4 = create_config_from_dict(config4)
        experiments.append(('our_method_no_adaptive', args4))
        
        # 实验5：无DP
        config5 = copy.deepcopy(base_config)
        config5.update({
            'method': 'our_method_no_dp',
            'use_ala': True,
            'use_pseudo': True,
            'use_adaptive_clip': True,
        })
        args5 = create_config_from_dict(config5)
        experiments.append(('our_method_no_dp', args5))
        
        return experiments
    
    elif experiment_type == 'adaptive_clip_comparison':
        """自适应裁剪比较实验"""
        
        # 你的方法（有自适应裁剪）
        config1 = copy.deepcopy(base_config)
        config1.update({
            'method': 'our_method',
            'use_adaptive_clip': True,
        })
        args1 = create_config_from_dict(config1)
        experiments.append(('our_method_adaptive', args1))
        
        # DP-FedAvg（固定裁剪）
        config2 = copy.deepcopy(base_config)
        config2.update({
            'method': 'dp_fedavg',
        })
        args2 = create_config_from_dict(config2)
        experiments.append(('dp_fedavg', args2))
        
        # FedADDP
        config3 = copy.deepcopy(base_config)
        config3.update({
            'method': 'fedaddp',
        })
        args3 = create_config_from_dict(config3)
        experiments.append(('fedaddp', args3))
        
        return experiments
    
    elif experiment_type == 'aggregation_comparison':
        """聚合方法比较实验"""
        
        # 你的方法（无噪声）
        config1 = copy.deepcopy(base_config)
        config1.update({
            'method': 'our_method_no_dp',
        })
        args1 = create_config_from_dict(config1)
        experiments.append(('our_method_no_dp', args1))
        
        # Ditto
        config2 = copy.deepcopy(base_config)
        config2.update({
            'method': 'ditto',
        })
        args2 = create_config_from_dict(config2)
        experiments.append(('ditto', args2))
        
        # FedALA
        config3 = copy.deepcopy(base_config)
        config3.update({
            'method': 'fedala',
        })
        args3 = create_config_from_dict(config3)
        experiments.append(('fedala', args3))
        
        return experiments
    
    elif experiment_type == 'final_comparison':
        """最终对比实验"""
        
        methods = [
            ('fedavg', {}),
            ('dp_fedavg', {}),
            ('ditto', {}),
            ('fedala', {}),
            ('fedaddp', {}),
            ('our_method_no_dp', {}),
            ('our_method', {}),
        ]
        
        for method_name, method_params in methods:
            config = copy.deepcopy(base_config)
            config.update({
                'method': method_name,
            })
            config.update(method_params)
            
            args = create_config_from_dict(config)
            experiments.append((method_name, args))
        
        return experiments
    
    else:
        raise ValueError(f"未知的实验类型: {experiment_type}")