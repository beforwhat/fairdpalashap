# quick_fedavg.py
import sys

sys.argv = [
    'quick_fedavg.py',
    '--mode', 'quick_test',
    '--method', 'fedavg',
    '--dataset', 'MNIST',        # 自动使用 SimpleCNN
    '--num_clients', '10',
    '--global_epochs', '20',
    '--local_epochs', '5',
    '--batch_size', '32',
    '--lr', '0.01',
    '--iid',                     # IID 数据分布（不加则 Non-IID）
    '--visualize'               # 生成图表
]

from main import main
if __name__ == '__main__':
    main()