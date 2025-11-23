"""可视化数据集样本
展示三个数据集（CWRU、SEU、MFPT）每个类别的原始时间序列信号样本
在同一张图上展示所有数据集的所有类别
"""
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

from single_tccl import CWRUDataset, SEUDataset, MFPTDataset

# 设置绘图参数
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 300

def set_random_seed(seed=42):
    """设置全局随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 数据集配置
DATASET_CONFIG = {
    'CWRU': {
        'label_names': ['Normal', 'Ball', 'Inner', 'Outer'],
        'colors': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
        'path': 'E:/AI/CWRU-dataset-main/48k007',
        'dataset_class': CWRUDataset
    },
    'SEU': {
        'label_names': ['Normal', 'Ball', 'Inner', 'Outer'],
        'colors': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
        'path': 'E:/AI/Mechanical-datasets-master/dataset',
        'dataset_class': SEUDataset
    },
    'MFPT': {
        'label_names': ['Normal', 'Inner', 'Outer'],
        'colors': ['#2E86AB', '#F18F01', '#C73E1D'],
        'path': 'E:/AI/MFPT-Fault-Data-Sets-20200227T131140Z-001/MFPT/MFPT',
        'dataset_class': MFPTDataset
    }
}

def load_dataset_samples(dataset_name, n_samples_per_class=3):
    """
    加载数据集并从每个类别中抽取样本
    
    Args:
        dataset_name: 数据集名称 ('CWRU', 'SEU', 'MFPT')
        n_samples_per_class: 每个类别抽取的样本数量
    
    Returns:
        samples_dict: {类别索引: [(信号数据, 标签), ...]}
        config: 数据集配置信息
    """
    config = DATASET_CONFIG[dataset_name]
    print(f"\nLoading {dataset_name} dataset...")
    
    # 加载数据集
    dataset = config['dataset_class'](
        config['path'], 
        window_size=1024, 
        stride=512, 
        augment=False, 
        verbose=False
    )
    
    print(f"  ✓ Total samples: {len(dataset)}")
    
    # 获取所有标签
    all_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        all_labels.append(label)
    all_labels = np.array(all_labels)
    
    # 从每个类别中抽取样本
    samples_dict = {}
    unique_labels = np.unique(all_labels)
    
    for label in unique_labels:
        # 找到该类别的所有样本索引
        label_indices = np.where(all_labels == label)[0]
        
        # 随机选择n_samples_per_class个样本
        if len(label_indices) >= n_samples_per_class:
            selected_indices = np.random.choice(
                label_indices, 
                n_samples_per_class, 
                replace=False
            )
        else:
            selected_indices = label_indices
        
        # 提取样本
        samples = []
        for idx in selected_indices:
            signal, lbl = dataset[idx]
            # 将tensor转换为numpy数组
            if torch.is_tensor(signal):
                signal = signal.numpy()
            samples.append((signal, lbl))
        
        samples_dict[label] = samples
        print(f"  ✓ Class {label} ({config['label_names'][label]}): {len(samples)} samples")
    
    return samples_dict, config

def visualize_all_datasets(output_dir='./visualization_results', n_samples_per_class=3):
    """
    在同一张图上可视化三个数据集的所有类别样本
    
    Args:
        output_dir: 输出目录
        n_samples_per_class: 每个类别显示的样本数量
    """
    print("\n" + "="*70)
    print("可视化三个数据集的所有类别样本")
    print("="*70)
    
    # 加载所有数据集的样本
    all_datasets = {}
    for ds_name in ['CWRU', 'SEU', 'MFPT']:
        samples_dict, config = load_dataset_samples(ds_name, n_samples_per_class)
        all_datasets[ds_name] = {
            'samples': samples_dict,
            'config': config
        }
    
    # 计算布局
    # 每个数据集一行，每个类别一列，每个类别显示n_samples_per_class个样本（上下堆叠）
    n_datasets = len(all_datasets)
    max_classes = max([len(all_datasets[ds]['samples']) for ds in all_datasets])
    
    # 创建图形
    fig = plt.figure(figsize=(max_classes * 4, n_datasets * n_samples_per_class * 1.5))
    gs = GridSpec(
        n_datasets * n_samples_per_class, 
        max_classes, 
        figure=fig, 
        hspace=0.35, 
        wspace=0.3
    )
    
    # 对每个数据集绘图
    for ds_idx, ds_name in enumerate(['CWRU', 'SEU', 'MFPT']):
        data = all_datasets[ds_name]
        samples_dict = data['samples']
        config = data['config']
        label_names = config['label_names']
        colors = config['colors']
        
        # 定义显示顺序：Normal, Inner, Outer, Ball（Ball放最后）
        if ds_name in ['CWRU', 'SEU']:
            # 0=Normal, 1=Ball, 2=Inner, 3=Outer -> 显示顺序为 [0, 2, 3, 1]
            display_order = [0, 2, 3, 1]
        else:  # MFPT
            # 0=Normal, 1=Inner, 2=Outer -> 保持原顺序
            display_order = [0, 1, 2]
        
        # 对每个类别绘图（按照自定义顺序）
        for class_idx, label in enumerate(display_order):
            samples = samples_dict[label]
            # 对每个样本绘图
            for sample_idx, (signal, lbl) in enumerate(samples):
                row = ds_idx * n_samples_per_class + sample_idx
                col = class_idx
                
                ax = fig.add_subplot(gs[row, col])
                
                # 绘制信号
                # signal的形状可能是 (1, 1024) 或 (1024,)
                if signal.ndim > 1:
                    signal = signal.flatten()
                
                time_points = np.arange(len(signal))
                ax.plot(time_points, signal, color=colors[label], linewidth=0.8, alpha=0.8)
                
                # 设置标题和标签
                if sample_idx == 0:  # 只在第一行显示类别名称
                    ax.set_title(
                        f"{ds_name} - {label_names[label]}", 
                        fontsize=11, 
                        fontweight='bold',
                        color=colors[label]
                    )
                
                if col == 0:  # 只在第一列显示y轴标签
                    ax.set_ylabel('Amplitude', fontsize=8)
                
                if row == n_datasets * n_samples_per_class - 1:  # 只在最后一行显示x轴标签
                    ax.set_xlabel('Time Points', fontsize=8)
                
                # 美化图形
                ax.grid(alpha=0.3, linestyle='--')
                ax.tick_params(labelsize=7)
                
                # 设置y轴范围一致（便于比较）
                ax.set_ylim(signal.min() - 0.1 * np.abs(signal.min()), 
                           signal.max() + 0.1 * np.abs(signal.max()))
    
    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'dataset_samples_visualization')
    plt.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.close()
    
    print("\n" + "="*70)
    print("✓ 可视化完成！")
    print("="*70)

def visualize_compact_version(output_dir='./visualization_results', n_samples_per_class=1):
    """
    紧凑版本：在同一张图上展示所有数据集的所有类别（每个类别只显示1个代表性样本）
    布局：3行（数据集）x 最多4列（类别）
    
    Args:
        output_dir: 输出目录
        n_samples_per_class: 每个类别显示的样本数量（默认1）
    """
    print("\n" + "="*70)
    print("可视化三个数据集的所有类别样本（紧凑版）")
    print("="*70)
    
    # 加载所有数据集的样本
    all_datasets = {}
    for ds_name in ['CWRU', 'SEU', 'MFPT']:
        samples_dict, config = load_dataset_samples(ds_name, n_samples_per_class)
        all_datasets[ds_name] = {
            'samples': samples_dict,
            'config': config
        }
    
    # 计算布局：3行 x 4列
    n_datasets = 3
    n_cols = 4  # CWRU和SEU有4个类别，MFPT有3个类别
    
    # 创建图形
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(n_datasets, n_cols, figure=fig, hspace=0.3, wspace=0.25)
    
    # 对每个数据集绘图
    for ds_idx, ds_name in enumerate(['CWRU', 'SEU', 'MFPT']):
        data = all_datasets[ds_name]
        samples_dict = data['samples']
        config = data['config']
        label_names = config['label_names']
        colors = config['colors']
        n_classes = len(samples_dict)
        
        # 定义显示顺序：Normal, Inner, Outer, Ball（Ball放最后）
        if ds_name in ['CWRU', 'SEU']:
            # 0=Normal, 1=Ball, 2=Inner, 3=Outer -> 显示顺序为 [0, 2, 3, 1]
            display_order = [0, 2, 3, 1]
        else:  # MFPT
            # 0=Normal, 1=Inner, 2=Outer -> 保持原顺序
            display_order = [0, 1, 2]
        
        # 对每个类别绘图（按照自定义顺序）
        for class_idx, label in enumerate(display_order):
            ax = fig.add_subplot(gs[ds_idx, class_idx])
            
            # 从样本字典中获取当前类别的样本
            samples = samples_dict[label]
            
            # 取第一个样本
            signal, lbl = samples[0]
            
            # 绘制信号
            if signal.ndim > 1:
                signal = signal.flatten()
            
            time_points = np.arange(len(signal))
            ax.plot(time_points, signal, color=colors[label], linewidth=1.0, alpha=0.9)
            
            # 设置标题
            if ds_idx == 0:  # 只在第一行显示类别名称
                ax.set_title(
                    f"{label_names[label]}", 
                    fontsize=11, 
                    fontweight='bold',
                    color=colors[label]
                )
            
            # 在每行第一个子图的左侧添加数据集名称
            if class_idx == 0:
                ax.text(
                    -0.25, 0.5, ds_name, 
                    transform=ax.transAxes,
                    fontsize=13,
                    fontweight='bold',
                    rotation=90,
                    verticalalignment='center',
                    horizontalalignment='center'
                )
            
            # 只在最左列显示y轴标签
            if class_idx == 0:
                ax.set_ylabel('Amplitude', fontsize=9)
            
            # 只在最后一行显示x轴标签
            if ds_idx == n_datasets - 1:
                ax.set_xlabel('Time Points', fontsize=9)
            
            # 美化图形
            ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
            ax.tick_params(labelsize=8)
            
            # 设置y轴范围
            margin = 0.15 * (signal.max() - signal.min())
            ax.set_ylim(signal.min() - margin, signal.max() + margin)
        
        # 如果MFPT数据集（只有3个类别），隐藏第4列
        if ds_name == 'MFPT' and n_classes < n_cols:
            ax = fig.add_subplot(gs[ds_idx, n_cols-1])
            ax.axis('off')
    
    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'dataset_samples_compact')
    plt.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.close()
    
    print("\n" + "="*70)
    print("✓ 可视化完成！")
    print("="*70)

def main():
    # 设置随机种子
    set_random_seed(42)
    
    output_dir = './visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成两个版本的可视化
    print("\n生成紧凑版本（推荐）...")
    visualize_compact_version(output_dir, n_samples_per_class=1)
    
    print("\n生成完整版本（每个类别3个样本）...")
    visualize_all_datasets(output_dir, n_samples_per_class=3)

if __name__ == '__main__':
    main()
