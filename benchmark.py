"""
综合基准测试脚本

对比14个方法：
- 传统方法（3个）：Raw+UMAP, Handcrafted Features, PCA+K-Means
- 深度聚类（3个）：DEC, JULE, SCAN
- 对比学习-经典（3个）：SimCLR, MoCo, BYOL
- 对比学习-SOTA（4个）：SimSiam, TS2Vec, VICReg, TimesNet
- 本文方法（1个）：TCCL

使用方法：
    # 单个数据集
    python benchmark.py --dataset CWRU --data_root E:/AI/CWRU-dataset-main/48k007 --epochs 50
    python benchmark.py --dataset SEU --data_root E:/AI/Mechanical-datasets-master/dataset --epochs 50
    python benchmark.py --dataset MFPT --data_root E:/AI/MFPT-Fault-Data-Sets-20200227T131140Z-001/MFPT/MFPT --epochs 50
    
    # 批量运行所有数据集（不显示图表，自动保存）
    python benchmark.py --all --epochs 50
    
    # 单数据集但不显示图表
    python benchmark.py --dataset CWRU --data_root E:/AI/CWRU-dataset-main/48k007 --epochs 50 --no_show
"""

import argparse
import copy
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import umap
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score

# 导入数据集
from datasets import CWRUDataset, SEUDataset, MFPTDataset

# 数据增强器（本地实现，与数据集接口匹配）
class SignalAugmentation:
    def __init__(self, noise_level: float = 0.05, scale_range=(0.9, 1.1)):
        self.noise_level = noise_level
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(1, device=x.device).uniform_(self.scale_range[0], self.scale_range[1]).item()
        noise = torch.randn_like(x) * self.noise_level
        return x * scale + noise

# 导入基线模型（新的模块化结构）
from models.baselines import (
    # 基础组件
    FeatureExtractor,
    # 传统方法
    RawUMAPModel,
    HandcraftedFeaturesModel,
    PCAKMeansModel,
    # 深度聚类
    DECModel,
    JULEModel,
    SCANModel,
    # 对比学习-经典
    SimCLRModel,
    MoCoModel,
    BYOLModel,
    # 对比学习-SOTA
    SimSiamModel,
    TS2VecModel,
    VICRegModel,
    TimesNetModel,
)

# 导入工具
from utils.benchmark_utils import (
    train_model,
    extract_features_from_loader
)


class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        # 统一随机种子
        try:
            set_random_seed(config.get('seed', 42))
        except Exception:
            pass
        
        self.models = {}
        self.results = defaultdict(dict)
        self.training_history = defaultdict(list)
        
        print("=" * 80)
        print("🚀 COMPREHENSIVE BENCHMARK FRAMEWORK")
        print("=" * 80)
        print(f"Dataset: {config['dataset']}")
        print(f"Device: {self.device}")
        print(f"Epochs: {config['epochs']}")
        print(f"Batch Size: {config['batch_size']}")
        print("=" * 80)
    
    def prepare_data(self):
        """准备数据集"""
        print("\n📂 Loading dataset...")
        
        # 数据增强
        augmentor = SignalAugmentation(
            noise_level=self.config.get('noise_level', 0.05),
            scale_range=self.config.get('scale_range', (0.9, 1.1))
        )
        
        # 数据集类映射
        dataset_classes = {
            'CWRU': CWRUDataset,
            'SEU': SEUDataset,
            'MFPT': MFPTDataset
        }
        
        if self.config['dataset'] not in dataset_classes:
            raise ValueError(f"Unknown dataset: {self.config['dataset']}")
        
        DatasetClass = dataset_classes[self.config['dataset']]
        
        # 训练集（带增强）
        train_dataset = DatasetClass(
            root_dir=self.config['data_root'],
            window_size=self.config['window_size'],
            step_size=self.config['step_size'],
            mode='full',
            augmentor=augmentor
        )
        
        # 评估集（不带增强）
        eval_dataset = DatasetClass(
            root_dir=self.config['data_root'],
            window_size=self.config['window_size'],
            step_size=self.config['step_size'],
            mode='full',
            augmentor=None
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 0),
            drop_last=True
        )
        
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=self.config.get('num_workers', 0)
        )
        
        self.class_names = {v: k for k, v in train_dataset.LABEL_MAP.items()}
        self.n_classes = len(self.class_names)
        
        print(f"✓ Train samples: {len(train_dataset)}")
        print(f"✓ Eval samples: {len(eval_dataset)}")
        print(f"✓ Classes: {list(self.class_names.values())}")
        
        # 初始化每模型温度覆盖（如未提供则使用合理默认）
        # 允许用户通过 config['temperature_overrides'] 覆盖
        default_overrides = {
            'TCCL': 0.01,                # 模板相关响应，较低 τ
            'SimCLR': 0.07,              # 余弦相似度范式常用 τ
            'MoCo': 0.07,                # 余弦相似度范式常用 τ
            'SimSiam': 0.05,
            'TS2Vec': 0.07,
            'TimesNet': 0.07,
        }
        user_overrides = self.config.get('temperature_overrides', {}) or {}
        self.temperature_overrides = {**default_overrides, **user_overrides}
    
    def create_models(self):
        """创建所有模型"""
        print("\n🔨 Creating models...")
        
        # 基础特征提取器
        feature_extractor_base = FeatureExtractor(in_channels=1, out_channels=64)
        
        # 定义所有模型
        models_config = {
            # === 传统方法 ===
            'Raw+UMAP': RawUMAPModel(),
            'Handcrafted Features': HandcraftedFeaturesModel(),
            'PCA+K-Means': PCAKMeansModel(n_components=64),
            
            # === 深度聚类 ===
            'DEC': DECModel(
                feature_extractor=copy.deepcopy(feature_extractor_base),
                n_clusters=self.n_classes
            ),
            'JULE': JULEModel(
                feature_extractor=copy.deepcopy(feature_extractor_base),
                n_clusters=self.n_classes
            ),
            'SCAN': SCANModel(
                feature_extractor=copy.deepcopy(feature_extractor_base),
                n_clusters=self.n_classes
            ),
            
            # === 对比学习（经典 2020） ===
            'SimCLR': SimCLRModel(
                feature_extractor=copy.deepcopy(feature_extractor_base),
                projection_dim=128,
                hidden_dim=2048
            ),
            'MoCo': MoCoModel(
                feature_extractor=copy.deepcopy(feature_extractor_base),
                projection_dim=128,
                hidden_dim=2048,
                queue_size=min(65536, len(self.train_loader.dataset)),
                momentum=0.999
            ),
            # 'BYOL': BYOLModel(
            #     feature_extractor=copy.deepcopy(feature_extractor_base),
            #     projection_dim=128,
            #     hidden_dim=2048
            # ),
            
            # === 对比学习（SOTA 2021-2023） ===
            'SimSiam': SimSiamModel(
                feature_extractor=copy.deepcopy(feature_extractor_base),
                projection_dim=128,
                hidden_dim=2048
            ),
            'TS2Vec': TS2VecModel(
                feature_extractor=copy.deepcopy(feature_extractor_base),
                projection_dim=128
            ),
            # 'VICReg': VICRegModel(
            #     feature_extractor=copy.deepcopy(feature_extractor_base),
            #     projection_dim=128,
            #     hidden_dim=2048
            # ),
            'TimesNet': TimesNetModel(
                feature_extractor=copy.deepcopy(feature_extractor_base),
                projection_dim=128
            ),
            
            # === 本文方法（对比学习风格实现）===
            'TCCL': __import__('models.baselines', fromlist=['TCCLModel', 'FeatureExtractor']).contrastive.TCCLModel(
                feature_extractor=copy.deepcopy(feature_extractor_base),
                kernel_width=3,
                temperature=self.config['temperature']
            ),
        }
        
        # 注册模型
        self.models = models_config
        
        # 打印模型信息
        print("\n📋 Registered Models:")
        print("-" * 80)
        
        categories = {
            'Traditional': ['Raw+UMAP', 'Handcrafted Features', 'PCA+K-Means'],
            'Deep Clustering': ['DEC', 'JULE', 'SCAN'],
            'Contrastive (2020)': ['SimCLR', 'MoCo', 'BYOL'],
            'Contrastive (SOTA)': ['SimSiam', 'TS2Vec', 'VICReg', 'TimesNet'],
            'TCCL (Ours)': ['TCCL']
        }
        
        for category, methods in categories.items():
            print(f"\n{category}:")
            for method in methods:
                if method in self.models:
                    model = self.models[method]
                    num_params = sum(p.numel() for p in model.parameters())
                    print(f"  ✓ {method:<25s} {num_params:>12,} params")
    
    def train_all_models(self):
        """训练所有需要训练的模型"""
        print("\n" + "=" * 80)
        print("🎓 TRAINING PHASE")
        print("=" * 80)
        
        for name, model in self.models.items():
            if not model.needs_training:
                print(f"\n⏭️  Skipping {name} (no training required)")
                continue
            
            print(f"\n{'='*60}")
            print(f"🚀 Training {name} [{model.model_type}]")
            print(f"{'='*60}")
            
            # 每模型温度覆盖
            temp = self.temperature_overrides.get(name, self.config['temperature'])

            history = train_model(
                model=model,
                train_loader=self.train_loader,
                device=self.device,
                epochs=self.config['epochs'],
                learning_rate=self.config['learning_rate'],
                temperature=temp,
                verbose=True
            )
            
            self.training_history[name] = history
            print(f"✓ {name} training completed")
    
    def _prefit_pca_on_eval(self):
        """在评估集上预拟合 PCA，以避免按 batch 拟合导致的偏置。"""
        if 'PCA+K-Means' not in self.models:
            return
        model = self.models['PCA+K-Means']
        if not hasattr(model, 'fit_pca'):
            return
        print("\n🧮 Prefitting PCA on full eval set...")
        all_signals = []
        with torch.no_grad():
            for x, _, _ in self.eval_loader:
                # x: [B, 1, L] -> [B, L]
                x_flat = x.view(x.size(0), -1).cpu().numpy()
                all_signals.append(x_flat)
        if all_signals:
            all_signals = np.concatenate(all_signals, axis=0)
            try:
                model.fit_pca(all_signals, verbose=True)
                print("  ✓ PCA prefitted on eval set")
            except Exception as e:
                print(f"  ❌ PCA prefit failed: {e}")

    @staticmethod
    def _zscore_features(features: np.ndarray) -> np.ndarray:
        """对特征进行 z-score 标准化（按特征维度），避免尺度影响聚类。"""
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        return (features - mean) / std

    def evaluate_all_models(self):
        """评估所有模型"""
        print("\n" + "=" * 80)
        print("📊 EVALUATION PHASE")
        print("=" * 80)
        # 注：按用户要求，恢复为不在评估前进行全量 PCA 预拟合
        # PCA 将在第一次调用 PCAKMeansModel.extract_features() 时按旧逻辑自动拟合
        
        for name, model in self.models.items():
            print(f"\n📊 Evaluating {name}...")
            
            try:
                # 提取特征
                print("  - Extracting features...")
                features, labels = extract_features_from_loader(
                    model, self.eval_loader, self.device
                )
                # 统一特征标准化（按模型）
                features = self._zscore_features(features)
                
                # 使用 single_tccl.py 风格：不做L2；UMAP->2D；DBSCAN；噪声KNN回填
                print("  - Computing metrics with UMAP+DBSCAN (single-style, no L2)...")
                eval_res = evaluate_clustering_single_style(features, labels)
                metrics = {
                    'accuracy_Nto1': eval_res['acc'],
                    'ari': eval_res['ari'],
                    'nmi': eval_res['nmi'],
                    'silhouette': eval_res['sil'],
                    'n_clusters': eval_res['n_clusters'],
                    'separation_ratio': float('nan'),
                    'davies_bouldin': float('nan'),
                    'calinski_harabasz': float('nan'),
                }
                pred_labels = eval_res['pred_labels']
                features_2d = eval_res['features_reduced']
                
                # 保存结果
                self.results[name] = {
                    'features': features,
                    'features_clean': features,  # 与single对齐，不做L2
                    'metrics': metrics,
                    'pred_labels': pred_labels,
                    'features_2d': features_2d,
                    'labels': labels
                }
                
                print(f"  ✓ {name}: Acc={metrics['accuracy_Nto1']:.4f}, "
                      f"ARI={metrics['ari']:.4f}, NMI={metrics['nmi']:.4f}, "
                      f"Sil={metrics['silhouette']:.4f}")
                
            except Exception as e:
                print(f"  ❌ Failed to evaluate {name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def print_comparison_table(self):
        """打印对比表格"""
        print("\n" + "=" * 140)
        print("📊 COMPREHENSIVE PERFORMANCE COMPARISON")
        print("=" * 140)
        
        categories = {
            'Traditional Methods': ['Raw+UMAP', 'Handcrafted Features', 'PCA+K-Means'],
            'Deep Clustering': ['DEC', 'JULE', 'SCAN'],
            'Contrastive Learning (2020)': ['SimCLR', 'MoCo', 'BYOL'],
            'Contrastive Learning (SOTA 2021-2023)': ['SimSiam', 'TS2Vec', 'VICReg', 'TimesNet'],
            'TCCL (Ours)': ['TCCL']
        }
        
        for category, methods in categories.items():
            print(f"\n{category}:")
            print("-" * 140)
            print(f"{'Method':<25} {'Acc↑':>8} {'ARI↑':>8} {'NMI↑':>8} {'Sil↑':>8} "
                  f"{'SepR↑':>8} {'DB↓':>8} {'CH↑':>9}")
            print("-" * 140)
            
            for method in methods:
                if method in self.results:
                    m = self.results[method]['metrics']
                    print(f"{method:<25} {m['accuracy_Nto1']:>8.4f} {m['ari']:>8.4f} "
                          f"{m['nmi']:>8.4f} {m['silhouette']:>8.4f} "
                          f"{m['separation_ratio']:>8.4f} {m['davies_bouldin']:>8.4f} "
                          f"{m['calinski_harabasz']:>9.1f}")
        
        print("=" * 140)
    
    def visualize_results(self):
        """可视化结果"""
        print("\n🎨 Generating visualizations...")
        
        # 1. 训练曲线
        if self.training_history:
            self._plot_training_curves()
        
        # 2. 性能对比柱状图
        self._plot_performance_bars()
        
        # 3. 雷达图
        self._plot_radar_chart()
        
        # 4. 聚类可视化
        if self.config.get('visualize_clusters', True):
            self._plot_clustering_results()
    
    def _plot_training_curves(self):
        """绘制训练曲线 - 两种方式：1) 所有损失在一张图上 2) 3*3子图布局"""
        
        # === 方式1: 所有损失绘制在一张图上 ===
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        # 定义颜色和线型
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
                 '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b',
                 '#8e44ad', '#27ae60', '#d35400', '#7f8c8d']
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
        
        for idx, (name, history) in enumerate(self.training_history.items()):
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            ax1.plot(history, linewidth=2.5, color=color, linestyle=linestyle, 
                    label=name, marker='o', markersize=3, markevery=max(1, len(history)//10))
        
        ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
        ax1.set_title('Training Curves - All Methods', fontsize=16, fontweight='bold')
        ax1.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        # plt.savefig('benchmark_training_curves_combined.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: benchmark_training_curves_combined.png")
        if not self.config.get('no_show', False):
            plt.show()
        plt.close(fig1)
        
        # === 方式2: 3*3子图布局 ===
        fig2, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        # 获取所有方法名称
        method_names = list(self.training_history.keys())
        
        # 为每个子图绘制对应的损失曲线
        for idx in range(9):  # 3*3 = 9个子图
            ax = axes[idx]
            
            if idx < len(method_names):
                name = method_names[idx]
                history = self.training_history[name]
                ax.plot(history, linewidth=2.5, color='#2ecc71', marker='o', 
                       markersize=4, markevery=max(1, len(history)//10))
                ax.set_xlabel(' ', fontsize=11)
                ax.set_ylabel(' ', fontsize=11)
                ax.set_title(f'{name}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
            else:
                # 隐藏多余的子图
                ax.axis('off')
        
        # plt.suptitle('Training Curves - 3×3 Subplot Layout', 
        #             fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        # plt.savefig('benchmark_training_curves_subplots.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: benchmark_training_curves_subplots.png")
        if not self.config.get('no_show', False):
            plt.show()
        plt.close(fig2)
    
    def _plot_performance_bars(self):
        """绘制性能对比柱状图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        metric_names = {
            'accuracy_Nto1': 'Clustering Accuracy (N-to-1)',
            'ari': 'Adjusted Rand Index',
            'nmi': 'Normalized Mutual Information',
            'silhouette': 'Silhouette Coefficient'
        }
        
        metric_keys = ['accuracy_Nto1', 'ari', 'nmi', 'silhouette']
        
        for idx, metric_key in enumerate(metric_keys):
            ax = axes[idx]
            
            methods = []
            values = []
            colors = []
            
            for method, result in self.results.items():
                methods.append(method)
                values.append(result['metrics'][metric_key])
                
                # 根据方法类别着色
                if method in ['Raw+UMAP', 'Handcrafted Features', 'PCA+K-Means']:
                    colors.append('#95a5a6')
                elif method in ['DEC', 'JULE', 'SCAN']:
                    colors.append('#3498db')
                elif method in ['SimCLR', 'MoCo', 'BYOL']:
                    colors.append('#9b59b6')
                elif method in ['SimSiam', 'TS2Vec', 'VICReg', 'TimesNet']:
                    colors.append('#2ecc71')
                else:  # TCCL
                    colors.append('#e74c3c')
            
            bars = ax.bar(range(len(methods)), values, color=colors,
                         edgecolor='black', linewidth=1.5)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
            
            ax.set_ylabel(metric_names[metric_key], fontsize=11, fontweight='bold')
            ax.set_title(metric_names[metric_key], fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(values) * 1.15)
        
        plt.suptitle('Performance Comparison Across All Methods',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        # plt.savefig('benchmark_performance_bars.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: benchmark_performance_bars.png")
        if not self.config.get('no_show', False):
            plt.show()
        plt.close(fig)
    
    def _plot_radar_chart(self):
        """绘制雷达图"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        key_methods = ['Handcrafted Features', 'DEC', 'SimCLR', 'TS2Vec', 'TimesNet', 'TCCL']
        metrics_to_plot = ['Acc', 'ARI', 'NMI', 'Sil']
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
        angles += angles[:1]
        
        colors = ['#95a5a6', '#3498db', '#9b59b6', '#16a085', '#2ecc71', '#e74c3c']
        
        for idx, method in enumerate(key_methods):
            if method in self.results:
                metrics = self.results[method]['metrics']
                values = [
                    metrics['accuracy_Nto1'],
                    metrics['ari'],
                    metrics['nmi'],
                    metrics['silhouette']
                ]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
                ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_to_plot, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        ax.set_title('Performance Comparison (Radar Chart)',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        # plt.savefig('benchmark_radar_chart.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: benchmark_radar_chart.png")
        if not self.config.get('no_show', False):
            plt.show()
        plt.close(fig)
    
    def _plot_clustering_results(self):
        """绘制聚类结果（真实标签 vs 聚类结果）"""
        n_methods = len(self.results)
        n_cols = min(3, n_methods)  # 减少列数，因为每个方法要显示2个子图
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        # 每个方法显示2个子图（真实标签 + 聚类结果），所以实际列数要 x2
        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(6*n_cols, 4*n_rows))
        
        # 处理axes的维度
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (method, result) in enumerate(self.results.items()):
            if result['features_2d'] is None:
                continue
            
            row = idx // n_cols
            col_base = (idx % n_cols) * 2
            
            ax_true = axes[row, col_base]      # 左侧：真实标签
            ax_pred = axes[row, col_base + 1]  # 右侧：聚类结果
            
            features_2d = result['features_2d']
            true_labels = result['labels']
            pred_labels = result['pred_labels']
            metrics = result['metrics']
            
            # === 左侧：真实标签分布 ===
            unique_true = np.unique(true_labels)
            cmap_true = plt.cm.get_cmap('tab10')
            
            for i, label in enumerate(unique_true):
                mask = (true_labels == label)
                class_name = self.class_names.get(label, f'Class {label}')
                color = cmap_true(i / max(len(unique_true), 10))  # 归一化到[0,1]
                ax_true.scatter(features_2d[mask, 0], features_2d[mask, 1],
                              c=[color], label=class_name,
                              s=20, alpha=0.8, edgecolors='none', linewidth=0)
            
            ax_true.set_title(f'{method} (Ground Truth)', 
                            fontsize=10, fontweight='bold')
            ax_true.set_xlabel(' ', fontsize=9)
            ax_true.set_ylabel(' ', fontsize=9)
            # ax_true.legend(loc='best', fontsize=7, framealpha=0.8)
            ax_true.grid(True, alpha=0.2)
            
            # === 右侧：聚类结果 ===
            unique_clusters = np.unique(pred_labels)
            cmap_cluster = plt.cm.get_cmap('Set3')
            
            for i, cluster in enumerate(unique_clusters):
                mask = (pred_labels == cluster)
                color = cmap_cluster(i / max(len(unique_clusters), 12))  # 归一化到[0,1]
                ax_pred.scatter(features_2d[mask, 0], features_2d[mask, 1],
                              c=[color], label=f'C{cluster}',
                              s=20, alpha=0.8, edgecolors='none', linewidth=0)
            
            acc = metrics['accuracy_Nto1']
            ari = metrics['ari']
            nmi = metrics['nmi']
            sil = metrics['silhouette']
            # ax_pred.set_title(f'{method}\n(Clustering K={metrics["n_clusters"]})\n'
            #                 f'Acc: {acc:.3f} | ARI: {ari:.3f} | NMI: {nmi:.3f}',
            #                 fontsize=10, fontweight='bold')
            
            ax_pred.set_title(f'{method}',
                            fontsize=10, fontweight='bold')
            ax_pred.set_xlabel(' ', fontsize=9)
            ax_pred.set_ylabel(' ', fontsize=9)
            # ax_pred.legend(loc='best', fontsize=7, framealpha=0.8)
            ax_pred.grid(True, alpha=0.2)
        
        # 隐藏多余的子图
        for idx in range(n_methods, n_rows * n_cols):
            row = idx // n_cols
            col_base = (idx % n_cols) * 2
            axes[row, col_base].axis('off')
            axes[row, col_base + 1].axis('off')
        
        # plt.suptitle('Clustering Results: Ground Truth vs Predictions',
        #             fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存 PDF 版本
        pdf_filename = f'benchmark_clustering_{self.config["dataset"]}.pdf'
        plt.savefig(pdf_filename, bbox_inches='tight')
        print(f"  ✓ Saved: {pdf_filename}")
        
        if not self.config.get('no_show', False):
            plt.show()
        plt.close(fig)
    
    def save_results(self):
        """保存结果到文件"""
        output_file = f"benchmark_results_{self.config['dataset']}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 140 + "\n")
            f.write(f"COMPREHENSIVE BENCHMARK RESULTS - {self.config['dataset']} Dataset\n")
            f.write("=" * 140 + "\n\n")
            
            # 配置信息
            f.write("Configuration:\n")
            f.write("-" * 140 + "\n")
            for key, value in self.config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 详细结果
            categories = {
                'Traditional Methods': ['Raw+UMAP', 'Handcrafted Features', 'PCA+K-Means'],
                'Deep Clustering': ['DEC', 'JULE', 'SCAN'],
                'Contrastive Learning (2020)': ['SimCLR', 'MoCo', 'BYOL'],
                'Contrastive Learning (SOTA 2021-2023)': ['SimSiam', 'TS2Vec', 'VICReg', 'TimesNet'],
                'TCCL (Ours)': ['TCCL']
            }
            
            for category, methods in categories.items():
                f.write(f"{category}:\n")
                f.write("-" * 140 + "\n")
                f.write(f"{'Method':<25} {'Acc↑':>8} {'ARI↑':>8} {'NMI↑':>8} {'Sil↑':>8} "
                       f"{'SepR↑':>8} {'DB↓':>8} {'CH↑':>9}\n")
                f.write("-" * 140 + "\n")
                
                for method in methods:
                    if method in self.results:
                        m = self.results[method]['metrics']
                        f.write(f"{method:<25} {m['accuracy_Nto1']:>8.4f} {m['ari']:>8.4f} "
                               f"{m['nmi']:>8.4f} {m['silhouette']:>8.4f} "
                               f"{m['separation_ratio']:>8.4f} {m['davies_bouldin']:>8.4f} "
                               f"{m['calinski_harabasz']:>9.1f}\n")
                f.write("\n")
            
            f.write("=" * 140 + "\n")
        
        print(f"\n✓ Results saved to: {output_file}")
    
    def run(self):
        """运行完整的基准测试流程"""
        try:
            # 步骤1: 准备数据
            self.prepare_data()
            
            # 步骤2: 创建模型
            self.create_models()
            
            # 步骤3: 训练模型
            self.train_all_models()
            
            # 步骤4: 评估模型
            self.evaluate_all_models()
            
            # 步骤5: 打印对比表格
            self.print_comparison_table()
            
            # 步骤6: 可视化结果
            self.visualize_results()
            
            # 步骤7: 保存结果
            self.save_results()
            
            print("\n" + "=" * 80)
            print("✅ BENCHMARK COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"📊 Total methods evaluated: {len(self.results)}")
            
            if 'TCCL' in self.results:
                tccl_metrics = self.results['TCCL']['metrics']
                print(f"\n🏆 TCCL Performance:")
                print(f"   Accuracy: {tccl_metrics['accuracy_Nto1']:.4f}")
                print(f"   ARI: {tccl_metrics['ari']:.4f}")
                print(f"   NMI: {tccl_metrics['nmi']:.4f}")
                print(f"   Silhouette: {tccl_metrics['silhouette']:.4f}")
            
            print("=" * 80)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Benchmark interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Error during benchmark: {e}")
            import traceback
            traceback.print_exc()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Benchmark for TCCL',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, required=False,
                       choices=['CWRU', 'SEU', 'MFPT'],
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, required=False,
                       help='Root directory of dataset')
    parser.add_argument('--all', action='store_true',
                       help='Run benchmark on all datasets (CWRU, SEU, MFPT) automatically')
    parser.add_argument('--cwru_root', type=str, default='E:/AI/CWRU-dataset-main/48k007',
                       help='CWRU dataset root (used with --all)')
    parser.add_argument('--seu_root', type=str, default='E:/AI/Mechanical-datasets-master/dataset',
                       help='SEU dataset root (used with --all)')
    parser.add_argument('--mfpt_root', type=str, default='E:/AI/MFPT-Fault-Data-Sets-20200227T131140Z-001/MFPT/MFPT',
                       help='MFPT dataset root (used with --all)')
    parser.add_argument('--window_size', type=int, default=1024,
                       help='Window size')
    parser.add_argument('--step_size', type=int, default=512,
                       help='Step size')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.01,
                       help='Temperature for contrastive loss')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # 其他参数
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--no_umap', action='store_true',
                       help='Disable UMAP for clustering')
    parser.add_argument('--no_visualize_clusters', action='store_true',
                       help='Disable cluster visualization')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not show plots (only save to files)')
    
    args = parser.parse_args()
    
    # 验证参数
    if not args.all and (not args.dataset or not args.data_root):
        parser.error('--dataset and --data_root are required unless --all is specified')
    
    # 转换为配置字典
    config = {
        'dataset': args.dataset,
        'data_root': args.data_root,
        'run_all': args.all,
        'no_show': args.no_show or args.all,  # --all 模式自动启用 no_show
        'window_size': args.window_size,
        'step_size': args.step_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'temperature': args.temperature,
        'device': args.device,
        'num_workers': args.num_workers,
        'use_umap': not args.no_umap,
        'visualize_clusters': not args.no_visualize_clusters,
        'noise_level': 0.05,
        'scale_range': (0.9, 1.1),
        'seed': args.seed,
    }
    
    return config, args


# 统一随机种子设置（与 single_tccl.py 一致）
def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ 随机种子已设置为: {seed}")


# 按 single_tccl.py 的方案评估聚类（不做L2；UMAP->2D；DBSCAN；噪声回填）
def evaluate_clustering_single_style(features: np.ndarray, labels: np.ndarray) -> Dict:
    n_true_classes = len(np.unique(labels))
    print(f"  UMAP dimensionality reduction to 2D (True classes={n_true_classes})...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    features_2d = reducer.fit_transform(features)

    print(f"  Estimating eps parameter...")
    n_samples = len(features_2d)
    min_samples = max(10, min(n_samples // 100, n_true_classes * 5))

    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(features_2d)
    distances, _ = nbrs.kneighbors(features_2d)
    k_distances = np.sort(distances[:, -1])

    eps_candidates = [
        np.percentile(k_distances, 95),
        np.percentile(k_distances, 96),
        np.percentile(k_distances, 97),
        np.percentile(k_distances, 98),
    ]
    eps = np.median(eps_candidates)

    print(f"  Running DBSCAN on 2D space (eps={eps:.4f}, min_samples={min_samples})...")
    max_attempts = 3
    for attempt in range(max_attempts):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        pred_labels = dbscan.fit_predict(features_2d)
        n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
        n_noise = list(pred_labels).count(-1)
        print(f"  → Attempt {attempt+1}: {n_clusters} clusters, {n_noise} noise points ({n_noise/len(pred_labels)*100:.1f}%)")
        if n_clusters <= n_true_classes * 3:
            print(f"  ✓ Cluster count is reasonable (≤ {n_true_classes * 3})")
            break
        elif attempt < max_attempts - 1:
            eps *= 1.3
            print(f"  ⚠ Too many clusters, increasing eps to {eps:.4f}...")
        else:
            print(f"  ⚠ Warning: Still {n_clusters} clusters after {max_attempts} attempts")

    if n_noise > 0:
        valid_mask = pred_labels != -1
        valid_features = features_2d[valid_mask]
        valid_labels = pred_labels[valid_mask]
        noise_mask = pred_labels == -1
        noise_features = features_2d[noise_mask]
        if len(valid_features) > 0:
            knn = KNeighborsClassifier(n_neighbors=min(5, len(valid_features)))
            knn.fit(valid_features, valid_labels)
            pred_labels[noise_mask] = knn.predict(noise_features)
            print(f"  Reassigned noise points to nearest clusters")

    unique_labels = np.unique(pred_labels)
    label_mapping = {old: new for new, old in enumerate(unique_labels)}
    pred_labels = np.array([label_mapping[lbl] for lbl in pred_labels])
    n_clusters = len(unique_labels)

    ari = adjusted_rand_score(labels, pred_labels)
    nmi = normalized_mutual_info_score(labels, pred_labels)
    if n_clusters > 1:
        sil = silhouette_score(features_2d, pred_labels)
    else:
        sil = 0.0

    acc = compute_cluster_accuracy(labels, pred_labels)

    return {
        'acc': acc,
        'ari': ari,
        'nmi': nmi,
        'sil': sil,
        'pred_labels': pred_labels,
        'n_clusters': n_clusters,
        'n_true_classes': n_true_classes,
        'features_reduced': features_2d,
    }


def compute_cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import confusion_matrix
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    cm = confusion_matrix(y_true, y_pred)
    n_clusters = len(np.unique(y_pred))
    best_mapping = {}
    for cluster_id in range(n_clusters):
        best_class = np.argmax(cm[:, cluster_id])
        best_mapping[cluster_id] = best_class
    mapped_pred = np.array([best_mapping[p] for p in y_pred])
    accuracy = np.mean(mapped_pred == y_true)
    return accuracy


def main():
    """主函数"""
    config, args = parse_args()
    
    # 检查是否使用 --all 模式
    if args.all:
        print("=" * 80)
        print("🚀 RUNNING BENCHMARK ON ALL DATASETS")
        print("=" * 80)
        
        # 定义所有数据集配置（使用命令行参数或默认值）
        datasets_config = {
            'CWRU': args.cwru_root,
            'SEU': args.seu_root,
            'MFPT': args.mfpt_root
        }
        
        for dataset_name, data_root in datasets_config.items():
            print(f"\n{'='*80}")
            print(f"📊 Processing dataset: {dataset_name}")
            print(f"{'='*80}")
            
            # 创建当前数据集的配置
            current_config = config.copy()
            current_config['dataset'] = dataset_name
            current_config['data_root'] = data_root
            
            try:
                runner = BenchmarkRunner(current_config)
                runner.run()
                print(f"\n✅ {dataset_name} completed successfully!")
            except Exception as e:
                print(f"\n❌ Error processing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "=" * 80)
        print("✅ ALL DATASETS COMPLETED!")
        print("=" * 80)
    else:
        # 单数据集模式
        runner = BenchmarkRunner(config)
        runner.run()


if __name__ == '__main__':
    main()

