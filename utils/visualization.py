"""
可视化模块 - 提供全面的可视化功能

功能包括:
- 训练曲线可视化
- 聚类结果可视化（2D散点图）
- 混淆矩阵可视化
- 性能对比图（雷达图、柱状图）
- 特征分布可视化
- t-SNE/UMAP降维可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torch


# 设置matplotlib中文显示和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')


class TrainingVisualizer:
    """训练过程可视化器"""
    
    @staticmethod
    def plot_training_curve(
        history: Dict[str, List],
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        绘制训练曲线
        
        Args:
            history: 训练历史字典，包含'epoch', 'loss', 'lr'等键
            figsize: 图像大小
            save_path: 保存路径（可选）
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        epochs = history.get('epoch', range(len(history['loss'])))
        
        # 损失曲线
        ax1.plot(epochs, history['loss'], linewidth=2, color='#2ecc71', marker='o', 
                markersize=3, label='Training Loss')
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 学习率曲线
        if 'lr' in history:
            ax2.plot(epochs, history['lr'], linewidth=2, color='#3498db', 
                    marker='s', markersize=3, label='Learning Rate')
            ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
            ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training curve saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_multiple_training_curves(
        histories: Dict[str, Dict],
        metric: str = 'loss',
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        绘制多个模型的训练曲线对比
        
        Args:
            histories: 多个训练历史字典 {model_name: history}
            metric: 要绘制的指标名称
            figsize: 图像大小
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.get_cmap('tab10', len(histories))
        
        for idx, (model_name, history) in enumerate(histories.items()):
            epochs = history.get('epoch', range(len(history[metric])))
            ax.plot(epochs, history[metric], linewidth=2, 
                   color=colors(idx), marker='o', markersize=3, 
                   label=model_name, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.capitalize()} Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison curve saved to {save_path}")
        
        plt.show()


class ClusteringVisualizer:
    """聚类结果可视化器"""
    
    @staticmethod
    def plot_clustering_results(
        features_2d: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        class_names: Optional[Dict[int, str]] = None,
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
    ):
        """
        并排绘制真实标签和聚类结果
        
        Args:
            features_2d: 2D特征 [n_samples, 2]
            true_labels: 真实标签
            pred_labels: 预测标签
            class_names: 类别名称字典
            figsize: 图像大小
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 真实标签
        unique_labels = np.unique(true_labels)
        colors = plt.get_cmap('tab10', len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            mask = (true_labels == label)
            label_name = class_names.get(label, f"Class {label}") if class_names else f"Class {label}"
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[colors(i)], label=label_name, s=20, alpha=0.6,
                       edgecolors='black', linewidth=0.3)
        
        ax1.set_title('True Labels', fontsize=14, fontweight='bold')
        ax1.set_xlabel('UMAP 1', fontsize=11)
        ax1.set_ylabel('UMAP 2', fontsize=11)
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.2)
        
        # 聚类结果
        unique_clusters = np.unique(pred_labels)
        colors_cluster = plt.get_cmap('tab20', len(unique_clusters))
        
        for i, cluster in enumerate(unique_clusters):
            mask = (pred_labels == cluster)
            ax2.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[colors_cluster(i)], label=f'Cluster {cluster}', 
                       s=20, alpha=0.6, edgecolors='black', linewidth=0.3)
        
        ax2.set_title('Predicted Clusters', fontsize=14, fontweight='bold')
        ax2.set_xlabel('UMAP 1', fontsize=11)
        ax2.set_ylabel('UMAP 2', fontsize=11)
        ax2.legend(fontsize=9, loc='best', ncol=2)
        ax2.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Clustering results saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        绘制混淆矩阵
        
        Args:
            confusion_matrix: 混淆矩阵
            class_names: 类别名称列表
            figsize: 图像大小
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names if class_names else 'auto',
                   yticklabels=class_names if class_names else 'auto',
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_cluster_centers(
        features_2d: np.ndarray,
        pred_labels: np.ndarray,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        绘制聚类结果及聚类中心
        
        Args:
            features_2d: 2D特征
            pred_labels: 预测标签
            figsize: 图像大小
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_clusters = np.unique(pred_labels)
        colors = plt.get_cmap('tab10', len(unique_clusters))
        
        # 绘制样本点
        for i, cluster in enumerate(unique_clusters):
            mask = (pred_labels == cluster)
            cluster_points = features_2d[mask]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      c=[colors(i)], label=f'Cluster {cluster}',
                      s=20, alpha=0.5, edgecolors='none')
            
            # 计算并绘制聚类中心
            center = cluster_points.mean(axis=0)
            ax.scatter(center[0], center[1], c=[colors(i)],
                      s=200, alpha=1.0, edgecolors='black',
                      linewidth=2, marker='*', zorder=10)
        
        ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
        ax.set_title('Clustering with Centers', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Cluster centers plot saved to {save_path}")
        
        plt.show()


class PerformanceVisualizer:
    """性能对比可视化器"""
    
    @staticmethod
    def plot_metrics_radar(
        results_dict: Dict[str, Dict],
        metrics: List[str] = ['accuracy_Nto1', 'ari', 'nmi', 'silhouette'],
        figsize: Tuple[int, int] = (10, 10),
        save_path: Optional[str] = None
    ):
        """
        绘制性能对比雷达图
        
        Args:
            results_dict: 结果字典 {model_name: results}
            metrics: 要显示的指标列表
            figsize: 图像大小
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#fafafa')
        
        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 设置标签
        metric_labels = {
            'accuracy_Nto1': 'Accuracy',
            'accuracy_1to1': 'Acc (1:1)',
            'ari': 'ARI',
            'nmi': 'NMI',
            'silhouette': 'Silhouette',
            'separation_ratio': 'Sep. Ratio',
            'davies_bouldin': 'Davies-Bouldin',
            'calinski_harabasz': 'Calinski-H'
        }
        
        labels = [metric_labels.get(m, m) for m in metrics]
        
        # 颜色方案（更柔和的可视化配色）
        palette = sns.color_palette('Set2', n_colors=len(results_dict))
        
        # 绘制每个模型
        for idx, (model_name, results) in enumerate(results_dict.items()):
            values = []
            for metric in metrics:
                val = results.get(metric, 0)
                # 对于Davies-Bouldin（越小越好），取倒数归一化
                if metric == 'davies_bouldin' and val > 0:
                    val = 1.0 / (1.0 + val)
                # 对于Calinski-Harabasz（通常很大），归一化到[0,1]
                if metric == 'calinski_harabasz':
                    val = min(val / 1000.0, 1.0)
                values.append(val)
            
            values += values[:1]  # 闭合
            
            ax.plot(
                angles,
                values,
                'o-',
                linewidth=2.2,
                markersize=4,
                solid_capstyle='round',
                label=model_name,
                color=palette[idx]
            )
            ax.fill(angles, values, alpha=0.12, color=palette[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12, fontweight='semibold', color='#444444')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, color='#808080')
        ax.set_rlabel_position(90)
        ax.grid(True, alpha=0.5, linewidth=0.8, color='#d9d9d9', linestyle='--')
        for spine in ax.spines.values():
            spine.set_color('#d0d0d0')
        # 参考最大轮廓
        ref = np.ones(len(angles))
        ax.plot(angles, ref, color='#bbbbbb', linewidth=1.0, alpha=0.6)
        legend = ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.12),
            ncol=min(3, max(1, len(results_dict))),
            frameon=True,
            fontsize=11,
            handlelength=1.8,
            columnspacing=1.2,
            handletextpad=0.6,
            borderaxespad=0.2,
            fancybox=True
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('#dddddd')
        legend.get_frame().set_linewidth(1.0)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Radar chart saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_metrics_bars(
        results_dict: Dict[str, Dict],
        metrics: List[str] = ['accuracy_Nto1', 'ari', 'nmi', 'silhouette'],
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ):
        """
        绘制性能对比柱状图
        
        Args:
            results_dict: 结果字典
            metrics: 要显示的指标列表
            figsize: 图像大小
            save_path: 保存路径
        """
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        metric_names = {
            'accuracy_Nto1': 'Clustering Accuracy (N-to-1)',
            'accuracy_1to1': 'Clustering Accuracy (1-to-1)',
            'ari': 'Adjusted Rand Index',
            'nmi': 'Normalized Mutual Information',
            'silhouette': 'Silhouette Coefficient',
            'davies_bouldin': 'Davies-Bouldin Index',
            'calinski_harabasz': 'Calinski-Harabasz Score',
            'separation_ratio': 'Separation Ratio'
        }
        
        colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
                       '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            models = list(results_dict.keys())
            values = [results_dict[m].get(metric, 0) for m in models]
            
            bars = ax.bar(range(len(models)), values,
                         color=colors_list[idx % len(colors_list)],
                         edgecolor='black', linewidth=1.5, alpha=0.8)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
            
            ax.set_ylabel(metric_names.get(metric, metric),
                         fontsize=10, fontweight='bold')
            ax.set_title(metric_names.get(metric, metric),
                        fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
        
        # 隐藏多余的子图
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Performance Comparison Across Metrics',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Bar charts saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_comparison_table(
        results_dict: Dict[str, Dict],
        metrics: List[str] = ['accuracy_Nto1', 'ari', 'nmi', 'silhouette',
                              'davies_bouldin', 'separation_ratio'],
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
    ):
        """
        绘制性能对比表格
        
        Args:
            results_dict: 结果字典
            metrics: 要显示的指标列表
            figsize: 图像大小
            save_path: 保存路径
        """
        # 准备数据
        table_data = []
        for model_name, results in results_dict.items():
            row = [model_name]
            for metric in metrics:
                val = results.get(metric, np.nan)
                row.append(f"{val:.4f}" if not np.isnan(val) else "N/A")
            table_data.append(row)
        
        # 创建表格
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('tight')
        ax.axis('off')
        
        metric_labels = {
            'accuracy_Nto1': 'Acc (N:1)',
            'ari': 'ARI',
            'nmi': 'NMI',
            'silhouette': 'Silhouette',
            'davies_bouldin': 'DB Index',
            'separation_ratio': 'Sep. Ratio',
            'calinski_harabasz': 'CH Score'
        }
        
        columns = ['Model'] + [metric_labels.get(m, m) for m in metrics]
        
        table = ax.table(cellText=table_data, colLabels=columns,
                        cellLoc='center', loc='center',
                        colWidths=[0.2] + [0.12] * len(metrics))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置交替行颜色
        for i in range(1, len(table_data) + 1):
            for j in range(len(columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        plt.title('Performance Comparison Table', fontsize=16,
                 fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison table saved to {save_path}")
        
        plt.show()


def save_all_visualizations(
    results: Dict[str, Any],
    history: Optional[Dict] = None,
    save_dir: str = './visualizations',
    prefix: str = ''
):
    """
    保存所有可视化结果
    
    Args:
        results: 评估结果字典
        history: 训练历史（可选）
        save_dir: 保存目录
        prefix: 文件名前缀
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Saving Visualizations")
    print("=" * 60)
    
    # 1. 训练曲线
    if history is not None:
        TrainingVisualizer.plot_training_curve(
            history,
            save_path=save_dir / f"{prefix}training_curve.png"
        )
    
    # 2. 聚类结果
    if results.get('features_2d') is not None:
        ClusteringVisualizer.plot_clustering_results(
            results['features_2d'],
            results['labels'],
            results['pred_labels'],
            save_path=save_dir / f"{prefix}clustering_results.png"
        )
        
        ClusteringVisualizer.plot_cluster_centers(
            results['features_2d'],
            results['pred_labels'],
            save_path=save_dir / f"{prefix}cluster_centers.png"
        )
    
    # 3. 混淆矩阵
    if 'confusion_matrix' in results:
        ClusteringVisualizer.plot_confusion_matrix(
            results['confusion_matrix'],
            save_path=save_dir / f"{prefix}confusion_matrix.png"
        )
    
    print(f"\n✓ All visualizations saved to {save_dir}")


if __name__ == "__main__":
    """测试代码"""
    print("Testing Visualization Module...")
    
    # 生成测试数据
    from sklearn.datasets import make_blobs
    
    X, y = make_blobs(n_samples=300, n_features=2, centers=4, random_state=42)
    pred_y = np.random.randint(0, 4, size=300)
    
    # 测试训练曲线
    history = {
        'epoch': list(range(50)),
        'loss': np.random.exponential(0.5, 50).cumsum()[::-1] / 10,
        'lr': [0.001 * (0.95 ** i) for i in range(50)]
    }
    TrainingVisualizer.plot_training_curve(history)
    
    # 测试聚类可视化
    ClusteringVisualizer.plot_clustering_results(X, y, pred_y)
    
    print("\n✓ Visualization test passed!")

