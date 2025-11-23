"""
评估模块 - 提供全面的聚类评估指标

功能包括:
- 无监督聚类指标：Silhouette、Davies-Bouldin、Calinski-Harabasz
- 有监督聚类指标：ARI、NMI
- 聚类准确率：1对1映射、多对1映射
- 分离度指标：类内/类间距离、分离比
- 最优聚类数搜索
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
    pairwise_distances
)
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple, Optional, List
import umap.umap_ as umap
from tqdm import tqdm


class ClusteringEvaluator:
    """聚类评估器"""
    
    def __init__(self, n_clusters: Optional[int] = None, use_umap: bool = True,
                 umap_params: Optional[Dict] = None):
        """
        初始化评估器
        
        Args:
            n_clusters: 聚类数量，None表示自动搜索最优K
            use_umap: 是否使用UMAP降维到2D进行聚类
            umap_params: UMAP参数字典
        """
        self.n_clusters = n_clusters
        self.use_umap = use_umap
        
        # 默认UMAP参数
        if umap_params is None:
            self.umap_params = {
                'n_neighbors': 20,
                'min_dist': 0.1,
                'metric': 'cosine',
                'random_state': 42
            }
        else:
            self.umap_params = umap_params
        
        self.reducer = None
        self.kmeans = None
        self.scaler = StandardScaler()
    
    def fit_transform_umap(self, features: np.ndarray) -> np.ndarray:
        """
        使用UMAP降维到2D
        
        Args:
            features: 高维特征 [n_samples, n_features]
        
        Returns:
            features_2d: 2D特征 [n_samples, 2]
        """
        print("  Fitting UMAP...")
        self.reducer = umap.UMAP(**self.umap_params)
        features_2d = self.reducer.fit_transform(features)
        return features_2d
    
    def find_optimal_k(self, features: np.ndarray, k_range: range = None) -> int:
        """
        使用轮廓系数寻找最优聚类数
        
        Args:
            features: 特征矩阵 [n_samples, n_features]
            k_range: 搜索范围
        
        Returns:
            optimal_k: 最优聚类数
        """
        if k_range is None:
            k_range = range(2, min(20, len(features) // 10))
        
        print(f"  Searching optimal K in range {list(k_range)[:3]}...{list(k_range)[-3:]}...")
        
        silhouettes = []
        for k in tqdm(k_range, desc="  Finding optimal K", leave=False):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            pred_labels = kmeans.fit_predict(features)
            silhouettes.append(silhouette_score(features, pred_labels))
        
        optimal_k = list(k_range)[np.argmax(silhouettes)]
        print(f"  ✓ Optimal K: {optimal_k} (Silhouette: {max(silhouettes):.4f})")
        
        return optimal_k
    
    def fit_kmeans(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        执行K-Means聚类
        
        Args:
            features: 特征矩阵 [n_samples, n_features]
            n_clusters: 聚类数
        
        Returns:
            pred_labels: 预测标签 [n_samples]
        """
        print(f"  Running K-Means (K={n_clusters})...")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        pred_labels = self.kmeans.fit_predict(features)
        return pred_labels
    
    def compute_unsupervised_metrics(
        self, 
        features: np.ndarray, 
        pred_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        计算无监督聚类指标
        
        Args:
            features: 特征矩阵
            pred_labels: 预测标签
        
        Returns:
            metrics: 指标字典
        """
        metrics = {}
        
        # Silhouette系数 (越大越好，范围[-1, 1])
        metrics['silhouette'] = silhouette_score(features, pred_labels)
        
        # Davies-Bouldin指数 (越小越好)
        metrics['davies_bouldin'] = davies_bouldin_score(features, pred_labels)
        
        # Calinski-Harabasz指数 (越大越好)
        metrics['calinski_harabasz'] = calinski_harabasz_score(features, pred_labels)
        
        return metrics
    
    def compute_supervised_metrics(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        计算有监督聚类指标
        
        Args:
            true_labels: 真实标签
            pred_labels: 预测标签
        
        Returns:
            metrics: 指标字典
        """
        metrics = {}
        
        # Adjusted Rand Index (范围[-1, 1]，越大越好)
        metrics['ari'] = adjusted_rand_score(true_labels, pred_labels)
        
        # Normalized Mutual Information (范围[0, 1]，越大越好)
        metrics['nmi'] = normalized_mutual_info_score(true_labels, pred_labels)
        
        return metrics
    
    def compute_accuracy(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray
    ) -> Dict[str, any]:
        """
        计算聚类准确率（需要标签映射）
        
        Args:
            true_labels: 真实标签
            pred_labels: 预测标签
        
        Returns:
            accuracy_metrics: 包含准确率和映射关系的字典
        """
        metrics = {}
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, pred_labels)
        n_true_classes = len(np.unique(true_labels))
        n_pred_clusters = len(np.unique(pred_labels))
        
        # 1对1映射（仅当簇数等于类数时）
        if n_pred_clusters == n_true_classes:
            row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
            accuracy_1to1 = cm[row_ind, col_ind].sum() / len(true_labels)
            metrics['accuracy_1to1'] = accuracy_1to1
            metrics['mapping_1to1'] = dict(zip(col_ind, row_ind))
        else:
            metrics['accuracy_1to1'] = None
            metrics['mapping_1to1'] = None
        
        # 多对1映射（N-to-1）
        best_mapping = {}
        for cluster_id in range(n_pred_clusters):
            best_class = np.argmax(cm[:, cluster_id])
            best_mapping[cluster_id] = best_class
        
        mapped_pred = np.array([best_mapping[p] for p in pred_labels])
        accuracy_Nto1 = np.mean(mapped_pred == true_labels)
        
        metrics['accuracy_Nto1'] = accuracy_Nto1
        metrics['mapping_Nto1'] = best_mapping
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def compute_separation_metrics(
        self,
        features: np.ndarray,
        pred_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        计算类内/类间距离和分离比
        
        Args:
            features: 特征矩阵
            pred_labels: 预测标签
        
        Returns:
            metrics: 分离度指标字典
        """
        n_clusters = len(np.unique(pred_labels))
        
        # 计算类内距离（平均）
        intra_cluster_dist = 0
        for cluster_id in range(n_clusters):
            cluster_mask = (pred_labels == cluster_id)
            if cluster_mask.sum() > 0:
                cluster_features = features[cluster_mask]
                center = cluster_features.mean(axis=0, keepdims=True)
                intra_cluster_dist += np.mean(pairwise_distances(cluster_features, center))
        intra_cluster_dist /= n_clusters
        
        # 计算类间距离（聚类中心之间的平均距离）
        centers = np.array([features[pred_labels == i].mean(axis=0) 
                           for i in range(n_clusters)])
        inter_cluster_dist = np.mean(pairwise_distances(centers))
        
        # 分离比（类间距离/类内距离，越大越好）
        separation_ratio = inter_cluster_dist / (intra_cluster_dist + 1e-8)
        
        metrics = {
            'intra_dist': intra_cluster_dist,
            'inter_dist': inter_cluster_dist,
            'separation_ratio': separation_ratio
        }
        
        return metrics
    
    def evaluate(
        self,
        features: np.ndarray,
        true_labels: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        完整的聚类评估流程
        
        Args:
            features: 特征矩阵 [n_samples, n_features]
            true_labels: 真实标签 [n_samples]
            verbose: 是否打印详细信息
        
        Returns:
            results: 包含所有评估指标和中间结果的字典
        """
        if verbose:
            print("=" * 60)
            print("Clustering Evaluation")
            print("=" * 60)
            print(f"Features shape: {features.shape}")
            print(f"Number of samples: {len(features)}")
            print(f"Number of true classes: {len(np.unique(true_labels))}")
        
        results = {}
        
        # 步骤1: UMAP降维（可选）
        if self.use_umap:
            features_2d = self.fit_transform_umap(features)
            features_to_cluster = features_2d
            results['features_2d'] = features_2d
        else:
            features_to_cluster = features
            results['features_2d'] = None
        
        # 步骤2: 特征标准化
        features_scaled = self.scaler.fit_transform(features_to_cluster)
        results['features_scaled'] = features_scaled
        
        # 步骤3: 确定聚类数
        n_true_classes = len(np.unique(true_labels))
        if self.n_clusters is None:
            optimal_k = self.find_optimal_k(features_scaled)
        else:
            optimal_k = self.n_clusters
        
        results['n_clusters'] = optimal_k
        results['n_true_classes'] = n_true_classes
        
        # 步骤4: K-Means聚类
        pred_labels = self.fit_kmeans(features_scaled, optimal_k)
        results['pred_labels'] = pred_labels
        
        # 步骤5: 计算无监督指标
        if verbose:
            print("\n📊 Computing metrics...")
        
        unsupervised_metrics = self.compute_unsupervised_metrics(
            features_scaled, pred_labels
        )
        results.update(unsupervised_metrics)
        
        # 步骤6: 计算有监督指标
        supervised_metrics = self.compute_supervised_metrics(
            true_labels, pred_labels
        )
        results.update(supervised_metrics)
        
        # 步骤7: 计算准确率
        accuracy_metrics = self.compute_accuracy(
            true_labels, pred_labels
        )
        results.update(accuracy_metrics)
        
        # 步骤8: 计算分离度
        separation_metrics = self.compute_separation_metrics(
            features_scaled, pred_labels
        )
        results.update(separation_metrics)
        
        # 打印结果
        if verbose:
            print("\n" + "=" * 60)
            print("Evaluation Results")
            print("=" * 60)
            print(f"Optimal K: {optimal_k} (True K: {n_true_classes})")
            print(f"\n📊 Unsupervised Metrics:")
            print(f"  Silhouette Score:      {unsupervised_metrics['silhouette']:.4f}")
            print(f"  Davies-Bouldin Index:  {unsupervised_metrics['davies_bouldin']:.4f}")
            print(f"  Calinski-Harabasz:     {unsupervised_metrics['calinski_harabasz']:.1f}")
            
            print(f"\n📈 Supervised Metrics:")
            print(f"  Adjusted Rand Index:   {supervised_metrics['ari']:.4f}")
            print(f"  Normalized Mutual Info:{supervised_metrics['nmi']:.4f}")
            
            print(f"\n🎯 Accuracy:")
            if accuracy_metrics['accuracy_1to1'] is not None:
                print(f"  1-to-1 Accuracy:       {accuracy_metrics['accuracy_1to1']:.4f}")
            print(f"  N-to-1 Accuracy:       {accuracy_metrics['accuracy_Nto1']:.4f}")
            
            print(f"\n🔍 Separation Metrics:")
            print(f"  Intra-cluster Distance:{separation_metrics['intra_dist']:.4f}")
            print(f"  Inter-cluster Distance:{separation_metrics['inter_dist']:.4f}")
            print(f"  Separation Ratio:      {separation_metrics['separation_ratio']:.4f}")
            print("=" * 60)
        
        return results


def evaluate_model(
    model,
    dataloader,
    device: str = 'cuda',
    use_umap: bool = True,
    n_clusters: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    便捷函数：评估TCCL模型
    
    Args:
        model: TCCL模型
        dataloader: 数据加载器
        device: 设备
        use_umap: 是否使用UMAP降维
        n_clusters: 聚类数（None表示自动搜索）
        verbose: 是否打印详细信息
    
    Returns:
        results: 评估结果字典
    """
    # 提取特征
    if verbose:
        print("Extracting features...")
    
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for view1, _, labels in tqdm(dataloader, desc="Extracting", disable=not verbose):
            view1 = view1.to(device)
            features = model.extract_features(view1, normalize=True)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    if verbose:
        print(f"✓ Extracted {len(features)} samples with {features.shape[1]} dimensions")
    
    # 创建评估器并评估
    evaluator = ClusteringEvaluator(n_clusters=n_clusters, use_umap=use_umap)
    results = evaluator.evaluate(features, labels, verbose=verbose)
    
    # 添加原始特征和标签
    results['features'] = features
    results['labels'] = labels
    
    return results


def compare_models(
    models_dict: Dict[str, any],
    dataloader,
    device: str = 'cuda',
    use_umap: bool = True,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    比较多个模型的性能
    
    Args:
        models_dict: 模型字典 {model_name: model}
        dataloader: 数据加载器
        device: 设备
        use_umap: 是否使用UMAP
        verbose: 是否打印详细信息
    
    Returns:
        comparison_results: 比较结果字典
    """
    comparison_results = {}
    
    print("=" * 80)
    print("Comparing Multiple Models")
    print("=" * 80)
    
    for model_name, model in models_dict.items():
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*80}")
        
        results = evaluate_model(
            model, dataloader, device=device,
            use_umap=use_umap, verbose=verbose
        )
        
        comparison_results[model_name] = results
    
    # 打印对比表格
    print("\n" + "=" * 80)
    print("Performance Comparison")
    print("=" * 80)
    print(f"{'Model':<20} {'Acc↑':>8} {'ARI↑':>8} {'NMI↑':>8} {'Sil↑':>8} {'SepR↑':>8} {'DB↓':>8}")
    print("-" * 80)
    
    for model_name, results in comparison_results.items():
        acc = results['accuracy_Nto1']
        ari = results['ari']
        nmi = results['nmi']
        sil = results['silhouette']
        sep = results['separation_ratio']
        db = results['davies_bouldin']
        
        print(f"{model_name:<20} {acc:>8.4f} {ari:>8.4f} {nmi:>8.4f} "
              f"{sil:>8.4f} {sep:>8.4f} {db:>8.4f}")
    
    print("=" * 80)
    
    return comparison_results


if __name__ == "__main__":
    """测试代码"""
    print("Testing Clustering Evaluator...")
    
    # 生成测试数据
    from sklearn.datasets import make_blobs
    
    X, y = make_blobs(n_samples=300, n_features=64, centers=4, random_state=42)
    
    # 创建评估器
    evaluator = ClusteringEvaluator(n_clusters=None, use_umap=True)
    
    # 执行评估
    results = evaluator.evaluate(X, y, verbose=True)
    
    print("\n✓ Evaluation test passed!")

