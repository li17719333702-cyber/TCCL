"""
基准测试辅助工具

提供用于基准测试的各种辅助功能：
- 特征预处理
- 自动聚类数搜索
- 评估指标计算
- 结果可视化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix, adjusted_rand_score,
    normalized_mutual_info_score, silhouette_score,
    davies_bouldin_score, calinski_harabasz_score
)
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
import umap.umap_ as umap
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def preprocess_features(features: np.ndarray, 
                       method: str = 'l2',
                       is_handcrafted: bool = False) -> np.ndarray:
    """
    预处理特征
    
    Args:
        features: 特征矩阵 [N, D]
        method: 预处理方法 ('l2', 'standard', 'none')
        is_handcrafted: 是否为手工特征
    
    Returns:
        预处理后的特征
    """
    if method == 'none':
        return features
    
    if is_handcrafted or method == 'standard':
        # 手工特征使用标准化
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)
    else:
        # 深度学习特征使用L2归一化
        features_norm = normalize(features, norm='l2', axis=1)
    
    return features_norm


def find_optimal_k(features: np.ndarray,
                  k_range: range = range(2, 20)) -> Tuple[int, list, list]:
    """
    使用轮廓系数寻找最优聚类数
    
    Args:
        features: 特征矩阵 [N, D]
        k_range: 聚类数搜索范围
    
    Returns:
        optimal_k: 最优聚类数
        k_values: 测试的K值列表
        silhouette_scores: 对应的轮廓系数列表
    """
    silhouette_scores = []
    k_values = list(k_range)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(features)
        sil_score = silhouette_score(features, pred_labels)
        silhouette_scores.append(sil_score)
    
    optimal_k = k_values[np.argmax(silhouette_scores)]
    
    return optimal_k, k_values, silhouette_scores


def compute_clustering_metrics(features: np.ndarray,
                               labels: np.ndarray,
                               n_clusters: Optional[int] = None,
                               use_umap: bool = True) -> Dict:
    """
    计算全面的聚类评估指标
    
    Args:
        features: 特征矩阵 [N, D]
        labels: 真实标签 [N]
        n_clusters: 聚类数（None表示自动搜索）
        use_umap: 是否使用UMAP降维
    
    Returns:
        包含所有评估指标的字典
    """
    # 特征预处理
    if use_umap:
        reducer = umap.UMAP(
            n_neighbors=20,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        features_2d = reducer.fit_transform(features)
        features_to_cluster = features_2d
    else:
        features_to_cluster = features
        features_2d = None
    
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_to_cluster)
    
    # 确定聚类数
    n_true_classes = len(np.unique(labels))
    if n_clusters is None:
        optimal_k, _, _ = find_optimal_k(features_scaled)
    else:
        optimal_k = n_clusters
    
    # K-Means聚类
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    pred_labels = kmeans.fit_predict(features_scaled)
    
    # 无监督指标
    silhouette = silhouette_score(features_scaled, pred_labels)
    davies_bouldin = davies_bouldin_score(features_scaled, pred_labels)
    calinski_harabasz = calinski_harabasz_score(features_scaled, pred_labels)
    
    # 有监督指标
    ari = adjusted_rand_score(labels, pred_labels)
    nmi = normalized_mutual_info_score(labels, pred_labels)
    
    # 准确率计算
    cm = confusion_matrix(labels, pred_labels)
    
    # 1对1映射（仅当簇数等于类数时）
    if optimal_k == n_true_classes:
        row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
        accuracy_one_to_one = cm[row_ind, col_ind].sum() / len(labels)
        mapping_1to1 = dict(zip(col_ind, row_ind))
    else:
        accuracy_one_to_one = None
        mapping_1to1 = None
    
    # 多对一映射
    best_mapping = {}
    for cluster_id in range(optimal_k):
        best_class = np.argmax(cm[:, cluster_id])
        best_mapping[cluster_id] = best_class
    
    mapped_pred = np.array([best_mapping[p] for p in pred_labels])
    accuracy_many_to_one = np.mean(mapped_pred == labels)
    
    # 类内类间距离
    intra_cluster_dist = 0
    for cluster_id in range(optimal_k):
        cluster_mask = (pred_labels == cluster_id)
        if cluster_mask.sum() > 0:
            cluster_features = features_scaled[cluster_mask]
            center = cluster_features.mean(axis=0)
            intra_cluster_dist += np.mean(pairwise_distances(cluster_features, [center]))
    intra_cluster_dist /= optimal_k
    
    centers = np.array([features_scaled[pred_labels == i].mean(axis=0) 
                       for i in range(optimal_k)])
    inter_cluster_dist = np.mean(pairwise_distances(centers))
    
    separation_ratio = inter_cluster_dist / (intra_cluster_dist + 1e-8)
    
    # 构建结果字典
    metrics = {
        'n_clusters': optimal_k,
        'n_true_classes': n_true_classes,
        'accuracy_Nto1': accuracy_many_to_one,
        'accuracy_1to1': accuracy_one_to_one,
        'ari': ari,
        'nmi': nmi,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz,
        'intra_dist': intra_cluster_dist,
        'inter_dist': inter_cluster_dist,
        'separation_ratio': separation_ratio,
        'mapping_Nto1': best_mapping,
        'mapping_1to1': mapping_1to1,
        'confusion_matrix': cm
    }
    
    return metrics, pred_labels, features_2d


def train_model(model: nn.Module,
               train_loader: DataLoader,
               device: str,
               epochs: int = 50,
               learning_rate: float = 0.001,
               temperature: float = 0.1,
               verbose: bool = True) -> list:
    """
    训练模型（统一的训练流程）
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        device: 设备
        epochs: 训练轮数
        learning_rate: 学习率
        temperature: 温度参数（对比学习）
        verbose: 是否打印进度
    
    Returns:
        训练损失历史
    """
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    # ⚡ 优化：使用 epochs 作为 T_max，避免计算 len(train_loader)
    # 这样学习率在整个训练过程中平滑衰减，而不是每个 batch
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,  # 改为按 epoch 衰减，而不是按 batch
        eta_min=0
    )
    
    history = []
    
    iterator = range(epochs) if not verbose else tqdm(range(epochs), desc="  Training")
    
    for epoch in iterator:
        total_loss = 0
        n_batches = 0  # ⚡ 计数器，避免使用 len(train_loader)
        
        for view1, view2, _ in train_loader:
            view1, view2 = view1.to(device), view2.to(device)
            batch_size = view1.size(0)
            
            optimizer.zero_grad()
            pending_queue_keys = None
            
            # 根据模型类型选择不同的训练方式
            if model.model_type == 'deep_clustering':
                # 深度聚类：直接使用返回的损失
                loss = model(view1, view2)
            
            elif model.model_type == 'contrastive':
                # 对比学习
                output = model(view1, view2)
                
                # 三种情况：
                # 1) VICReg 等：返回标量损失
                if isinstance(output, torch.Tensor) and output.dim() == 0:
                    loss = output
                # 2) MoCo 全量：返回 (logits, targets[, keys])
                elif isinstance(output, (tuple, list)) and (len(output) == 2 or len(output) == 3):
                    if len(output) == 3:
                        logits, targets, pending_queue_keys = output
                    else:
                        logits, targets = output
                    loss = F.cross_entropy(logits / temperature, targets)
                # 3) 其他对比：返回相似度矩阵 [B, B]
                else:
                    sim_matrix = output
                    labels = torch.arange(batch_size, device=device)
                    logits = sim_matrix / temperature
                    loss_12 = F.cross_entropy(logits, labels)
                    # 对称项
                    logits_rev = sim_matrix.T / temperature
                    loss_21 = F.cross_entropy(logits_rev, labels)
                    loss = (loss_12 + loss_21) / 2
            else:
                raise ValueError(f"Unknown model type: {model.model_type}")
            
            loss.backward()
            optimizer.step()

            # 安全地在反向传播后更新 MoCo 队列
            if pending_queue_keys is not None and hasattr(model, 'enqueue_dequeue'):
                with torch.no_grad():
                    model.enqueue_dequeue(pending_queue_keys)
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches  # ⚡ 使用计数器而不是 len(train_loader)
        history.append(avg_loss)
        
        # ⚡ 在每个 epoch 结束后更新学习率
        # scheduler.step()
        
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            if isinstance(iterator, range):
                print(f"    Epoch [{epoch+1:3d}/{epochs}] Loss: {avg_loss:.6f}")
    
    return history


def extract_features_from_loader(model: nn.Module,
                                 dataloader: DataLoader,
                                 device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从数据加载器中提取所有特征
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        features: 特征矩阵 [N, D]
        labels: 标签 [N]
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for view1, _, labels in dataloader:
            features = model.extract_features(view1.to(device))
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()
            all_features.append(features)
            all_labels.append(labels.numpy())
    
    features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)
    
    return features, labels

