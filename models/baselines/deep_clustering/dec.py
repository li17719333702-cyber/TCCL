"""
DEC (Deep Embedded Clustering)

论文: Xie et al., "Unsupervised Deep Embedding for Clustering Analysis", ICML 2016

核心思想：
- 通过自编码器学习嵌入表示
- 使用Student's t-distribution作为软分配
- 通过KL散度优化聚类中心
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel


class DECModel(BaseModel):
    """
    DEC: Deep Embedded Clustering (ICML 2016)
    
    方法特点：
    1. 使用可学习的聚类中心
    2. Student's t-distribution计算软分配
    3. 最小化当前分配与目标分布的KL散度
    
    优点：
    - 端到端学习
    - 同时优化表示和聚类
    - 效果稳定
    
    缺点：
    - 需要预训练或初始化
    - 对聚类数敏感
    
    论文链接：https://arxiv.org/abs/1511.06335
    """
    
    def __init__(self, feature_extractor: nn.Module, 
                 n_clusters: int = 10, alpha: float = 1.0, **kwargs):
        """
        Args:
            feature_extractor: 特征提取器
            n_clusters: 聚类数
            alpha: Student's t-distribution的自由度参数
        """
        super().__init__(**kwargs)
        self.model_type = 'deep_clustering'
        self.feature_extractor = feature_extractor
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # 聚类中心（可学习参数）
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, 64))
    
    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算DEC损失
        
        Args:
            view1: 第一个视图 [B, 1, L]
            view2: 第二个视图（不使用）
        
        Returns:
            KL散度损失（标量）
        """
        # 提取特征
        feat = self.feature_extractor(view1)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)  # [B, 64]
        
        # 计算软分配 q
        q = self._soft_assignment(feat)
        
        # 计算目标分布 p
        p = self._target_distribution(q)
        
        # KL散度损失
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        
        return loss
    
    def _soft_assignment(self, feat: torch.Tensor) -> torch.Tensor:
        """
        计算软分配矩阵 Q
        
        使用Student's t-distribution作为核函数：
        q_ij = (1 + ||z_i - μ_j||^2 / α)^(-(α+1)/2) / Σ_j'(...)
        
        Args:
            feat: 特征 [B, 64]
        
        Returns:
            软分配矩阵 [B, n_clusters]
        """
        # 计算距离：[B, n_clusters]
        distances = torch.sum((feat.unsqueeze(1) - self.cluster_centers)**2, dim=2)
        
        # Student's t-distribution
        q = 1.0 / (1.0 + distances / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        
        return q
    
    def _target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """
        计算目标分布 P
        
        p_ij = q_ij^2 / Σ_i q_ij / Σ_j'(...)
        
        这个目标分布具有以下特性：
        - 增强高置信度的分配
        - 防止大簇主导
        - 归一化以保持概率分布
        
        Args:
            q: 软分配矩阵 [B, n_clusters]
        
        Returns:
            目标分布 [B, n_clusters]
        """
        weight = q ** 2 / torch.sum(q, dim=0, keepdim=True)
        p = weight / torch.sum(weight, dim=1, keepdim=True)
        return p
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征向量"""
        feat = self.feature_extractor(x)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        return feat.cpu().numpy()
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'description': 'Deep Embedded Clustering with learnable cluster centers',
            'year': 2016,
            'conference': 'ICML',
            'reference': 'Xie et al., Unsupervised Deep Embedding for Clustering Analysis',
            'n_clusters': self.n_clusters,
            'alpha': self.alpha
        })
        return info

