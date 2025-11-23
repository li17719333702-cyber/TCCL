"""
JULE (Joint Unsupervised Learning)

论文: Yang et al., "Joint Unsupervised Learning of Deep Representations and Image Clusters", CVPR 2016

核心思想：
- 联合学习表示和聚类分配
- 通过熵正则化鼓励确定性预测
- 平衡聚类分布
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel


class JULEModel(BaseModel):
    """
    JULE: Joint Unsupervised Learning (CVPR 2016)
    
    方法特点：
    1. 最小化聚类熵（鼓励确定性）
    2. 最大化聚类平衡（防止退化）
    3. 端到端学习
    
    损失函数：
    L = H(clustering) - λ * H(balance)
    其中：
    - H(clustering): 样本级聚类熵，希望最小化（确定性预测）
    - H(balance): 聚类级熵，希望最大化（均匀分布）
    
    优点：
    - 简单有效
    - 不需要聚类中心初始化
    - 自动平衡聚类
    
    缺点：
    - 可能陷入局部最优
    - 对超参数λ敏感
    
    论文链接：https://arxiv.org/abs/1604.03628
    """
    
    def __init__(self, feature_extractor: nn.Module, 
                 n_clusters: int = 10, **kwargs):
        """
        Args:
            feature_extractor: 特征提取器
            n_clusters: 聚类数
        """
        super().__init__(**kwargs)
        self.model_type = 'deep_clustering'
        self.feature_extractor = feature_extractor
        self.n_clusters = n_clusters
        
        # 聚类预测头
        self.cluster_layer = nn.Linear(64, n_clusters)
    
    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算JULE损失
        
        Args:
            view1: 第一个视图 [B, 1, L]
            view2: 第二个视图（不使用）
        
        Returns:
            JULE损失（标量）
        """
        # 提取特征
        feat = self.feature_extractor(view1)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)  # [B, 64]
        
        # 聚类预测
        cluster_pred = self.cluster_layer(feat)  # [B, n_clusters]
        cluster_prob = F.softmax(cluster_pred, dim=1)
        
        # 损失1：聚类熵（鼓励确定性预测）
        # H = -Σ p_i * log(p_i)，最小化
        entropy_loss = -torch.mean(
            torch.sum(cluster_prob * torch.log(cluster_prob + 1e-10), dim=1)
        )
        
        # 损失2：聚类平衡损失（鼓励均匀分布）
        # 计算每个聚类的平均概率
        mean_prob = torch.mean(cluster_prob, dim=0)  # [n_clusters]
        # 最大化聚类分布的熵（等价于最小化负熵）
        balance_loss = -torch.sum(mean_prob * torch.log(mean_prob + 1e-10))
        
        # 总损失：最小化样本内熵，最大化聚类间熵
        loss = entropy_loss - 0.1 * balance_loss
        
        return loss
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征向量"""
        feat = self.feature_extractor(x)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        return feat.cpu().numpy()
    
    def get_cluster_assignment(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取聚类分配
        
        Args:
            x: 输入 [B, 1, L]
        
        Returns:
            聚类ID [B]
        """
        feat = self.feature_extractor(x)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        cluster_pred = self.cluster_layer(feat)
        return torch.argmax(cluster_pred, dim=1)
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'description': 'Joint Unsupervised Learning with entropy regularization',
            'year': 2016,
            'conference': 'CVPR',
            'reference': 'Yang et al., Joint Unsupervised Learning of Deep Representations and Image Clusters',
            'n_clusters': self.n_clusters
        })
        return info

