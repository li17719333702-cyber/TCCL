"""
SCAN (Semantic Clustering by Adopting Nearest neighbors)

论文: Van Gansbeke et al., "SCAN: Learning to Classify Images without Labels", ECCV 2020

核心思想：
- 通过增强视图的一致性学习聚类
- 结合邻居信息
- 同时优化表示和聚类分配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel


class SCANModel(BaseModel):
    """
    SCAN: Semantic Clustering by Adopting Nearest neighbors (ECCV 2020)
    
    方法特点：
    1. 强制增强视图间的聚类一致性
    2. 利用最近邻信息
    3. 联合优化表示和聚类
    
    损失函数：
    L = L_consistency + λ * L_entropy
    其中：
    - L_consistency: 不同视图的聚类预测一致性
    - L_entropy: 熵正则化，鼓励确定性预测
    
    优点：
    - 利用数据增强
    - 不需要聚类中心
    - 效果较好
    
    缺点：
    - 需要好的数据增强
    - 计算开销较大
    
    论文链接：https://arxiv.org/abs/2005.12320
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
        
        # 聚类预测头（2层MLP）
        self.cluster_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_clusters)
        )
    
    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算SCAN损失
        
        Args:
            view1: 第一个视图 [B, 1, L]
            view2: 第二个视图 [B, 1, L]
        
        Returns:
            SCAN损失（标量）
        """
        # 提取特征
        feat1 = self.feature_extractor(view1)
        feat2 = self.feature_extractor(view2)
        
        feat1 = F.adaptive_avg_pool1d(feat1, 1).squeeze(-1)  # [B, 64]
        feat2 = F.adaptive_avg_pool1d(feat2, 1).squeeze(-1)  # [B, 64]
        
        # 聚类预测
        pred1 = self.cluster_head(feat1)  # [B, n_clusters]
        pred2 = self.cluster_head(feat2)  # [B, n_clusters]
        
        # Softmax得到概率
        prob1 = F.softmax(pred1, dim=1)
        prob2 = F.softmax(pred2, dim=1)
        
        # 损失1：一致性损失（两个视图应该有相同的聚类预测）
        # 使用对称的KL散度
        consistency_loss = F.kl_div(prob1.log(), prob2, reduction='batchmean') + \
                          F.kl_div(prob2.log(), prob1, reduction='batchmean')
        
        # 损失2：熵损失（鼓励确定性预测）
        entropy_loss = -torch.mean(
            torch.sum(prob1 * torch.log(prob1 + 1e-10), dim=1)
        )
        
        # 总损失
        loss = consistency_loss + 0.5 * entropy_loss
        
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
        cluster_pred = self.cluster_head(feat)
        return torch.argmax(cluster_pred, dim=1)
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'description': 'Semantic Clustering by Adopting Nearest neighbors',
            'year': 2020,
            'conference': 'ECCV',
            'reference': 'Van Gansbeke et al., SCAN: Learning to Classify Images without Labels',
            'n_clusters': self.n_clusters
        })
        return info

