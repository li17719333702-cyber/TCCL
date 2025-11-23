"""
SimCLR (Simple Framework for Contrastive Learning of Visual Representations)

论文: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020

核心思想：
- 简单的对比学习框架
- 使用数据增强创建正样本对
- InfoNCE损失
- 投影头提升表示质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel


class SimCLRModel(BaseModel):
    """
    SimCLR (ICML 2020)
    
    方法特点：
    1. 简单有效的对比学习框架
    2. 使用投影头（projection head）
    3. InfoNCE损失
    4. 需要大batch size效果更好
    
    损失函数：
    L = -log[exp(sim(z_i, z_j)/τ) / Σ exp(sim(z_i, z_k)/τ)]
    其中：
    - z_i, z_j: 同一样本的两个增强视图的投影
    - τ: 温度参数
    - k: 所有负样本
    
    优点：
    - 简单易实现
    - 效果稳定
    - 可扩展性好
    
    缺点：
    - 需要大batch size
    - 需要大量负样本
    
    论文链接：https://arxiv.org/abs/2002.05709
    """
    
    def __init__(self, feature_extractor: nn.Module,
                 projection_dim: int = 128, hidden_dim: int = 2048, **kwargs):
        """
        Args:
            feature_extractor: 特征提取器
            projection_dim: 投影维度
            hidden_dim: 隐藏层维度
        """
        super().__init__(**kwargs)
        self.model_type = 'contrastive'
        self.feature_extractor = feature_extractor
        
        # 投影头 (2层MLP with BN)
        self.projection_head = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
    
    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回相似度矩阵
        
        Args:
            view1: 第一个视图 [B, 1, L]
            view2: 第二个视图 [B, 1, L]
        
        Returns:
            相似度矩阵 [B, B]
        """
        # 提取特征
        feat1 = self.feature_extractor(view1)
        feat2 = self.feature_extractor(view2)
        
        feat1 = F.adaptive_avg_pool1d(feat1, 1).squeeze(-1)  # [B, 64]
        feat2 = F.adaptive_avg_pool1d(feat2, 1).squeeze(-1)  # [B, 64]
        
        # 投影
        proj1 = self.projection_head(feat1)  # [B, projection_dim]
        proj2 = self.projection_head(feat2)  # [B, projection_dim]
        
        # L2归一化
        proj1_norm = F.normalize(proj1, dim=1)
        proj2_norm = F.normalize(proj2, dim=1)
        
        # 计算相似度矩阵
        similarity = proj1_norm @ proj2_norm.T  # [B, B]
        
        return similarity
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取骨干网络的全局特征（未归一化，64维）。"""
        feat = self.feature_extractor(x)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        return feat.detach().cpu().numpy()
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'description': 'Simple Framework for Contrastive Learning',
            'year': 2020,
            'conference': 'ICML',
            'reference': 'Chen et al., A Simple Framework for Contrastive Learning of Visual Representations',
            'projection_dim': self.projection_head[-1].out_features
        })
        return info

