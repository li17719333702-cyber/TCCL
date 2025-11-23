"""
SimSiam (Exploring Simple Siamese Representation Learning)

论文: Chen & He, "Exploring Simple Siamese Representation Learning", CVPR 2021

核心思想：
- 极简的孪生网络
- 无需负样本、动量编码器、大batch size
- 仅通过stop-gradient防止坍塌
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel


class SimSiamModel(BaseModel):
    """
    SimSiam (CVPR 2021)
    
    特点：
    - 极简设计
    - 不需要负样本
    - 不需要动量编码器
    - 不需要大batch size
    
    论文链接：https://arxiv.org/abs/2011.10566
    """
    
    def __init__(self, feature_extractor: nn.Module,
                 projection_dim: int = 128, hidden_dim: int = 2048, **kwargs):
        super().__init__(**kwargs)
        self.model_type = 'contrastive'
        self.feature_extractor = feature_extractor
        
        # 投影头（3层MLP）
        self.projection_head = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
        
        # 预测头（2层MLP）
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, projection_dim)
        )
    
    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """
        SimSiam loss (scalar):
        loss = D(p1, stopgrad(z2)) / 2 + D(p2, stopgrad(z1)) / 2
        where D(a,b) = -cosine_similarity(a,b)
        """
        feat1 = self.feature_extractor(view1)
        feat2 = self.feature_extractor(view2)

        feat1 = F.adaptive_avg_pool1d(feat1, 1).squeeze(-1)
        feat2 = F.adaptive_avg_pool1d(feat2, 1).squeeze(-1)

        z1 = self.projection_head(feat1)
        z2 = self.projection_head(feat2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # normalize
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # stop-gradient on targets
        loss1 = - (p1 * z2.detach()).sum(dim=1).mean()
        loss2 = - (p2 * z1.detach()).sum(dim=1).mean()
        loss = 0.5 * (loss1 + loss2)
        return loss
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        proj = self.projection_head(feat)
        proj_norm = F.normalize(proj, dim=1)
        return proj_norm.cpu().numpy()
    
    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            'description': 'Simple Siamese Representation Learning',
            'year': 2021,
            'conference': 'CVPR',
            'reference': 'Chen & He, Exploring Simple Siamese Representation Learning'
        })
        return info

