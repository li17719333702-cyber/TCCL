"""
VICReg (Variance-Invariance-Covariance Regularization)

论文: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning", ICLR 2022

核心思想：
- 通过三个正则化项防止坍塌
- 方差：防止维度坍塌
- 不变性：相同样本的表示应相似
- 协方差：不同维度应独立
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel


class VICRegModel(BaseModel):
    """
    VICReg (ICLR 2022)
    
    特点：
    - 无需负样本
    - 三个正则化项
    - 稳定训练
    
    论文链接：https://arxiv.org/abs/2105.04906
    """
    
    def __init__(self, feature_extractor: nn.Module,
                 projection_dim: int = 128, hidden_dim: int = 2048,
                 sim_weight: float = 25.0, var_weight: float = 25.0,
                 cov_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.model_type = 'contrastive'
        self.feature_extractor = feature_extractor
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        
        # 扩展器
        self.expander = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        feat1 = self.feature_extractor(view1)
        feat2 = self.feature_extractor(view2)
        
        feat1 = F.adaptive_avg_pool1d(feat1, 1).squeeze(-1)
        feat2 = F.adaptive_avg_pool1d(feat2, 1).squeeze(-1)
        
        z1 = self.expander(feat1)
        z2 = self.expander(feat2)
        
        # 不变性损失
        inv_loss = F.mse_loss(z1, z2)
        
        # 方差损失
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        
        # 协方差损失
        N, D = z1.shape
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)
        
        cov_z1 = (z1_centered.T @ z1_centered) / (N - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (N - 1)
        
        cov_loss = self._off_diagonal(cov_z1).pow(2).sum() / D + \
                   self._off_diagonal(cov_z2).pow(2).sum() / D
        
        total_loss = self.sim_weight * inv_loss + \
                     self.var_weight * var_loss + \
                     self.cov_weight * cov_loss
        
        return total_loss
    
    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        proj = self.expander(feat)
        return proj.cpu().numpy()
    
    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            'description': 'Variance-Invariance-Covariance Regularization',
            'year': 2022,
            'conference': 'ICLR',
            'reference': 'Bardes et al., VICReg',
            'weights': f'sim={self.sim_weight}, var={self.var_weight}, cov={self.cov_weight}'
        })
        return info

