"""
BYOL (Bootstrap Your Own Latent)

论文: Grill et al., "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning", NeurIPS 2020

核心思想：
- 无需负样本
- 使用动量编码器和预测器
- 通过stop-gradient防止坍塌
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ..base import BaseModel


class BYOLModel(BaseModel):
    """
    BYOL: Bootstrap Your Own Latent (NeurIPS 2020)
    
    特点：
    - 无需负样本对比
    - 使用预测器（predictor）
    - 动量更新目标网络
    
    论文链接：https://arxiv.org/abs/2006.07733
    """
    
    def __init__(self, feature_extractor: nn.Module,
                 projection_dim: int = 128, hidden_dim: int = 2048,
                 momentum: float = 0.999, **kwargs):
        super().__init__(**kwargs)
        self.model_type = 'contrastive'
        self.feature_extractor = feature_extractor
        self.momentum = momentum
        
        # 在线网络
        self.online_head = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
        
        # 预测器
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # 目标网络
        self.target_extractor = copy.deepcopy(feature_extractor)
        self.target_head = copy.deepcopy(self.online_head)
        
        for param in self.target_extractor.parameters():
            param.requires_grad = False
        for param in self.target_head.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def _update_target_network(self):
        for param_online, param_target in zip(
            self.feature_extractor.parameters(),
            self.target_extractor.parameters()
        ):
            param_target.data = param_target.data * self.momentum + \
                               param_online.data * (1 - self.momentum)
        
        for param_online, param_target in zip(
            self.online_head.parameters(),
            self.target_head.parameters()
        ):
            param_target.data = param_target.data * self.momentum + \
                               param_online.data * (1 - self.momentum)
    
    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        feat1 = self.feature_extractor(view1)
        feat1 = F.adaptive_avg_pool1d(feat1, 1).squeeze(-1)
        proj1 = self.online_head(feat1)
        pred1 = self.predictor(proj1)
        pred1 = F.normalize(pred1, dim=1)
        
        with torch.no_grad():
            self._update_target_network()
            feat2 = self.target_extractor(view2)
            feat2 = F.adaptive_avg_pool1d(feat2, 1).squeeze(-1)
            proj2 = self.target_head(feat2)
            proj2 = F.normalize(proj2, dim=1)
        
        sim_matrix = pred1 @ proj2.T
        return sim_matrix
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        proj = self.online_head(feat)
        proj = F.normalize(proj, dim=1)
        return proj.cpu().numpy()
    
    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            'description': 'Bootstrap Your Own Latent without negative samples',
            'year': 2020,
            'conference': 'NeurIPS',
            'reference': 'Grill et al., Bootstrap Your Own Latent',
            'momentum': self.momentum
        })
        return info

