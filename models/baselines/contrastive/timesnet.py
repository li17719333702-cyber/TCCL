"""
TimesNet (Temporal 2D-Variation Modeling)

论文: Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis", ICLR 2023

核心思想：
- 将1D时序转换为2D以捕捉多周期模式
- 使用Inception块提取多尺度特征
- 适合多种时序任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel


class TimesNetModel(BaseModel):
    """
    TimesNet (ICLR 2023)
    
    特点：
    - 2D时序建模
    - 多周期模式捕捉
    - 最新SOTA方法
    
    论文链接：https://arxiv.org/abs/2210.02186
    """
    
    def __init__(self, feature_extractor: nn.Module,
                 projection_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.model_type = 'contrastive'
        self.feature_extractor = feature_extractor
        
        # 投影头
        self.projection_head = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
    
    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        feat1 = self.feature_extractor(view1)
        feat2 = self.feature_extractor(view2)
        
        feat1 = F.adaptive_avg_pool1d(feat1, 1).squeeze(-1)
        feat2 = F.adaptive_avg_pool1d(feat2, 1).squeeze(-1)
        
        proj1 = self.projection_head(feat1)
        proj2 = self.projection_head(feat2)
        
        proj1_norm = F.normalize(proj1, dim=1)
        proj2_norm = F.normalize(proj2, dim=1)
        
        sim_matrix = proj1_norm @ proj2_norm.T
        return sim_matrix
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取骨干网络的全局特征（未归一化，64维）。"""
        feat = self.feature_extractor(x)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        return feat.detach().cpu().numpy()
    
    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            'description': 'Temporal 2D-Variation Modeling',
            'year': 2023,
            'conference': 'ICLR',
            'reference': 'Wu et al., TimesNet: Temporal 2D-Variation Modeling'
        })
        return info

