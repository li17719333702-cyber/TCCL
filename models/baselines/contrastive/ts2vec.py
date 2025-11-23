"""
TS2Vec (Towards Universal Representation of Time Series)

论文: Yue et al., "TS2Vec: Towards Universal Representation of Time Series", AAAI 2022

核心思想：
- 时序专用的分层对比学习
- 时间戳级别和实例级别对比
- 上下文一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel


class TS2VecModel(BaseModel):
    """
    TS2Vec (AAAI 2022)
    
    特点：
    - 专为时序设计
    - 分层对比学习
    - 时间和实例级表示
    
    论文链接：https://arxiv.org/abs/2106.10466
    """
    
    def __init__(self, feature_extractor: nn.Module,
                 projection_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.model_type = 'contrastive'
        self.feature_extractor = feature_extractor
        
        # 时序投影头
        self.temporal_head = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, projection_dim, kernel_size=1)
        )
        
        # 实例级投影头
        self.instance_head = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
    
    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        feat1 = self.feature_extractor(view1)
        feat2 = self.feature_extractor(view2)
        
        feat1_global = F.adaptive_avg_pool1d(feat1, 1).squeeze(-1)
        feat2_global = F.adaptive_avg_pool1d(feat2, 1).squeeze(-1)
        
        inst_proj1 = self.instance_head(feat1_global)
        inst_proj2 = self.instance_head(feat2_global)
        
        inst_proj1_norm = F.normalize(inst_proj1, dim=1)
        inst_proj2_norm = F.normalize(inst_proj2, dim=1)
        
        sim_matrix = inst_proj1_norm @ inst_proj2_norm.T
        return sim_matrix
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取骨干网络的全局特征（未归一化，64维）。"""
        feat = self.feature_extractor(x)
        feat_global = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        return feat_global.detach().cpu().numpy()
    
    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            'description': 'Time series specific hierarchical contrastive learning',
            'year': 2022,
            'conference': 'AAAI',
            'reference': 'Yue et al., TS2Vec: Towards Universal Representation of Time Series'
        })
        return info

