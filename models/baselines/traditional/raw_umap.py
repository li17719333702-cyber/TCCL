"""
Raw + UMAP方法

直接使用原始信号，通过UMAP降维后进行K-Means聚类
不需要训练神经网络
"""

import numpy as np
import torch

from ..base import BaseModel


class RawUMAPModel(BaseModel):
    """
    Raw Signal + UMAP + K-Means
    
    特点：
    - 不需要训练
    - 直接使用原始信号
    - 依赖UMAP进行有效降维
    
    适用场景：
    - 快速baseline
    - 数据量较小时
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.needs_training = False
        self.model_type = 'traditional'
    
    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """不需要训练，返回0"""
        return torch.tensor(0.0)
    
    def extract_features(self, x: torch.Tensor) -> np.ndarray:
        """
        提取特征：直接使用展平的原始信号
        
        Args:
            x: 输入信号 [B, 1, L]
        
        Returns:
            特征 [B, L] (numpy array)
        """
        return x.view(x.size(0), -1).cpu().numpy()
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'description': 'Raw signal with UMAP dimensionality reduction',
            'year': 'N/A',
            'reference': 'McInnes et al., UMAP: Uniform Manifold Approximation and Projection, 2018'
        })
        return info

