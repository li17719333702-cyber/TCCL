"""
Handcrafted Features方法

提取19维手工时频域特征进行聚类
不需要训练神经网络
"""

import numpy as np
import torch

from ..base import BaseModel
from ..common import extract_handcrafted_features


class HandcraftedFeaturesModel(BaseModel):
    """
    Handcrafted Features + K-Means
    
    特点：
    - 提取19维手工特征（10个时域 + 9个频域）
    - 基于领域知识设计的特征
    - 可解释性强
    
    特征列表：
    时域特征（10个）：
    - 均值、标准差、均方根、峰值
    - 偏度、峭度
    - 波形因子、峭度因子、脉冲因子、裕度因子
    
    频域特征（9个）：
    - 频域均值、频率中心、频域均方根
    - 频域标准差、频域平均频率
    - 稳定因子、变异系数
    - 频域偏度、频域峭度
    
    适用场景：
    - 需要可解释性
    - 小样本学习
    - 传统振动分析
    """
    
    def __init__(self, fs: float = 12000, **kwargs):
        """
        Args:
            fs: 采样频率
        """
        super().__init__(**kwargs)
        self.needs_training = False
        self.model_type = 'traditional'
        self.fs = fs
    
    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """不需要训练，返回0"""
        return torch.tensor(0.0)
    
    def extract_features(self, x: torch.Tensor) -> np.ndarray:
        """
        提取手工特征
        
        Args:
            x: 输入信号 [B, 1, L]
        
        Returns:
            手工特征 [B, 19] (numpy array)
        """
        batch_size = x.size(0)
        features_list = []
        
        for i in range(batch_size):
            signal = x[i, 0].cpu().numpy()  # 取出单个信号
            features = extract_handcrafted_features(signal, fs=self.fs)
            features_list.append(features)
        
        return np.array(features_list)
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'description': '19-dimensional handcrafted time-frequency features',
            'year': 'N/A',
            'reference': 'Traditional signal processing features',
            'feature_dim': 19,
            'sampling_frequency': self.fs
        })
        return info

