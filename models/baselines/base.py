"""
基线模型的基类

定义了所有基线模型的统一接口
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseModel(nn.Module, ABC):
    """
    所有基线模型的基类
    
    提供统一的接口用于：
    - 训练
    - 特征提取
    - 损失计算
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.needs_training = True  # 是否需要训练
        self.model_type = 'base'  # 模型类型标识
        self.kwargs = kwargs
    
    @abstractmethod
    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            view1: 第一个视图 [B, C, L]
            view2: 第二个视图 [B, C, L]
        
        Returns:
            输出（具体取决于模型类型）
            - 对比学习：相似度矩阵 [B, B]
            - 深度聚类：损失值（标量）
            - 传统方法：0（不需要训练）
        """
        pass
    
    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征向量
        
        Args:
            x: 输入信号 [B, C, L]
        
        Returns:
            特征向量 [B, D]
        """
        pass
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            包含模型信息的字典
        """
        return {
            'name': self.__class__.__name__,
            'type': self.model_type,
            'needs_training': self.needs_training,
            'num_parameters': sum(p.numel() for p in self.parameters()),
        }



