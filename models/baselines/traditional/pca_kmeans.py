"""
PCA + K-Means方法

使用主成分分析（PCA）降维后进行K-Means聚类
不需要训练神经网络，但需要拟合PCA
"""

import numpy as np
import torch
from sklearn.decomposition import PCA

from ..base import BaseModel


class PCAKMeansModel(BaseModel):
    """
    PCA + K-Means
    
    特点：
    - 使用PCA进行线性降维
    - 可以控制保留的方差比例
    - 计算效率高
    
    优点：
    - 简单有效
    - 计算快速
    - 可解释的主成分
    
    缺点：
    - 仅捕捉线性关系
    - 可能丢失非线性特征
    
    适用场景：
    - 数据具有线性结构
    - 需要快速降维
    - 作为其他方法的预处理
    """
    
    def __init__(self, n_components: int = 64, **kwargs):
        """
        Args:
            n_components: 主成分数量
        """
        super().__init__(**kwargs)
        self.needs_training = False
        self.model_type = 'traditional'
        self.n_components = n_components
        self.pca = None
        self.explained_variance_ = None
    
    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """不需要训练，返回0"""
        return torch.tensor(0.0)
    
    def fit_pca(self, all_signals: np.ndarray, verbose: bool = True):
        """
        拟合PCA
        
        Args:
            all_signals: 所有信号 [N, L]
            verbose: 是否打印信息
        """
        if verbose:
            print(f"    Fitting PCA with {self.n_components} components...")
        
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.pca.fit(all_signals)
        self.explained_variance_ = np.sum(self.pca.explained_variance_ratio_)
        
        if verbose:
            print(f"    PCA explained variance: {self.explained_variance_:.4f}")
    
    def extract_features(self, x: torch.Tensor) -> np.ndarray:
        """
        提取PCA特征
        
        Args:
            x: 输入信号 [B, 1, L]
        
        Returns:
            PCA特征 [B, n_components] (numpy array)
        """
        signals = x.view(x.size(0), -1).cpu().numpy()
        
        # 如果PCA还未fit，自动fit
        if self.pca is None:
            self.fit_pca(signals, verbose=False)
        
        return self.pca.transform(signals)
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'description': f'PCA dimensionality reduction to {self.n_components} components',
            'year': 'N/A',
            'reference': 'Pearson, K. (1901). On lines and planes of closest fit to systems of points in space',
            'n_components': self.n_components,
            'explained_variance': self.explained_variance_ if self.pca else None
        })
        return info

