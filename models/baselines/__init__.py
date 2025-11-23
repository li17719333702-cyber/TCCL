"""
基线方法模块（模块化版本）

包含用于对比实验的各种基线方法，每个方法独立文件：

传统方法 (Traditional Methods):
- Raw+UMAP: 原始信号 + UMAP降维
- Handcrafted Features: 19维手工时频域特征
- PCA+K-Means: PCA降维 + K-Means

深度聚类 (Deep Clustering):
- DEC: Deep Embedded Clustering (ICML 2016)
- JULE: Joint Unsupervised Learning (CVPR 2016)
- SCAN: Semantic Clustering by Adopting Nearest neighbors (ECCV 2020)

对比学习-经典 (Contrastive Learning - Classic 2020):
- SimCLR: Simple Framework for Contrastive Learning (ICML 2020)
- MoCo: Momentum Contrast (CVPR 2020)
- BYOL: Bootstrap Your Own Latent (NeurIPS 2020)

对比学习-SOTA (Contrastive Learning - SOTA 2021-2023):
- SimSiam: Exploring Simple Siamese Representation Learning (CVPR 2021)
- TS2Vec: Towards Universal Representation of Time Series (AAAI 2022)
- VICReg: Variance-Invariance-Covariance Regularization (ICLR 2022)
- TimesNet: Temporal 2D-Variation Modeling (ICLR 2023)
"""

# 基础组件
from .base import BaseModel
from .common import FeatureExtractor, extract_handcrafted_features

# 传统方法
from .traditional import (
    RawUMAPModel,
    HandcraftedFeaturesModel,
    PCAKMeansModel
)

# 深度聚类
from .deep_clustering import (
    DECModel,
    JULEModel,
    SCANModel
)

# 对比学习
from .contrastive import (
    # 经典 (2020)
    SimCLRModel,
    MoCoModel,
    BYOLModel,
    # SOTA (2021-2023)
    SimSiamModel,
    TS2VecModel,
    VICRegModel,
    TimesNetModel,
    TCCLModel,
)

__all__ = [
    # Base
    'BaseModel',
    'FeatureExtractor',
    'extract_handcrafted_features',
    
    # Traditional
    'RawUMAPModel',
    'HandcraftedFeaturesModel',
    'PCAKMeansModel',
    
    # Deep Clustering
    'DECModel',
    'JULEModel',
    'SCANModel',
    
    # Contrastive Learning (Classic)
    'SimCLRModel',
    'MoCoModel',
    'BYOLModel',
    
    # Contrastive Learning (SOTA)
    'SimSiamModel',
    'TS2VecModel',
    'VICRegModel',
    'TimesNetModel',
    'TCCLModel',
]

# 方法分类字典（方便基准测试使用）
METHOD_CATEGORIES = {
    'Traditional Methods': [
        ('Raw+UMAP', RawUMAPModel),
        ('Handcrafted Features', HandcraftedFeaturesModel),
        ('PCA+K-Means', PCAKMeansModel),
    ],
    'Deep Clustering': [
        ('DEC', DECModel),
        ('JULE', JULEModel),
        ('SCAN', SCANModel),
    ],
    'Contrastive Learning (2020)': [
        ('SimCLR', SimCLRModel),
        ('MoCo', MoCoModel),
        ('BYOL', BYOLModel),
    ],
    'Contrastive Learning (SOTA 2021-2023)': [
        ('SimSiam', SimSiamModel),
        ('TS2Vec', TS2VecModel),
        ('VICReg', VICRegModel),
        ('TimesNet', TimesNetModel),
    ],
}
