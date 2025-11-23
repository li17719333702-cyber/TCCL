"""
对比学习方法

包括经典对比学习方法（2020）和最新SOTA方法（2021-2023）
"""

# 经典方法 (2020)
from .simclr import SimCLRModel
from .moco import MoCoModel
from .byol import BYOLModel

# SOTA方法 (2021-2023)
from .simsiam import SimSiamModel
from .ts2vec import TS2VecModel
from .vicreg import VICRegModel
from .timesnet import TimesNetModel
from .tccl import TCCLModel

__all__ = [
    # Classic (2020)
    'SimCLRModel',
    'MoCoModel',
    'BYOLModel',
    # SOTA (2021-2023)
    'SimSiamModel',
    'TS2VecModel',
    'VICRegModel',
    'TimesNetModel',
    'TCCLModel',
]

