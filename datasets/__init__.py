"""
TCCL数据集模块

支持的数据集：
- MFPTDataset: MFPT轴承故障数据集 (3类)
- CWRUDataset: CWRU轴承故障数据集 (4类)
- SEUDataset: SEU轴承故障数据集 (4类)

使用方法：
    from datasets import MFPTDataset
    
    # 主实验（无监督聚类）
    dataset = MFPTDataset(root_dir='./data', mode='full')
    
    # 消融实验（训练/测试）
    train_dataset = MFPTDataset(root_dir='./train', mode='train')
    stats = train_dataset.get_normalization_stats()
    
    test_dataset = MFPTDataset(root_dir='./test', mode='test',
                               normalization_stats=stats)

更多信息请参考 datasets/README.md
"""

from .MFPT import MFPTDataset
from .CWRU import CWRUDataset
from .SEU import SEUDataset

__all__ = ['MFPTDataset', 'CWRUDataset', 'SEUDataset']
__version__ = '2.0.0'

