"""
Single-File TCCL Implementation
完整的TCCL实现，包含数据集加载、模型定义、训练和可视化
不依赖其他项目文件，可独立运行

Usage:
    # 使用默认参数运行三个数据集
    python single_tccl.py
    
    # 自定义随机种子和训练参数
    python single_tccl.py --seed 123 --epochs 100 --batch_size 64 --lr 0.0001
    
    # 自定义数据集路径
    python single_tccl.py --cwru_path /path/to/cwru --seu_path /path/to/seu --mfpt_path /path/to/mfpt
    
    # 查看所有可用参数
    python single_tccl.py --help
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score, confusion_matrix
from sklearn.manifold import TSNE
import umap
from scipy.io import loadmat
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 随机种子设置
# ============================================================================

def set_random_seed(seed=42):
    """
    设置所有随机数生成器的种子，确保实验可复现
    
    Args:
        seed: 随机种子值，默认为42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量以确保完全的确定性
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ 随机种子已设置为: {seed}")


# ============================================================================
# 数据集加载类
# ============================================================================

class CWRUDataset(Dataset):
    """CWRU轴承数据集"""
    
    def __init__(self, data_root, window_size=1024, stride=512, split='full', augment=False, verbose=True):
        """
        Args:
            data_root: 数据根目录
            window_size: 窗口大小
            stride: 滑动步长
            split: 'train', 'test', 'full'
            augment: 是否使用数据增强
            verbose: 是否输出详细加载信息
        """
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.verbose = verbose
        
        # 加载数据
        self.data, self.labels = self._load_data(data_root)
        
        # 归一化
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.data = (self.data - self.mean) / (self.std + 1e-8)
        
        if verbose:
            print(f"[CWRU] 加载完成！共 {len(self.data)} 个样本")
            print(f"[CWRU] 各类别样本数: {self._count_labels()}")
    
    def _load_data(self, data_root):
        """加载CWRU数据集"""
        data_list = []
        label_list = []
        
        label_map = {
            'normal': 0,
            'ball': 1,
            'inner_race': 2,
            'outer_race': 3
        }
        
        # 扫描所有.mat文件
        mat_files = [f for f in os.listdir(data_root) if f.endswith('.mat')]
        
        if self.verbose:
            print(f"[CWRU] 发现 {len(mat_files)} 个.mat文件")
        
        for mat_file in sorted(mat_files):
            file_path = os.path.join(data_root, mat_file)
            
            # 根据文件名判断类别 - CWRU数据集的命名规则
            # 正常: 97.mat, 98.mat, 99.mat, 100.mat 等（纯数字）
            # 滚珠故障: B007_0.mat, B014_1.mat, B021_2.mat 等（以B开头）
            # 内圈故障: IR007_0.mat, IR014_1.mat, IR021_2.mat 等（以IR开头）
            # 外圈故障: OR007@6_0.mat, OR014@6_1.mat 等（以OR开头）
            
            fname_upper = mat_file.upper()
            
            if fname_upper.startswith('IR'):
                label = label_map['inner_race']
            elif fname_upper.startswith('OR'):
                label = label_map['outer_race']
            elif fname_upper.startswith('B') and not fname_upper.startswith('BA'):
                # B开头但不是BA（BA是baseline accelerometer的缩写）
                label = label_map['ball']
            elif any(x in fname_upper for x in ['NORMAL', 'BASELINE']) or mat_file[0].isdigit():
                # 包含normal/baseline关键字，或者文件名以数字开头
                label = label_map['normal']
            else:
                print(f"  跳过无法识别的文件: {mat_file}")
                continue
            
            # 加载.mat文件
            try:
                mat_data = loadmat(file_path)
                # CWRU数据集中，振动信号通常在DE_time, FE_time等键中
                signal_loaded = False
                for key in mat_data.keys():
                    if 'DE_time' in key or 'FE_time' in key or 'BA_time' in key:
                        signal = mat_data[key].flatten()
                        
                        # 滑动窗口切片
                        num_windows = 0
                        for i in range(0, len(signal) - self.window_size, self.stride):
                            window = signal[i:i + self.window_size]
                            data_list.append(window)
                            label_list.append(label)
                            num_windows += 1
                        
                        label_name = [k for k, v in label_map.items() if v == label][0]
                        if self.verbose:
                            print(f"  ✓ {mat_file:30s} -> {label_name:12s} ({num_windows:4d} windows)")
                        signal_loaded = True
                        break
                
                if not signal_loaded and self.verbose:
                    print(f"  ✗ {mat_file}: 未找到振动信号键")
                    
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ {mat_file}: 加载失败 - {e}")
                continue
        
        return np.array(data_list, dtype=np.float32), np.array(label_list, dtype=np.int64)
    
    def _count_labels(self):
        """统计各类别样本数"""
        unique, counts = np.unique(self.labels, return_counts=True)
        label_names = {0: 'normal', 1: 'ball', 2: 'inner_race', 3: 'outer_race'}
        return {label_names[k]: v for k, v in zip(unique, counts)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]
        
        if self.augment:
            # 数据增强
            x1 = self._augment(x)
            x2 = self._augment(x)
            return torch.FloatTensor(x1).unsqueeze(0), torch.FloatTensor(x2).unsqueeze(0), label
        else:
            return torch.FloatTensor(x).unsqueeze(0), label
    
    def _augment(self, x):
        """数据增强"""
        # 幅值缩放
        scale = np.random.uniform(0.9, 1.1)
        x = x * scale
        
        # 高斯噪声
        noise = np.random.normal(0, 0.05, x.shape)
        x = x + noise
        
        return x


class SEUDataset(Dataset):
    """SEU轴承数据集"""
    
    def __init__(self, data_root, window_size=1024, stride=512, split='full', augment=False, verbose=True):
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.verbose = verbose
        
        self.data, self.labels = self._load_data(data_root)
        
        # 归一化
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.data = (self.data - self.mean) / (self.std + 1e-8)
        
        if verbose:
            print(f"[SEU] 加载完成！共 {len(self.data)} 个样本")
            print(f"[SEU] 各类别样本数: {self._count_labels()}")
    
    def _load_data(self, data_root):
        """加载SEU数据集"""
        data_list = []
        label_list = []
        
        label_map = {
            'normal': 0,
            'ball': 1,
            'inner_race': 2,
            'outer_race': 3
        }
        
        mat_files = [f for f in os.listdir(data_root) if f.endswith('.mat')]
        if self.verbose:
            print(f"\n[SEU] 找到 {len(mat_files)} 个.mat文件")
        
        for mat_file in sorted(mat_files):
            file_path = os.path.join(data_root, mat_file)
            
            # 根据文件名判断类别 - SEU数据集的命名规则
            # SEU使用简写：normal, ir (内圈), or (外圈), b (滚珠)
            # 文件名示例：normal_0.mat, ir007.mat, or007@6.mat, b007.mat
            fname_lower = mat_file.lower()
            fname_base = mat_file.replace('.mat', '').lower()
            
            # 优先级判断（关键词匹配顺序很重要）
            if 'normal' in fname_lower:
                label = label_map['normal']
                label_name = 'normal'
            elif fname_base.startswith('ir') or '_ir' in fname_lower:
                # IR开头或包含_ir的是内圈故障
                label = label_map['inner_race']
                label_name = 'inner_race'
            elif fname_base.startswith('or') or '_or' in fname_lower:
                # OR开头或包含_or的是外圈故障
                label = label_map['outer_race']
                label_name = 'outer_race'
            elif fname_base.startswith('b') or '_b' in fname_lower or 'ball' in fname_lower:
                # B开头或包含_b或ball的是滚珠故障
                label = label_map['ball']
                label_name = 'ball'
            else:
                if self.verbose:
                    print(f"  [SEU] 跳过未识别的文件: {mat_file}")
                continue
            
            if self.verbose:
                print(f"  [SEU] {mat_file} -> {label_name}")
            
            try:
                mat_data = loadmat(file_path)
                # SEU数据集的信号存储在以 _DE_time 或 _FE_time 结尾的键中
                signal = None
                
                # 优先查找驱动端信号 (DE = Drive End)
                for key in mat_data.keys():
                    if key.endswith('_DE_time'):
                        signal = mat_data[key].flatten()
                        break
                
                # 如果找不到，查找风扇端信号 (FE = Fan End)
                if signal is None:
                    for key in mat_data.keys():
                        if key.endswith('_FE_time'):
                            signal = mat_data[key].flatten()
                            break
                
                # 最后的备选方案：查找任何非元数据键
                if signal is None:
                    for key in mat_data.keys():
                        if not key.startswith('__'):
                            signal = mat_data[key].flatten()
                            break
                
                if signal is not None and len(signal) > self.window_size:
                    n_windows_before = len(data_list)
                    for i in range(0, len(signal) - self.window_size, self.stride):
                        window = signal[i:i + self.window_size]
                        data_list.append(window)
                        label_list.append(label)
                    if self.verbose:
                        n_windows_added = len(data_list) - n_windows_before
                        print(f"    -> 生成 {n_windows_added} 个窗口")
                else:
                    if self.verbose:
                        print(f"    -> 信号长度不足 ({len(signal) if signal is not None else 0})")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  加载失败: {mat_file} - {e}")
                continue
        
        return np.array(data_list, dtype=np.float32), np.array(label_list, dtype=np.int64)
    
    def _count_labels(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        label_names = {0: 'normal', 1: 'ball', 2: 'inner_race', 3: 'outer_race'}
        return {label_names[k]: v for k, v in zip(unique, counts)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]
        
        if self.augment:
            x1 = self._augment(x)
            x2 = self._augment(x)
            return torch.FloatTensor(x1).unsqueeze(0), torch.FloatTensor(x2).unsqueeze(0), label
        else:
            return torch.FloatTensor(x).unsqueeze(0), label
    
    def _augment(self, x):
        scale = np.random.uniform(0.9, 1.1)
        x = x * scale
        noise = np.random.normal(0, 0.05, x.shape)
        x = x + noise
        return x


class MFPTDataset(Dataset):
    """MFPT轴承数据集"""
    
    def __init__(self, data_root, window_size=1024, stride=512, split='full', augment=False, verbose=True):
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.verbose = verbose
        
        self.data, self.labels = self._load_data(data_root)
        
        # 归一化
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.data = (self.data - self.mean) / (self.std + 1e-8)
        
        if verbose:
            print(f"[MFPT] 加载完成！共 {len(self.data)} 个样本")
            print(f"[MFPT] 各类别样本数: {self._count_labels()}")
    
    def _load_data(self, data_root):
        """加载MFPT数据集"""
        data_list = []
        label_list = []
        
        label_map = {
            'normal': 0,
            'inner_race': 1,
            'outer_race': 2
        }
        
        mat_files = [f for f in os.listdir(data_root) if f.endswith('.mat')]
        if self.verbose:
            print(f"\n[MFPT] 找到 {len(mat_files)} 个.mat文件")
        
        for mat_file in sorted(mat_files):
            file_path = os.path.join(data_root, mat_file)
            
            # MFPT数据集文件名规则：
            # baseline*.mat → normal
            # *innerracefault*.mat 或 *inner*.mat → inner_race
            # *outerracefault*.mat 或 *outer*.mat → outer_race
            fname_lower = mat_file.lower()
            
            if 'baseline' in fname_lower:
                label = label_map['normal']
                label_name = 'normal'
            elif 'innerracefault' in fname_lower or ('inner' in fname_lower and 'race' in fname_lower):
                label = label_map['inner_race']
                label_name = 'inner_race'
            elif 'outerracefault' in fname_lower or ('outer' in fname_lower and 'race' in fname_lower):
                label = label_map['outer_race']
                label_name = 'outer_race'
            elif 'inner' in fname_lower:
                label = label_map['inner_race']
                label_name = 'inner_race'
            elif 'outer' in fname_lower:
                label = label_map['outer_race']
                label_name = 'outer_race'
            else:
                if self.verbose:
                    print(f"  [MFPT] 跳过未识别的文件: {mat_file}")
                continue
            
            if self.verbose:
                print(f"  [MFPT] {mat_file} -> {label_name}")
            
            try:
                mat_data = loadmat(file_path)
                signal = None
                
                # MFPT数据集特定的数据结构: mat_data['bearing']['gs'][0, 0]
                try:
                    signal = mat_data['bearing']['gs'][0, 0].flatten()
                except:
                    # 如果标准结构不存在，尝试其他键
                    for key in ['bearing', 'gs', 'sr']:
                        if key in mat_data:
                            if isinstance(mat_data[key], np.ndarray):
                                if mat_data[key].dtype.names:  # 结构体
                                    for field in mat_data[key].dtype.names:
                                        if 'gs' in field or 'sr' in field or 'vibration' in field:
                                            signal = mat_data[key][field][0][0].flatten()
                                            break
                                else:
                                    signal = mat_data[key].flatten()
                                break
                    
                    # 最后的备选方案
                    if signal is None:
                        for key in mat_data.keys():
                            if not key.startswith('__'):
                                try:
                                    signal = mat_data[key].flatten()
                                    break
                                except:
                                    continue
                
                if signal is not None and len(signal) > self.window_size:
                    n_windows_before = len(data_list)
                    for i in range(0, len(signal) - self.window_size, self.stride):
                        window = signal[i:i + self.window_size]
                        data_list.append(window)
                        label_list.append(label)
                    if self.verbose:
                        n_windows_added = len(data_list) - n_windows_before
                        print(f"    -> 生成 {n_windows_added} 个窗口")
                else:
                    if self.verbose:
                        print(f"    -> 信号长度不足 ({len(signal) if signal is not None else 0})")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  加载失败: {mat_file} - {e}")
                continue
        
        return np.array(data_list, dtype=np.float32), np.array(label_list, dtype=np.int64)
    
    def _count_labels(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        label_names = {0: 'normal', 1: 'inner_race', 2: 'outer_race'}
        return {label_names[k]: v for k, v in zip(unique, counts)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]
        
        if self.augment:
            x1 = self._augment(x)
            x2 = self._augment(x)
            return torch.FloatTensor(x1).unsqueeze(0), torch.FloatTensor(x2).unsqueeze(0), label
        else:
            return torch.FloatTensor(x).unsqueeze(0), label
    
    def _augment(self, x):
        scale = np.random.uniform(0.9, 1.1)
        x = x * scale
        noise = np.random.normal(0, 0.05, x.shape)
        x = x + noise
        return x


# ============================================================================
# TCCL模型定义
# ============================================================================

class FeatureEncoder(nn.Module):
    """特征编码器 - 3层1D-CNN"""
    
    def __init__(self, in_channels=1, feature_dim=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Block 1: 1 -> 32, 1024 -> 256
            nn.Conv1d(in_channels, 16, kernel_size=17, stride=1, padding=8),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # Block 2: 32 -> 64, 256 -> 64
            nn.Conv1d(16, 32, kernel_size=17, stride=1, padding=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # Block 3: 64 -> 128, 64 -> 16
            nn.Conv1d(32, feature_dim, kernel_size=17, stride=1, padding=8),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, L) - 输入信号
        Returns:
            h: (B, C, W) - 特征图，C=64, W=16
        """
        return self.encoder(x)

class TemplateHead(nn.Module):
    """模板头 - 生成模板核（与benchmark.py完全一致）"""
    
    def __init__(self, in_channels=64, kernel_width=5):
        super().__init__()
        self.kernel_width = kernel_width
        
        # 与benchmark中TCCLModel的template_head完全一致
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(kernel_width)
        )
    
    def forward(self, h):
        """
        Args:
            h: (B, C, W) - 特征图
        Returns:
            k: (B, C, W_k) - 模板核，W_k=8
        """
        return self.head(h)


class TCCLModel(nn.Module):
    """完整的TCCL模型（与benchmark.py完全一致）"""
    
    def __init__(self, feature_dim=64, kernel_width=5, temperature=0.1, template_lambda=0.5):
        super().__init__()
        
        self.encoder = FeatureEncoder(in_channels=1, feature_dim=feature_dim)
        self.template_head = TemplateHead(in_channels=feature_dim, kernel_width=kernel_width)
        self.temperature = temperature
        self.template_lambda = template_lambda  # 模板对比损失权重
    
    def forward(self, x1, x2):
        """
        Args:
            x1, x2: (B, 1, L) - 两个增强视图
        Returns:
            loss: 对比学习损失
        """
        # 提取特征
        h1 = self.encoder(x1)  # (B, C, W)
        h2 = self.encoder(x2)  # (B, C, W)
        
        # 生成模板核
        k1 = self.template_head(h1)  # (B, C, W_k)
        k2 = self.template_head(h2)  # (B, C, W_k)
        
        # 计算相似度矩阵
        sim_1to2 = self._template_similarity(k1, h2)  # (B, B)
        sim_2to1 = self._template_similarity(k2, h1)  # (B, B)
        
        # 对称的InfoNCE损失（特征-模板匹配损失）
        loss_1to2 = self._infonce_loss(sim_1to2)
        loss_2to1 = self._infonce_loss(sim_2to1)
        loss_matching = (loss_1to2 + loss_2to1) / 2
        
        # 模板对比损失（增强模板判别性）
        loss_template = self._template_contrastive_loss(k1, k2)
        
        # 总损失
        loss = loss_matching + self.template_lambda * loss_template
        
        return loss
    
    def _template_similarity(self, k, h):
        """
        计算模板核与特征图的滑动余弦相似度 (Sliding Cosine Similarity)
        通过在卷积前对模板核和特征图进行L2归一化，将标准互相关转换为余弦相似度
        
        Args:
            k: (B, C, W_k) - 模板核
            h: (B, C, W) - 特征图
        Returns:
            sim: (B, B) - 相似度矩阵
        """
        B, C, W_k = k.shape
        _, _, W = h.shape
        
        # 扩展模板和特征图以计算所有配对
        templates_exp = k.unsqueeze(1).expand(-1, B, -1, -1)  # (B, B, C, W_k)
        templates_exp = templates_exp.reshape(B * B, C, W_k)
        
        features_exp = h.unsqueeze(0).expand(B, -1, -1, -1)  # (B, B, C, W)
        features_exp = features_exp.reshape(B * B, C, W)
        
        # 重塑为分组卷积格式
        templates_conv = templates_exp.reshape(B * B * C, 1, W_k)
        features_conv = features_exp.reshape(1, B * B * C, W)
        
        # 零中心化 + L2归一化：将标准互相关转换为滑动余弦相似度
        # 先零中心化（减去均值），让数据包含正负值
        templates_conv = templates_conv - templates_conv.mean(dim=-1, keepdim=True)
        features_conv = features_conv - features_conv.mean(dim=-1, keepdim=True)
        
        # 再进行L2归一化
        # 对模板核进行L2归一化 (在最后一个维度，即 W_k 维度)
        templates_conv = F.normalize(templates_conv, p=2, dim=-1)
        # 对特征图进行L2归一化 (在最后一个维度，即 W 维度)
        features_conv = F.normalize(features_conv, p=2, dim=-1)
        
        # 分组卷积：现在计算的是余弦相似度而非互相关
        # Output = Σ(h/|h| · k/|k|)，等价于 cos(θ)
        conv_out = F.conv1d(features_conv, templates_conv, groups=B * B * C, padding='same')
        conv_out = conv_out.view(B * B, C, W)
        
        # 沿通道和时间维度聚合
        response = torch.sum(conv_out, dim=1)  # (B*B, W)
        
        # 全局平均池化
        similarity = F.adaptive_avg_pool1d(response.unsqueeze(1), 1).squeeze()
        
        return similarity.view(B, B)
    
    def _infonce_loss(self, sim_matrix):
        """
        InfoNCE损失（NT-Xent）- 使用数值稳定的实现
        Args:
            sim_matrix: (B, B) - 相似度矩阵
        Returns:
            loss: 标量损失
        """
        B = sim_matrix.shape[0]
        
        # 检查相似度矩阵是否包含异常值
        if torch.isnan(sim_matrix).any() or torch.isinf(sim_matrix).any():
            print(f"警告: 相似度矩阵包含异常值! nan: {torch.isnan(sim_matrix).sum()}, inf: {torch.isinf(sim_matrix).sum()}")
            # 替换异常值为0
            sim_matrix = torch.nan_to_num(sim_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 缩放相似度（温度参数）
        logits = sim_matrix / self.temperature
        
        # 正样本标签（对角线）
        labels = torch.arange(B, device=sim_matrix.device)
        
        # 使用PyTorch的数值稳定cross_entropy
        # cross_entropy内部使用log_softmax，避免了exp溢出
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def _template_contrastive_loss(self, k1, k2):
        """
        模板对比损失：让正样本对的模板相似，负样本对的模板远离
        通过在模板空间施加对比约束，增强模板的判别性
        
        Args:
            k1, k2: (B, C, W_k) - 同一样本两个增强视图的模板核
        Returns:
            loss: 标量损失
        """
        B, C, W_k = k1.shape
        
        # 展平模板核：(B, C*W_k)
        k1_flat = k1.view(B, -1)
        k2_flat = k2.view(B, -1)
        
        # L2归一化
        k1_norm = F.normalize(k1_flat, p=2, dim=1)
        k2_norm = F.normalize(k2_flat, p=2, dim=1)
        
        # 计算模板相似度矩阵：(B, B)
        # k1_norm @ k2_norm.T = cosine similarity
        template_sim = torch.matmul(k1_norm, k2_norm.T)
        
        # 使用InfoNCE损失：对角线是正样本对（同一样本的两个视图）
        # 非对角线是负样本对（不同样本）
        logits = template_sim / self.temperature
        labels = torch.arange(B, device=k1.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def extract_features(self, x):
        """提取特征用于评估"""
        h = self.encoder(x)  # (B, C, W)
        # 全局平均池化
        # z = h.mean(dim=2)  # (B, C)
        # 展平操作
        z = h.view(h.size(0), -1)  # (B, C*W)
        # 归一化操作
        # z = z / torch.norm(z, dim=1, keepdim=True)
        return z


# ============================================================================
# 训练和评估函数
# ============================================================================

def train_tccl(model, train_loader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for x1, x2, _ in pbar:
        x1, x2 = x1.to(device), x2.to(device)
        
        optimizer.zero_grad()
        loss = model(x1, x2)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def extract_all_features(model, data_loader, device):
    """提取所有样本的特征"""
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for x, labels in tqdm(data_loader, desc='Extracting features'):
            x = x.to(device)
            features = model.extract_features(x)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    return features, labels


def evaluate_clustering(features, labels):
    """
    使用DBSCAN进行聚类评估（自动确定聚类数）
    直接在2维UMAP空间进行聚类，保证可视化和评价指标一致
    
    Args:
        features: 特征矩阵
        labels: 真实标签
    """
    n_true_classes = len(np.unique(labels))
    
    # 直接降维到2维，保证可视化和评价指标一致
    print(f"  UMAP dimensionality reduction to 2D (True classes={n_true_classes})...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    features_2d = reducer.fit_transform(features)
    
    # 自动估计eps参数（使用改进的k-distance方法）
    print(f"  Estimating eps parameter...")
    
    # 调整min_samples：样本越多，可以更大
    n_samples = len(features_2d)
    min_samples = max(10, min(n_samples // 100, n_true_classes * 5))
    
    # 计算k-距离
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(features_2d)
    distances, _ = nbrs.kneighbors(features_2d)
    k_distances = np.sort(distances[:, -1])
    
    # 使用肘部法则（elbow）自动选择eps
    # 找到k-distance曲线中斜率变化最大的点
    # 使用95-98分位数之间的值，避免过度碎片化
    eps_candidates = [
        np.percentile(k_distances, 95),
        np.percentile(k_distances, 96),
        np.percentile(k_distances, 97),
        np.percentile(k_distances, 98)
    ]
    
    # 选择中位数作为eps（更稳健）
    eps = np.median(eps_candidates)
    
    print(f"  Running DBSCAN on 2D space (eps={eps:.4f}, min_samples={min_samples})...")
    
    # DBSCAN聚类（在2维空间），带自适应调整
    max_attempts = 3
    for attempt in range(max_attempts):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        pred_labels = dbscan.fit_predict(features_2d)
        
        # 统计聚类结果
        n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
        n_noise = list(pred_labels).count(-1)
        
        print(f"  → Attempt {attempt+1}: {n_clusters} clusters, {n_noise} noise points ({n_noise/len(pred_labels)*100:.1f}%)")
        
        # 检查聚类数是否合理（不超过真实类别数的3倍）
        if n_clusters <= n_true_classes * 3:
            print(f"  ✓ Cluster count is reasonable (≤ {n_true_classes * 3})")
            break
        elif attempt < max_attempts - 1:
            # 聚类数过多，增大eps（放宽密度要求）
            eps *= 1.3
            print(f"  ⚠ Too many clusters, increasing eps to {eps:.4f}...")
        else:
            print(f"  ⚠ Warning: Still {n_clusters} clusters after {max_attempts} attempts")
    
    # 处理噪声点：将噪声点(-1)分配给最近的簇
    if n_noise > 0:
        # 找到所有非噪声点
        valid_mask = pred_labels != -1
        valid_features = features_2d[valid_mask]
        valid_labels = pred_labels[valid_mask]
        
        # 对噪声点进行k近邻分类
        noise_mask = pred_labels == -1
        noise_features = features_2d[noise_mask]
        
        if len(valid_features) > 0:
            knn = KNeighborsClassifier(n_neighbors=min(5, len(valid_features)))
            knn.fit(valid_features, valid_labels)
            pred_labels[noise_mask] = knn.predict(noise_features)
            print(f"  Reassigned noise points to nearest clusters")
    
    # 重新映射标签为连续整数（0, 1, 2, ...）
    unique_labels = np.unique(pred_labels)
    label_mapping = {old: new for new, old in enumerate(unique_labels)}
    pred_labels = np.array([label_mapping[lbl] for lbl in pred_labels])
    n_clusters = len(unique_labels)  # 更新聚类数
    
    # 计算指标（在2维空间）
    ari = adjusted_rand_score(labels, pred_labels)
    nmi = normalized_mutual_info_score(labels, pred_labels)
    
    # 只有当聚类数>1时才计算轮廓系数
    if n_clusters > 1:
        sil = silhouette_score(features_2d, pred_labels)
    else:
        sil = 0.0
    
    # 计算准确率（通过N-to-1映射）
    acc = cluster_accuracy(labels, pred_labels)
    
    return {
        'acc': acc,
        'ari': ari,
        'nmi': nmi,
        'sil': sil,
        'pred_labels': pred_labels,
        'n_clusters': n_clusters,
        'n_true_classes': n_true_classes,
        'features_reduced': features_2d  # 返回2维特征，直接用于可视化
    }


def cluster_accuracy(y_true, y_pred):
    """
    计算聚类准确率（多对一映射，N-to-1）
    参考 benchmark.py 中的实现
    
    每个聚类簇映射到包含最多样本的真实类别，允许多个簇映射到同一类别
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    
    # 构建混淆矩阵：cm[true_label, pred_cluster]
    cm = confusion_matrix(y_true, y_pred)
    
    # 多对一映射：每个聚类簇映射到包含最多样本的真实类别
    n_clusters = len(np.unique(y_pred))
    best_mapping = {}
    for cluster_id in range(n_clusters):
        # 找出该簇中最多的真实类别
        best_class = np.argmax(cm[:, cluster_id])
        best_mapping[cluster_id] = best_class
    
    # 根据映射计算准确率
    mapped_pred = np.array([best_mapping[p] for p in y_pred])
    accuracy = np.mean(mapped_pred == y_true)
    
    return accuracy


def visualize_results(features, labels, pred_labels, save_path, method='umap'):
    """可视化聚类结果"""
    # 降维到2D
    if method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        features_2d = reducer.fit_transform(features)
    else:  # tsne
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 真实标签
    scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=labels, cmap='tab10', s=5, alpha=0.6)
    ax1.set_title('True Labels', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    plt.colorbar(scatter1, ax=ax1)
    
    # 聚类结果
    scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=pred_labels, cmap='tab10', s=5, alpha=0.6)
    ax2.set_title('Clustering Results', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 可视化结果已保存至: {save_path}")


def plot_training_curve(losses, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('TCCL Training Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 训练曲线已保存至: {save_path}")


# ============================================================================
# 主函数
# ============================================================================

def run_single_dataset(dataset_name, data_root, epochs=50, batch_size=128, 
                       window_size=1024, stride=512, feature_dim=128, 
                       kernel_width=5, temperature=0.1, template_lambda=0.5,
                       lr=0.001, weight_decay=1e-6, device_str='cuda', seed=42,
                       patience=5, min_delta=0.0001, early_stop=True):
    """运行单个数据集的训练和评估，返回结果"""
    
    # 设置随机种子确保实验可复现
    set_random_seed(seed)
    
    # 设置设备
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"🚀 TCCL实验 - {dataset_name}")
    print(f"{'='*80}")
    print(f"数据集: {dataset_name}")
    print(f"设备: {device}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # 1. 加载数据集
    # ========================================================================
    print("📂 加载数据集...")
    
    if dataset_name == 'CWRU':
        train_dataset = CWRUDataset(
            data_root, 
            window_size=window_size,
            stride=stride,
            augment=True,
            verbose=True
        )
        eval_dataset = CWRUDataset(
            data_root,
            window_size=window_size,
            stride=stride,
            augment=False,
            verbose=False  # 不重复输出
        )
    elif dataset_name == 'SEU':
        train_dataset = SEUDataset(
            data_root,
            window_size=window_size,
            stride=stride,
            augment=True,
            verbose=True
        )
        eval_dataset = SEUDataset(
            data_root,
            window_size=window_size,
            stride=stride,
            augment=False,
            verbose=False  # 不重复输出
        )
    else:  # MFPT
        train_dataset = MFPTDataset(
            data_root,
            window_size=window_size,
            stride=stride,
            augment=True,
            verbose=True
        )
        eval_dataset = MFPTDataset(
            data_root,
            window_size=window_size,
            stride=stride,
            augment=False,
            verbose=False  # 不重复输出
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows下设为0避免进程管理问题
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Windows下设为0避免进程管理问题
        pin_memory=True
    )
    
    n_classes = len(np.unique(train_dataset.labels))
    print(f"✓ 数据集加载完成！类别数: {n_classes}\n")
    
    # ========================================================================
    # 2. 创建模型
    # ========================================================================
    print("🔨 创建TCCL模型...")
    
    model = TCCLModel(
        feature_dim=feature_dim,
        kernel_width=kernel_width,
        temperature=temperature,
        template_lambda=template_lambda
    ).to(device)
    
    # 统计参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型创建完成！参数量: {n_params:,}\n")
    
    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    # ========================================================================
    # 3. 训练模型（带早停）
    # ========================================================================
    if early_stop:
        print(f"🎓 开始训练（早停机制: patience={patience}, min_delta={min_delta}）...\n")
    else:
        print("🎓 开始训练（早停机制已禁用）...\n")
    
    losses = []
    
    # 早停参数
    best_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_state = None
    early_stop_enabled = early_stop  # 是否启用早停
    
    for epoch in range(1, epochs + 1):
        loss = train_tccl(model, train_loader, optimizer, device, epoch)
        losses.append(loss)
        
        # 检查是否是最佳模型
        if loss < best_loss - min_delta:
            best_loss = loss
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型状态（深拷贝）
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            status = "✓ 新的最佳模型！"
        else:
            patience_counter += 1
            status = f"(无改进: {patience_counter}/{patience})"
        
        # scheduler.step()
        
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f} - Best: {best_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f} {status}")
        
        # 早停检查
        if early_stop_enabled and patience_counter >= patience:
            print(f"\n⚠️ 早停触发！连续{patience}个epoch无改进")
            print(f"   最佳Epoch: {best_epoch}, 最佳Loss: {best_loss:.4f}")
            break
    
    # 恢复最佳模型权重
    if best_model_state is not None:
        print(f"\n🔄 恢复最佳模型权重 (Epoch {best_epoch}, Loss {best_loss:.4f})")
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    print(f"\n✓ 训练完成！最终损失: {losses[-1]:.4f}, 最佳损失: {best_loss:.4f}\n")
    
    # ========================================================================
    # 保存模型
    # ========================================================================
    print("💾 保存模型...")
    
    # 创建模型保存目录
    model_dir = './saved_models'
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存完整模型（包括结构和权重）
    model_path = os.path.join(model_dir, f'tccl_{dataset_name.lower()}_best.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': best_epoch,  # 保存最佳epoch
        'loss': best_loss,    # 保存最佳loss
        'final_epoch': len(losses),  # 实际训练的epoch数
        'final_loss': losses[-1],     # 最终的loss
        'feature_dim': feature_dim,
        'kernel_width': kernel_width,
        'temperature': temperature,
        'template_lambda': template_lambda,
        'dataset_name': dataset_name
    }, model_path)
    
    print(f"✓ 模型已保存至: {model_path}\n")
    
    # ========================================================================
    # 4. 评估模型
    # ========================================================================
    print("📊 评估模型性能...\n")
    
    # 提取特征
    features, labels = extract_all_features(model, eval_loader, device)
    print(f"✓ 特征提取完成！形状: {features.shape}\n")
    
    # 聚类评估（使用DBSCAN自动确定聚类数）
    print("执行聚类评估...")
    results = evaluate_clustering(features, labels)
    
    print("\n" + "="*80)
    print("📊 聚类性能结果")
    print("="*80)
    print(f"  聚类数/真实类别数: {results['n_clusters']}/{results['n_true_classes']} " + 
          f"({'=' if results['n_clusters'] == results['n_true_classes'] else 'N-to-1 映射'})")
    print(f"  Accuracy:  {results['acc']:.4f}")
    print(f"  ARI:       {results['ari']:.4f}")
    print(f"  NMI:       {results['nmi']:.4f}")
    print(f"  Silhouette: {results['sil']:.4f}")
    print("="*80 + "\n")
    
    # ========================================================================
    # 5. 获取2D特征（已在聚类评估中生成）
    # ========================================================================
    # 直接使用evaluate_clustering返回的2D特征，保证可视化和评价指标一致
    features_2d = results['features_reduced']
    
    print("="*80)
    print(f"✅ {dataset_name} 实验完成！")
    print("="*80)
    
    # 返回结果供后续对比
    return {
        'dataset': dataset_name,
        'features': features,
        'features_2d': features_2d,
        'labels': labels,
        'pred_labels': results['pred_labels'],
        'metrics': results,
        'losses': losses,  # 完整的损失历史
        'final_loss': losses[-1]
    }


def compare_datasets(results_list):
    """Compare UMAP visualization results of three datasets (Ground Truth vs Clustering vs Loss)"""
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    for idx, result in enumerate(results_list):
        dataset_name = result['dataset']
        features_2d = result['features_2d']
        labels = result['labels']
        pred_labels = result['pred_labels']
        metrics = result['metrics']
        losses = result['losses']
        
        # Column 1: Ground Truth
        ax_true = axes[idx, 0]
        unique_labels = np.unique(labels)
        colors_true = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            ax_true.scatter(features_2d[mask, 0], features_2d[mask, 1],
                          c=[colors_true[i]], label=f'True Class {label}',
                          s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
        
        ax_true.set_title(f'{dataset_name} - Ground Truth', fontsize=12, fontweight='bold')
        ax_true.set_xlabel('UMAP 1', fontsize=10)
        ax_true.set_ylabel('UMAP 2', fontsize=10)
        ax_true.legend(loc='best', fontsize=8)
        ax_true.grid(True, alpha=0.3)
        
        # Column 2: Clustering Results
        ax_pred = axes[idx, 1]
        unique_clusters = np.unique(pred_labels)
        colors_pred = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster in enumerate(unique_clusters):
            mask = (pred_labels == cluster)
            ax_pred.scatter(features_2d[mask, 0], features_2d[mask, 1],
                          c=[colors_pred[i]], label=f'Cluster {cluster}',
                          s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
        
        ax_pred.set_title(f'{dataset_name} - Clustering (K={metrics["n_clusters"]})\nAcc: {metrics["acc"]:.3f} | ARI: {metrics["ari"]:.3f} | NMI: {metrics["nmi"]:.3f}',
                         fontsize=12, fontweight='bold')
        ax_pred.set_xlabel('UMAP 1', fontsize=10)
        ax_pred.set_ylabel('UMAP 2', fontsize=10)
        ax_pred.legend(loc='best', fontsize=8)
        ax_pred.grid(True, alpha=0.3)
        
        # Column 3: Training Loss Curve
        ax_loss = axes[idx, 2]
        epochs = range(1, len(losses) + 1)
        ax_loss.plot(epochs, losses, linewidth=2, color='#e74c3c', marker='o', 
                    markersize=3, markevery=max(1, len(losses)//20))
        
        # 标注初始和最终损失
        ax_loss.text(1, losses[0], f'{losses[0]:.3f}', 
                    fontsize=9, va='bottom', ha='right', color='#e74c3c')
        ax_loss.text(len(losses), losses[-1], f'{losses[-1]:.3f}', 
                    fontsize=9, va='bottom', ha='left', color='#e74c3c')
        
        # 判断收敛性
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        converged = "✓" if improvement > 10 else "?"
        
        ax_loss.set_title(f'{dataset_name} - Training Loss {converged}\nImprovement: {improvement:.1f}%',
                         fontsize=12, fontweight='bold')
        ax_loss.set_xlabel('Epoch', fontsize=10)
        ax_loss.set_ylabel('Loss', fontsize=10)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_xlim(0, len(losses) + 1)
    
    plt.suptitle('TCCL Results - Three Datasets (Ground Truth | Clustering | Training Loss)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    print(f"\n✓ Comparison plot displayed")


def main(seed=42, epochs=100, batch_size=128, lr=0.0005, temperature=0.5, 
         template_lambda=0.5, feature_dim=128, device_str='cuda', 
         cwru_path=None, seu_path=None, mfpt_path=None,
         patience=15, min_delta=0.0001, early_stop=True):
    """主函数：依次运行三个数据集并对比"""
    
    # ========================================================================
    # 数据集配置
    # ========================================================================
    datasets_config = [
        {
            'name': 'CWRU',
            'path': cwru_path or 'E:/AI/CWRU-dataset-main/48k007'
        },
        {
            'name': 'SEU', 
            'path': seu_path or 'E:/AI/Mechanical-datasets-master/dataset'
        },
        {
            'name': 'MFPT',
            'path': mfpt_path or 'E:/AI/MFPT-Fault-Data-Sets-20200227T131140Z-001/MFPT/MFPT'
        }
    ]
    
    # 统一的训练参数
    train_config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'window_size': 1024,
        'stride': 512,
        'feature_dim': feature_dim,
        'kernel_width': 5,  # 改为5，增强模板表达能力
        'temperature': temperature,
        'template_lambda': template_lambda,  # 模板对比损失权重
        'lr': lr,
        'weight_decay': 1e-4,
        'device_str': device_str,
        'seed': seed,                # 随机种子，确保实验可复现
        'patience': patience,        # 早停耐心值
        'min_delta': min_delta,      # 最小改进阈值
        'early_stop': early_stop     # 是否启用早停
    }
    
    print("\n" + "="*100)
    print("🚀 TCCL 三数据集批量实验")
    print("="*100)
    print(f"将依次运行: {', '.join([d['name'] for d in datasets_config])}")
    print(f"随机种子: {train_config['seed']}")
    print(f"训练轮数: {train_config['epochs']}")
    print(f"批次大小: {train_config['batch_size']}")
    print(f"学习率: {train_config['lr']}")
    print(f"温度参数: {train_config['temperature']}")
    print(f"模板损失权重: {train_config['template_lambda']}")
    print(f"特征维度: {train_config['feature_dim']}")
    print(f"早停机制: {'启用' if train_config['early_stop'] else '禁用'} (patience={train_config['patience']}, min_delta={train_config['min_delta']})")
    print("="*100)
    
    # ========================================================================
    # 依次运行三个数据集
    # ========================================================================
    all_results = []
    
    for dataset_cfg in datasets_config:
        result = run_single_dataset(
            dataset_name=dataset_cfg['name'],
            data_root=dataset_cfg['path'],
            **train_config
        )
        all_results.append(result)
    
    # ========================================================================
    # 打印汇总表格
    # ========================================================================
    print("\n" + "="*115)
    print("📊 三数据集性能汇总")
    print("="*115)
    print(f"{'数据集':<15} {'K/C':>6} {'Accuracy':>10} {'ARI':>10} {'NMI':>10} {'Silhouette':>12} {'Loss':>10}")
    print(f"{'':15} {'':6} {'(N-to-1)':>10}")
    print("-"*115)
    
    for result in all_results:
        dataset = result['dataset']
        m = result['metrics']
        loss = result['final_loss']
        k_ratio = f"{m['n_clusters']}/{m['n_true_classes']}"
        print(f"{dataset:<15} {k_ratio:>6} {m['acc']:>10.4f} {m['ari']:>10.4f} {m['nmi']:>10.4f} "
              f"{m['sil']:>12.4f} {loss:>10.4f}")
    
    print("="*115)
    print("注: K/C = 聚类数/真实类别数, K>C 表示使用了 N-to-1 映射")
    
    # ========================================================================
    # 生成对比可视化（最后展示）
    # ========================================================================
    print("\n" + "="*100)
    print("📊 展示三数据集UMAP对比可视化...")
    print("="*100)
    
    compare_datasets(all_results)
    
    print("\n" + "="*100)
    print("✅ 所有实验完成！")
    print("="*100)
    

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TCCL三数据集批量实验')
    
    # 基本参数
    parser.add_argument('--seed', type=int, default=42, 
                       help='随机种子 (默认: 42)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='训练轮数 (默认: 100, 推荐: 100-200)')
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='批次大小 (默认: 128)')
    parser.add_argument('--lr', type=float, default=0.0005, 
                       help='学习率 (默认: 0.0005, 推荐范围: 0.0003-0.001)')
    parser.add_argument('--temperature', type=float, default=0.5, 
                       help='温度参数 (默认: 0.5, 推荐范围: 0.3-0.7)')
    parser.add_argument('--template_lambda', type=float, default=0.5,
                       help='模板对比损失权重 (默认: 0.5, 推荐范围: 0.3-1.0)')
    parser.add_argument('--feature_dim', type=int, default=128, 
                       help='特征维度 (默认: 128, 推荐: 64/128/256)')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='设备 (默认: cuda)')
    
    # 早停参数
    parser.add_argument('--patience', type=int, default=3,
                       help='早停耐心值 (默认: 15, 连续N个epoch无改进则停止)')
    parser.add_argument('--min_delta', type=float, default=0.0001,
                       help='最小改进阈值 (默认: 0.0001)')
    parser.add_argument('--no_early_stop', action='store_true',
                       help='禁用早停机制 (默认: 启用)')
    
    # 数据集路径（可选，如果不提供则使用默认路径）
    parser.add_argument('--cwru_path', type=str, 
                       default='E:/AI/CWRU-dataset-main/48k007',
                       help='CWRU数据集路径')
    parser.add_argument('--seu_path', type=str, 
                       default='E:/AI/Mechanical-datasets-master/dataset',
                       help='SEU数据集路径')
    parser.add_argument('--mfpt_path', type=str, 
                       default='E:/AI/MFPT-Fault-Data-Sets-20200227T131140Z-001/MFPT/MFPT',
                       help='MFPT数据集路径')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # 使用命令行参数更新配置
    main(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
         lr=args.lr, temperature=args.temperature, template_lambda=args.template_lambda,
         feature_dim=args.feature_dim, device_str=args.device, 
         cwru_path=args.cwru_path, seu_path=args.seu_path, mfpt_path=args.mfpt_path,
         patience=args.patience, min_delta=args.min_delta, 
         early_stop=not args.no_early_stop)

