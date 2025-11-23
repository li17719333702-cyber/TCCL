"""
CWRU轴承故障数据集 - 改进版

支持三种标准化模式：
1. mode='full': 使用全数据集统计量标准化（推荐用于无监督聚类）
2. mode='train': 训练模式，计算并保存统计量
3. mode='test': 测试模式，使用训练集的统计量

标准化策略：
- 数据集级别的Z-score标准化
- 保留不同文件间的振幅差异（重要的故障特征）
- 确保训练-测试一致性，避免数据泄露
"""

import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict


def load_cwru_signal(file_path: str) -> np.ndarray:
    """
    从CWRU格式的.mat文件中加载驱动端(DE)的振动信号
    自动查找以 "_DE_time" 或 "_FE_time" 结尾的键
    """
    try:
        mat_contents = sio.loadmat(file_path)
        signal_key = None
        
        # 优先查找驱动端信号
        for key in mat_contents.keys():
            if key.endswith("_DE_time"):
                signal_key = key
                break
        
        # 如果找不到，则查找风扇端信号作为备用
        if not signal_key:
            for key in mat_contents.keys():
                if key.endswith("_FE_time"):
                    signal_key = key
                    break
        
        if signal_key:
            return mat_contents[signal_key].flatten()
        else:
            print(f"警告: 在文件 {Path(file_path).name} 中找不到 _DE_time 或 _FE_time 键")
            return np.array([])
    
    except Exception as e:
        print(f"警告: 加载文件 {Path(file_path).name} 失败: {e}")
        return np.array([])


class CWRUDataset(Dataset):
    """CWRU轴承故障数据集"""
    
    LABEL_MAP = {
        'normal': 0,
        'ball': 1,          # 滚动体故障 (B)
        'inner_race': 2,    # 内圈故障 (IR)
        'outer_race': 3     # 外圈故障 (OR)
    }
    
    def __init__(self, 
                 root_dir: str, 
                 window_size: int = 1024, 
                 step_size: int = 512, 
                 augmentor=None,
                 normalization_stats: Optional[Dict] = None,
                 mode: str = 'full'):
        """
        初始化CWRU数据集
        
        参数:
            root_dir: 数据集根目录，包含.mat文件
            window_size: 滑动窗口大小
            step_size: 滑动窗口步长
            augmentor: 数据增强器（用于对比学习）
            normalization_stats: 预先计算的统计量 {'mean': ..., 'std': ...}
            mode: 模式选择
                - 'full': 全数据集模式，计算并使用全部数据的统计量（推荐用于无监督聚类）
                - 'train': 训练模式，计算并保存统计量
                - 'test': 测试模式，必须提供normalization_stats
        """
        self.root_dir = Path(root_dir)
        self.window_size = window_size
        self.step_size = step_size
        self.augmentor = augmentor
        self.mode = mode
        self.class_names = {v: k for k, v in self.LABEL_MAP.items()}
        
        self.samples = []
        self.labels = []
        
        # 标准化策略：根据模式选择不同的统计量计算方式
        if mode == 'full':
            print(f"[CWRU - 全数据集模式] 目录: {self.root_dir}")
            self.dataset_mean, self.dataset_std = self._compute_dataset_stats()
            print(f"数据集统计量: μ={self.dataset_mean:.6f}, σ={self.dataset_std:.6f}")
        elif mode == 'train':
            print(f"[CWRU - 训练集模式] 目录: {self.root_dir}")
            self.dataset_mean, self.dataset_std = self._compute_dataset_stats()
            print(f"训练集统计量: μ={self.dataset_mean:.6f}, σ={self.dataset_std:.6f}")
        elif mode == 'test':
            if normalization_stats is None:
                raise ValueError("测试模式(mode='test')必须提供normalization_stats参数")
            print(f"[CWRU - 测试集模式] 目录: {self.root_dir}")
            self.dataset_mean = normalization_stats['mean']
            self.dataset_std = normalization_stats['std']
            print(f"使用训练集统计量: μ={self.dataset_mean:.6f}, σ={self.dataset_std:.6f}")
        else:
            raise ValueError(f"不支持的模式: {mode}，请使用 'full', 'train' 或 'test'")
        
        self._load_data()
    
    def _compute_dataset_stats(self) -> tuple:
        """计算整个数据集的全局统计量（均值和标准差）"""
        mat_files = sorted(list(self.root_dir.glob('*.mat')))
        if not mat_files:
            raise FileNotFoundError(f"在 {self.root_dir} 中未找到.mat文件")
        
        print(f"扫描 {len(mat_files)} 个文件，计算全局统计量...")
        all_signals = []
        
        for file_path in tqdm(mat_files, desc="收集数据", leave=False):
            signal = load_cwru_signal(str(file_path))
            if signal.size > 0:
                all_signals.append(signal)
        
        if not all_signals:
            raise ValueError("没有成功加载任何数据文件")
        
        # 合并所有信号并计算统计量
        all_data = np.concatenate(all_signals)
        mean = all_data.mean()
        std = all_data.std() + 1e-6  # 防止除零
        
        return mean, std
    
    def _get_label_from_filename(self, filename: str) -> int:
        """根据文件名解析标签"""
        name = filename.lower()
        # 关键词匹配顺序很重要，优先匹配更具体的长字符串
        if 'normal' in name:
            return self.LABEL_MAP['normal']
        elif 'ir' in name:
            return self.LABEL_MAP['inner_race']
        elif 'or' in name:
            return self.LABEL_MAP['outer_race']
        elif 'b' in name:
            return self.LABEL_MAP['ball']
        return -1
    
    def _load_data(self):
        """加载数据并进行预处理"""
        mat_files = sorted(list(self.root_dir.glob('*.mat')))
        print(f"加载 {len(mat_files)} 个.mat文件...")
        
        for file_path in tqdm(mat_files, desc=f"加载数据({self.mode})", leave=False):
            label = self._get_label_from_filename(file_path.name)
            if label == -1:
                print(f"警告: 无法识别文件 {file_path.name} 的标签，已跳过")
                continue
            
            signal = load_cwru_signal(str(file_path))
            if signal.size < self.window_size:
                print(f"警告: 文件 {file_path.name} 信号过短 ({signal.size})，已跳过")
                continue
            
            # 使用数据集级别的统计量进行标准化
            num_windows = (len(signal) - self.window_size) // self.step_size + 1
            for i in range(num_windows):
                start_idx = i * self.step_size
                segment = signal[start_idx: start_idx + self.window_size]
                # 关键：使用数据集统计量，而非文件或窗口统计量
                segment = (segment - self.dataset_mean) / self.dataset_std
                self.samples.append(segment)
                self.labels.append(label)
        
        # 统计信息
        print(f"数据加载完成！共生成 {len(self.samples)} 个样本")
        label_counts = {name: self.labels.count(idx) for name, idx in self.LABEL_MAP.items()}
        print(f"各类别样本数: {label_counts}")
    
    def get_normalization_stats(self) -> Dict:
        """返回标准化统计量，供测试集使用"""
        return {
            'mean': self.dataset_mean,
            'std': self.dataset_std
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        signal = torch.from_numpy(self.samples[idx]).float().unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.augmentor:
            view1 = self.augmentor(signal)
            view2 = self.augmentor(signal)
            return view1, view2, label
        
        return signal, signal, label

# ==================== 使用示例 ====================
if __name__ == '__main__':
    DATA_ROOT = r'E:\AI\CWRU-dataset-main\48k007'
    
    data_path = Path(DATA_ROOT)
    if not data_path.exists():
        print(f"⚠️  示例目录 '{DATA_ROOT}' 不存在")
        print("请修改 DATA_ROOT 为你的实际数据路径")
        exit(0)
    
    print("=" * 80)
    print("CWRU数据集 - 全数据集模式示例")
    print("=" * 80)
    
    # 全数据集模式
    dataset = CWRUDataset(
        root_dir=DATA_ROOT,
        window_size=1024,
        step_size=512,
        mode='full'
    )
    
    print(f"\n✅ 数据集加载成功！")
    print(f"   总样本数: {len(dataset)}")
    print(f"   类别映射: {dataset.LABEL_MAP}")
    
    # 查看一个样本
    sample_signal, _, sample_label = dataset[0]
    print(f"   样本形状: {sample_signal.shape}")
    print(f"   样本标签: {sample_label.item()} ({dataset.class_names[sample_label.item()]})")
    
    print("\n💡 更多使用示例请参考 datasets/mfpt.py")
    print("=" * 80)



