"""
公共组件

提供所有基线方法共用的组件：
- 特征提取器
- 手工特征提取函数
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import skew, kurtosis


class FeatureExtractor(nn.Module):
    """
    标准的3层1D-CNN特征提取器
    
    用于深度学习方法（深度聚类和对比学习）
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数（特征维度）
        """
        super().__init__()
        self.out_channels = out_channels
        
        self.encoder = nn.Sequential(
            # 第一层
            nn.Conv1d(in_channels, 16, kernel_size=17, stride=1, padding=8),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # 第二层
            nn.Conv1d(16, 32, kernel_size=17, stride=1, padding=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # 第三层
            nn.Conv1d(32, out_channels, kernel_size=17, stride=1, padding=8),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入信号 [B, C, L]
        
        Returns:
            特征图 [B, out_channels, L']
        """
        return self.encoder(x)


# ==================== 手工特征提取 ====================

def get_fft(x: np.ndarray, fs: float = 12000) -> tuple:
    """计算FFT"""
    n = len(x)
    fft_vals = np.fft.fft(x)
    fft_freqs = np.fft.fftfreq(n, 1.0/fs)
    mask = fft_freqs >= 0
    return fft_freqs[mask], (2.0/n) * np.abs(fft_vals[mask])


# 时域特征函数
def tf1_mean(x): return np.mean(x)
def tf2_std(x): return np.std(x)
def tf3_rms(x): return np.sqrt(np.mean(x**2))
def tf4_peak(x): return np.max(np.abs(x))
def tf5_skewness(x): return skew(x)
def tf6_kurtosis(x): return kurtosis(x, fisher=False)
def tf7_crest_factor(x): return tf4_peak(x) / tf3_rms(x) if tf3_rms(x) > 0 else 0
def tf8_clearance_factor(x):
    den = np.mean(np.sqrt(np.abs(x)))**2
    return tf4_peak(x) / den if den > 0 else 0
def tf9_shape_factor(x):
    den = np.mean(np.abs(x))
    return tf3_rms(x) / den if den > 0 else 0
def tf10_impulse_factor(x):
    den = np.mean(np.abs(x))
    return tf4_peak(x) / den if den > 0 else 0


# 频域特征函数
def ff1_mean_freq(spectrum): 
    return np.mean(spectrum)

def ff2_freq_center(freqs, spectrum):
    den = np.sum(spectrum)
    return np.sum(freqs * spectrum) / den if den > 0 else 0

def ff3_rms_freq(freqs, spectrum):
    den = np.sum(spectrum)
    return np.sqrt(np.sum((freqs**2) * spectrum) / den) if den > 0 else 0

def ff4_std_freq(freqs, spectrum):
    fc = ff2_freq_center(freqs, spectrum)
    return np.sqrt(np.sum(((freqs - fc)**2) * spectrum) / np.sum(spectrum)) if np.sum(spectrum) > 0 else 0

def ff5_avg_freq(freqs, spectrum):
    den = np.sum((freqs**2) * spectrum)
    return np.sqrt(np.sum((freqs**3) * spectrum) / den) if den > 0 else 0

def ff6_stabilization_factor(freqs, spectrum):
    num = np.sum((freqs**2) * spectrum)
    den = np.sqrt(np.sum(spectrum) * np.sum((freqs**4) * spectrum))
    return num / den if den > 0 else 0

def ff7_coeff_variability(freqs, spectrum):
    ff2_val = ff2_freq_center(freqs, spectrum)
    ff4_val = ff4_std_freq(freqs, spectrum)
    return ff4_val / ff2_val if ff2_val > 0 else 0

def ff8_freq_skewness(freqs, spectrum):
    fc = ff2_freq_center(freqs, spectrum)
    ff4_val = ff4_std_freq(freqs, spectrum)**3
    num = np.sum(((freqs - fc)**3) * spectrum)
    return num / (len(spectrum) * ff4_val) if ff4_val > 0 else 0

def ff9_freq_kurtosis(freqs, spectrum):
    fc = ff2_freq_center(freqs, spectrum)
    ff4_val = ff4_std_freq(freqs, spectrum)**4
    num = np.sum(((freqs - fc)**4) * spectrum)
    return num / (len(spectrum) * ff4_val) if ff4_val > 0 else 0


def extract_handcrafted_features(signal: np.ndarray, fs: float = 12000) -> np.ndarray:
    """
    提取19维手工特征（10个时域 + 9个频域）
    
    Args:
        signal: 一维信号
        fs: 采样频率
    
    Returns:
        19维特征向量
    """
    # 时域特征（10个）
    td_features = [
        tf1_mean(signal), tf2_std(signal), tf3_rms(signal), tf4_peak(signal),
        tf5_skewness(signal), tf6_kurtosis(signal), tf7_crest_factor(signal),
        tf8_clearance_factor(signal), tf9_shape_factor(signal), tf10_impulse_factor(signal)
    ]
    
    # 频域特征（9个）
    freqs, spectrum = get_fft(signal, fs)
    fd_features = [
        ff1_mean_freq(spectrum), ff2_freq_center(freqs, spectrum),
        ff3_rms_freq(freqs, spectrum), ff4_std_freq(freqs, spectrum),
        ff5_avg_freq(freqs, spectrum), ff6_stabilization_factor(freqs, spectrum),
        ff7_coeff_variability(freqs, spectrum), ff8_freq_skewness(freqs, spectrum),
        ff9_freq_kurtosis(freqs, spectrum)
    ]
    
    return np.array(td_features + fd_features)

