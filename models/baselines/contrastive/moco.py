"""
MoCo (Momentum Contrast)

论文: He et al., "Momentum Contrast for Unsupervised Visual Representation Learning", CVPR 2020

核心思想：
- 使用动量编码器
- 维护队列存储负样本
- 解决batch size限制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ..base import BaseModel


class MoCoModel(BaseModel):
    """
    MoCo: Momentum Contrast (CVPR 2020)
    
    方法特点：
    1. 动量编码器（momentum encoder）
    2. 队列（queue）机制存储负样本
    3. 不需要大batch size
    
    论文链接：https://arxiv.org/abs/1911.05722
    """
    
    def __init__(self, feature_extractor: nn.Module,
                 projection_dim: int = 128, hidden_dim: int = 2048,
                 queue_size: int = 65536, momentum: float = 0.999, **kwargs):
        super().__init__(**kwargs)
        self.model_type = 'contrastive'
        self.feature_extractor = feature_extractor
        self.projection_dim = projection_dim
        self.queue_size = queue_size
        self.momentum = momentum
        
        # 在线编码器的投影头
        self.online_head = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # 目标编码器（动量更新）
        self.target_extractor = copy.deepcopy(feature_extractor)
        self.target_head = copy.deepcopy(self.online_head)
        
        for param in self.target_extractor.parameters():
            param.requires_grad = False
        for param in self.target_head.parameters():
            param.requires_grad = False
        
        # 队列
        self.register_buffer('queue', torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_encoder(self):
        """动量更新目标编码器"""
        for param_online, param_target in zip(
            self.feature_extractor.parameters(),
            self.target_extractor.parameters()
        ):
            param_target.data = param_target.data * self.momentum + \
                               param_online.data * (1 - self.momentum)
        
        for param_online, param_target in zip(
            self.online_head.parameters(),
            self.target_head.parameters()
        ):
            param_target.data = param_target.data * self.momentum + \
                               param_online.data * (1 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """更新队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size
        else:
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
            ptr = batch_size - remaining
        
        self.queue_ptr[0] = ptr
    
    def forward(self, view1: torch.Tensor, view2: torch.Tensor):
        """
        Full MoCo forward producing logits against a dynamic queue of negatives.
        Returns:
            logits: [B, 1 + K] where first column is positive, others are negatives
            targets: [B] all zeros (positive index)
        """
        # Query from online branch
        q_feat = self.feature_extractor(view1)
        q_feat = F.adaptive_avg_pool1d(q_feat, 1).squeeze(-1)
        q = self.online_head(q_feat)
        q = F.normalize(q, dim=1)  # [B, D]

        # Key from momentum branch
        with torch.no_grad():
            self._momentum_update_encoder()
            k_feat = self.target_extractor(view2)
            k_feat = F.adaptive_avg_pool1d(k_feat, 1).squeeze(-1)
            k = self.target_head(k_feat)
            k = F.normalize(k, dim=1)  # [B, D]

        # Positives: inner-product between q and k
        l_pos = torch.einsum('bd,bd->b', [q, k]).unsqueeze(1)  # [B,1]

        # Negatives: inner-product between q and all entries in the queue
        # queue shape: [D, K]
        l_neg = torch.einsum('bd,dk->bk', [q, self.queue])  # [B,K]

        # Logits and targets (class 0 is positive)
        logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+K]
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # Do NOT update queue here to avoid in-place modification before backward
        # Return keys so the training loop can update the queue safely after backward
        return logits, targets, k.detach()

    @torch.no_grad()
    def enqueue_dequeue(self, keys: torch.Tensor):
        """Public method to update the queue safely under no_grad."""
        self._dequeue_and_enqueue(keys)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        proj = self.online_head(feat)
        proj = F.normalize(proj, dim=1)
        return proj.cpu().numpy()
    
    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            'description': 'Momentum Contrast with queue mechanism',
            'year': 2020,
            'conference': 'CVPR',
            'reference': 'He et al., Momentum Contrast for Unsupervised Visual Representation Learning',
            'queue_size': self.queue_size,
            'momentum': self.momentum
        })
        return info

