"""
TCCL (Temporal Convolutional Contrastive Learning)

Baseline-style implementation under models/baselines/contrastive, consistent with other contrastive models.
- Uses BaseModel interface
- Accepts an external FeatureExtractor
- Forward returns a similarity matrix [B, B]
- Extracts features for evaluation
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel
from ..common import FeatureExtractor


class TemplateHead(nn.Module):
    """Generate temporal templates from feature maps."""

    def __init__(self, in_channels: int, kernel_width: int = 3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(kernel_width),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


class TCCLModel(BaseModel):
    """
    TCCL baseline model (contrastive):
    - feature_extractor: 1D CNN producing [B, C, W]
    - template_head: generates templates [B, C, W_k] from view1 features
    - forward returns similarity matrix [B, B]
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        kernel_width: int = 3,
        temperature: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = 'contrastive'
        self.temperature = temperature

        # Default feature extractor if not provided
        self.feature_extractor = (
            feature_extractor if feature_extractor is not None else FeatureExtractor(in_channels=1, out_channels=64)
        )
        self.template_head = TemplateHead(
            in_channels=getattr(self.feature_extractor, 'out_channels', 64),
            kernel_width=kernel_width,
        )

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        # Extract features
        feat1 = self.feature_extractor(view1)  # [B, C, W]
        feat2 = self.feature_extractor(view2)  # [B, C, W]

        # Templates from view1
        templates = self.template_head(feat1)  # [B, C, W_k]
        features = feat2  # [B, C, W]

        B, C, W_k = templates.shape
        _, _, W = features.shape

        # Pair all templates and features via grouped conv
        templates_exp = templates.view(B, -1).unsqueeze(1).expand(-1, B, -1)
        templates_exp = templates_exp.reshape(B * B, C, W_k)

        features_exp = features.unsqueeze(0).expand(B, -1, -1, -1)
        features_exp = features_exp.reshape(B * B, C, W)

        templates_conv = templates_exp.reshape(B * B * C, 1, W_k)
        features_conv = features_exp.reshape(1, B * B * C, W)

        conv_out = F.conv1d(features_conv, templates_conv, groups=B * B * C, padding='same')
        conv_out = conv_out.view(B * B, C, W)

        response = torch.sum(conv_out, dim=1)  # [B*B, W]
        similarity = F.adaptive_avg_pool1d(response.unsqueeze(1), 1).squeeze().view(B, B)

        return similarity

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_extractor(x)  # [B, C, W]
        z = h.view(h.size(0), -1)      # [B, C*W]
        return z.detach().cpu().numpy()
