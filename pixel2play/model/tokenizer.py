"""
ConvTokenizer: maps a raw video frame to a fixed-size embedding vector.

Uses the first 6 blocks of a pretrained EfficientNet-b0 as a feature extractor,
followed by a linear projection to the transformer's embedding dimension.
"""

import torch
import torch.nn as nn
import torchvision


_FEATURE_DIMS = {
    # (frame_h, frame_w) → spatial output size of EfficientNet block 0:5
    (192, 192): (112, 12, 12),
    (256, 256): (112, 16, 16),
}


class ConvTokenizer(nn.Module):
    def __init__(self, frame_height: int, frame_width: int, embed_dim: int, n_tokens: int = 1):
        super().__init__()
        self.n_tokens = n_tokens

        key = (frame_height, frame_width)
        if key not in _FEATURE_DIMS:
            raise ValueError(f"Unsupported frame size {frame_height}×{frame_width}. "
                             f"Supported: {list(_FEATURE_DIMS)}")
        C, H, W = _FEATURE_DIMS[key]
        n_features = C * H * W

        efficientnet = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.backbone = efficientnet.features[0:6]
        # Freeze backbone: it's pretrained and has dropout/BN that causes train > val loss
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()  # keep BN in eval mode permanently

        self.proj = nn.Sequential(
            nn.Linear(n_features, embed_dim * n_tokens),
            nn.LayerNorm(embed_dim * n_tokens),
        )
        self.embed_dim = embed_dim

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()   # always keep frozen backbone in eval mode
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        features = self.backbone(x.reshape(B * T, C, H, W))   # (BT, 112, h, w)
        features = features.reshape(B * T, -1)
        out = self.proj(features)                               # (BT, embed_dim * n_tokens)
        return out.reshape(B, T, self.n_tokens, self.embed_dim)

    def n_img_tokens(self) -> int:
        return self.n_tokens
