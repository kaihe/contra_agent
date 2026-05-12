"""
SmallResNetEncoder: CNN vision encoder for NES frames.

Input : (B, T, C, H, W) uint8   — e.g. (B, T, 1, 84, 84) or (B, T, 1, 168, 168) grayscale
Output: (B, T, 1, D)            — single image token per timestep

Architecture
------------
Stem  : Conv 1→32, stride=2      (H → H/2)
Block1: ResBlock 32→32           (H/2 × H/2)
Down1 : Conv 32→64, stride=2     (H/2 → H/4)
Block2: ResBlock 64→64           (H/4 × H/4)
Down2 : Conv 64→128, stride=2    (H/4 → H/8)
Block3: ResBlock 128→128         (H/8 × H/8)
Pool  : AdaptiveAvgPool2d(4)     (any → 4)
Proj  : Linear(128*4*4, D)       (2048 → D)
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Basic ResNet block with batch norm."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class SmallResNetEncoder(nn.Module):
    """3-block ResNet encoder: 32 → 64 → 128 channels.
    Outputs a grid of spatial tokens rather than a single flattened token.
    """

    def __init__(self, embed_dim: int = 1024, in_channels: int = 1, grid_size: int = 6):
        super().__init__()
        self.grid_size = grid_size
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.block1 = ResBlock(32, 32)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block2 = ResBlock(64, 64)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.block3 = ResBlock(128, 128)
        self.pool = nn.AdaptiveAvgPool2d(grid_size)
        
        # Project each spatial location's channels to embed_dim independently
        self.proj = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) uint8
        Returns:
            (B, T, grid_size*grid_size, embed_dim)
        """
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        x = x.float() / 255.0

        x = self.stem(x)      # (B*T, 32, H/2, H/2)
        x = self.block1(x)    # (B*T, 32, H/2, H/2)
        x = self.down1(x)     # (B*T, 64, H/4, H/4)
        x = self.block2(x)    # (B*T, 64, H/4, H/4)
        x = self.down2(x)     # (B*T, 128, H/8, H/8)
        x = self.block3(x)    # (B*T, 128, H/8, H/8)
        x = self.pool(x)      # (B*T, 128, grid_size, grid_size)
        
        x = x.flatten(2).transpose(1, 2)  # (B*T, grid_size*grid_size, 128)
        x = self.proj(x)                  # (B*T, grid_size*grid_size, embed_dim)

        return x.view(B, T, self.grid_size * self.grid_size, -1)

    def n_img_tokens(self) -> int:
        return self.grid_size * self.grid_size
