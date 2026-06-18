"""Model helpers for Contra BC warm-start experiments."""

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from contra.action_space import DEFAULT as ACTION_SPACE

FIRE_BIT = torch.tensor(ACTION_SPACE.actions_np()[:, 0].astype(np.float32))
JUMP_BIT = torch.tensor(ACTION_SPACE.actions_np()[:, 8].astype(np.float32))


class SpaceStub(gym.Env):
    """Minimal env exposing the PPO observation/action spaces."""

    def __init__(self, stack, resolution=84):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            0,
            255,
            (resolution, resolution, stack),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Discrete(ACTION_SPACE.num_actions)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class ResCnnFeaturesExtractor(BaseFeaturesExtractor):
    """Residual visual backbone for the `--backbone rescnn` experiment."""

    def __init__(
        self,
        observation_space,
        features_dim=512,
        channels=(32, 64, 128, 128),
    ):
        super().__init__(observation_space, features_dim)
        in_channels = observation_space.shape[0]
        layers = []
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    ResidualBlock(out_channels),
                ]
            )
            in_channels = out_channels

        self.cnn = nn.Sequential(
            *layers,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))


def build_policy_kwargs(backbone, features_dim):
    if backbone == "nature":
        return None
    if backbone == "rescnn":
        return {
            "features_extractor_class": ResCnnFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": features_dim},
        }
    raise ValueError(f"Unknown backbone: {backbone}")


def build_ppo_model(stack, resolution, backbone, features_dim, gamma, device):
    env = DummyVecEnv([lambda: SpaceStub(stack, resolution)])
    model = PPO(
        "CnnPolicy",
        env,
        device=device,
        gamma=gamma,
        verbose=0,
        policy_kwargs=build_policy_kwargs(backbone, features_dim),
    )
    return model


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def forward_actor(policy, obs_hwc, device):
    """(B,H,W,C) uint8 observations -> action logits."""
    obs = obs_hwc.to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
    features = policy.extract_features(obs)
    latent_pi, _ = policy.mlp_extractor(features)
    return policy.action_net(latent_pi)


def class_weights(actions, num_actions, scheme="inv_sqrt"):
    counts = np.bincount(actions, minlength=num_actions).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    if scheme == "inv":
        weights = 1.0 / counts
    else:
        weights = 1.0 / np.sqrt(counts)
    weights *= num_actions / weights.sum()
    return weights.astype(np.float32)


def button_recall(pred_idx, true_idx, bit):
    pressed = bit[true_idx] > 0.5
    n = int(pressed.sum())
    if n == 0:
        return float("nan"), 0
    hit = (bit[pred_idx][pressed] > 0.5).float().sum().item()
    return hit / n, n
