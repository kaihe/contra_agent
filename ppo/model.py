"""Shared policy network for Contra — one builder for BC pretraining and PPO RL.

The agent observes a Dict ``{image, priv}``:
  image : (res, res, stack) uint8   stacked grayscale frames
  priv  : (STATE_DIM,) float32      structured RAM state (contra.game_state)

A single shared features extractor (NatureCNN on ``image`` + MLP on ``priv``)
feeds both the actor and the critic, so the network is identical whether it is
behavior-cloned (ppo/pretrain_train.py) or trained with PPO (ppo/train.py).
The privileged RAM state gives the value head the signals it cannot read from
pixels (boss HP, positions, scene flags), and the actor benefits from them too.
"""

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from contra.action_space import DEFAULT as ACTION_SPACE
from contra.game_state import STATE_DIM

POLICY = "MultiInputPolicy"

FIRE_BIT = torch.tensor(ACTION_SPACE.actions_np()[:, 0].astype(np.float32))
JUMP_BIT = torch.tensor(ACTION_SPACE.actions_np()[:, 8].astype(np.float32))


def observation_space(stack, resolution=84, priv_dim=STATE_DIM):
    """The Dict observation space shared by the wrapper, BC, and PPO."""
    return gym.spaces.Dict(
        {
            "image": gym.spaces.Box(0, 255, (resolution, resolution, stack), np.uint8),
            "priv": gym.spaces.Box(-np.inf, np.inf, (priv_dim,), np.float32),
        }
    )


class SpaceStub(gym.Env):
    """Minimal env exposing the observation/action spaces (for building a model)."""

    def __init__(self, stack, resolution=84, priv_dim=STATE_DIM):
        super().__init__()
        self.observation_space = observation_space(stack, resolution, priv_dim)
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


class ResCnn(nn.Module):
    """Residual conv backbone; resolution-agnostic via global average pool.

    Four stride-2 residual stages downsample the frame, then AdaptiveAvgPool2d
    collapses to 1x1 so the head size is independent of input resolution —
    unlike NatureCNN, whose flatten FC balloons (~13M params) at 192x192. Deeper
    and far lighter, it actually uses the extra resolution convolutionally.
    """

    def __init__(self, image_space, features_dim=512, channels=(32, 64, 128, 128)):
        super().__init__()
        in_ch = image_space.shape[0]
        layers = []
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                ResidualBlock(out_ch),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.linear = nn.Sequential(
            nn.Flatten(), nn.Linear(in_ch, features_dim), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.linear(self.cnn(x))


_BACKBONES = {"nature": NatureCNN, "rescnn": ResCnn}


class ImageStateExtractor(BaseFeaturesExtractor):
    """Image backbone + MLP over the RAM state, concatenated.

    Shared by actor and critic. The image backbone is selectable
    (``nature`` for 84x84, ``rescnn`` for higher resolutions). A leading
    BatchNorm standardises the heterogeneously-scaled ``priv`` vector per feature
    (so one-hots and the ~3000-px scroll coordinate coexist), removing the need
    for VecNormalize.
    """

    def __init__(self, observation_space, backbone="nature", cnn_dim=512, state_dim=128):
        super().__init__(observation_space, features_dim=cnn_dim + state_dim)
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}; choose from {sorted(_BACKBONES)}")
        self.cnn = _BACKBONES[backbone](observation_space["image"], features_dim=cnn_dim)
        n_priv = observation_space["priv"].shape[0]
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(n_priv),
            nn.Linear(n_priv, state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs):
        return torch.cat([self.cnn(obs["image"]), self.mlp(obs["priv"])], dim=1)


def policy_kwargs(backbone="nature", cnn_dim=512, state_dim=128):
    return dict(
        features_extractor_class=ImageStateExtractor,
        features_extractor_kwargs=dict(backbone=backbone, cnn_dim=cnn_dim, state_dim=state_dim),
    )


def build_ppo_model(stack, resolution, gamma, device,
                    backbone="nature", cnn_dim=512, state_dim=128, **ppo_kwargs):
    """Build a PPO model on the shared policy (used by BC pretraining)."""
    env = DummyVecEnv([lambda: SpaceStub(stack, resolution)])
    return PPO(
        POLICY,
        env,
        device=device,
        gamma=gamma,
        verbose=0,
        policy_kwargs=policy_kwargs(backbone, cnn_dim, state_dim),
        **ppo_kwargs,
    )


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def forward_actor(policy, image_hwc, priv, device):
    """(B,H,W,C) uint8 images + (B,STATE_DIM) priv -> action logits."""
    obs = {
        "image": image_hwc.to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous(),
        "priv": priv.to(device, non_blocking=True),
    }
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
