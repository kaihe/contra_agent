"""Behavior-clone the PPO CnnPolicy (actor + critic) on Contra win traces.

Supervised warm-start: build (or load) the BC dataset (ppo/pretrain_dataset.py),
then fit a fresh PPO("CnnPolicy") so its `action_net` imitates the demonstrated
actions and its `value_net` regresses the returns-to-go. Save a checkpoint that
`ppo/train.py --resume` consumes unchanged.

The first-order risk is **action class imbalance** (the actor learning only the
frequent move-right/up-right and ignoring the decisive rare jump/fire — see
ppo/cnn_pretrain_design.md §7.1). So the actor loss is class-weighted and the
gate metric is **per-button fire/jump recall**, never aggregate accuracy.

Usage:
    python ppo/pretrain_cnn.py --level 1 --epochs 30
    python ppo/pretrain_cnn.py --level 1 --dataset tmp/ppo/pretrain/level1_bc.npz
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(__file__))
from contra_wrapper import save_config_to_model  # noqa: E402
import pretrain_dataset as PD  # noqa: E402

from contra.action_space import DEFAULT as ACTION_SPACE

OBS_SHAPE = (84, 84, 3)            # channels-last, as ContraWrapper emits
OUT_DIR = "tmp/ppo/pretrain"

_FIRE_BIT = torch.tensor(ACTION_SPACE.actions_np()[:, 0].astype(np.float32))  # B
_JUMP_BIT = torch.tensor(ACTION_SPACE.actions_np()[:, 8].astype(np.float32))  # A


class _SpaceStub(gym.Env):
    """Minimal env exposing the training obs/action spaces, so PPO builds the
    right CnnPolicy without spinning up an emulator. Never stepped for rollouts."""

    def __init__(self, stack):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, stack), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(ACTION_SPACE.num_actions)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}


def class_weights(actions, num_actions, scheme="inv_sqrt"):
    """Per-action loss weights to fight imbalance. inv_sqrt(freq), mean-normalized."""
    counts = np.bincount(actions, minlength=num_actions).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    if scheme == "inv":
        w = 1.0 / counts
    else:  # inv_sqrt — gentler, the default
        w = 1.0 / np.sqrt(counts)
    w *= num_actions / w.sum()  # mean weight ~1 so the loss scale stays comparable
    return w.astype(np.float32)


def per_button_recall(pred_idx, true_idx, bit):
    """Recall of a button (fire/jump): of steps where the demo pressed it, how
    often did the prediction also press it (via whatever action carries it)."""
    pressed = bit[true_idx] > 0.5
    n = int(pressed.sum())
    if n == 0:
        return float("nan"), 0
    hit = (bit[pred_idx][pressed] > 0.5).float().sum().item()
    return hit / n, n


def evaluate(policy, obs, actions, returns, device, batch=4096):
    """Aggregate accuracy, per-button recall, critic explained-variance, entropy."""
    policy.set_training_mode(False)
    fire_bit, jump_bit = _FIRE_BIT.to(device), _JUMP_BIT.to(device)
    correct = total = 0
    fire_hit = fire_n = jump_hit = jump_n = 0
    vals = []
    with torch.no_grad():
        for i in range(0, len(actions), batch):
            ob = obs[i:i + batch]
            a = torch.as_tensor(actions[i:i + batch], device=device)
            obs_t, _ = policy.obs_to_tensor(ob)
            feats = policy.extract_features(obs_t)
            lat_pi, lat_vf = policy.mlp_extractor(feats)
            logits = policy.action_net(lat_pi)
            value = policy.value_net(lat_vf).squeeze(-1)
            pred = logits.argmax(-1)
            correct += (pred == a).sum().item()
            total += len(a)
            r, n = per_button_recall(pred, a, fire_bit); fire_hit += (r * n if n else 0); fire_n += n
            r, n = per_button_recall(pred, a, jump_bit); jump_hit += (r * n if n else 0); jump_n += n
            vals.append(value.cpu())
    v = torch.cat(vals).numpy()
    g = returns
    ev = 1.0 - np.var(g - v) / (np.var(g) + 1e-8)
    return {
        "acc": correct / total,
        "fire_recall": fire_hit / fire_n if fire_n else float("nan"),
        "jump_recall": jump_hit / jump_n if jump_n else float("nan"),
        "critic_ev": float(ev),
    }


def train(policy, obs, actions, returns, *, device, epochs, batch_size, lr,
          vf_coef, ent_coef, cls_w):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    cls_w_t = torch.as_tensor(cls_w, device=device)
    actions_t = torch.as_tensor(actions, device=device)
    returns_t = torch.as_tensor(returns, device=device)
    n = len(actions)

    for epoch in range(1, epochs + 1):
        policy.set_training_mode(True)
        perm = torch.randperm(n)
        ep_actor = ep_critic = ep_ent = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            ob = obs[idx.numpy()]
            a = actions_t[idx]
            g = returns_t[idx]

            obs_t, _ = policy.obs_to_tensor(ob)
            feats = policy.extract_features(obs_t)
            lat_pi, lat_vf = policy.mlp_extractor(feats)
            logits = policy.action_net(lat_pi)
            value = policy.value_net(lat_vf).squeeze(-1)

            l_actor = F.cross_entropy(logits, a, weight=cls_w_t)
            l_critic = F.mse_loss(value, g)
            ent = torch.distributions.Categorical(logits=logits).entropy().mean()
            loss = l_actor + vf_coef * l_critic - ent_coef * ent

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = len(idx)
            ep_actor += l_actor.item() * bs
            ep_critic += l_critic.item() * bs
            ep_ent += ent.item() * bs

        m = evaluate(policy, obs, actions, returns, device)
        print(f"  epoch {epoch:3d}  actor_ce {ep_actor / n:6.3f}  critic_mse {ep_critic / n:8.1f}  "
              f"ent {ep_ent / n:5.3f}  | acc {m['acc']:.3f}  "
              f"fire_recall {m['fire_recall']:.3f}  jump_recall {m['jump_recall']:.3f}  "
              f"critic_ev {m['critic_ev']:.3f}", flush=True)
    return m


def load_or_build_dataset(args):
    path = args.dataset or os.path.join(PD.CACHE_DIR, f"level{args.level}_bc.npz")
    if os.path.isfile(path):
        print(f"Loading dataset from {path}")
        d = np.load(path)
        return d["obs"], d["actions"], d["returns"], int(d["stack"])
    print(f"No cache at {path}; building dataset for level {args.level}")
    traces = PD.resolve_traces(args.level)
    obs, act, ret = PD.build_dataset(traces, args.reward_config, args.stack, args.gamma)
    return obs, act, ret, args.stack


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--dataset", default=None, help="BC dataset npz (default: CACHE_DIR/level<N>_bc.npz)")
    p.add_argument("--reward-config", default="stable")
    p.add_argument("--stack", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0, help="Entropy bonus in BC (monitor-only by default)")
    p.add_argument("--weight-scheme", default="inv_sqrt", choices=["inv_sqrt", "inv", "none"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="Checkpoint path (default: OUT_DIR/level<N>_bc.zip)")
    args = p.parse_args()

    obs, actions, returns, stack = load_or_build_dataset(args)
    print(f"dataset: {len(actions)} samples  obs {obs.shape} {obs.dtype}")

    device = args.device if torch.cuda.is_available() else "cpu"
    env = DummyVecEnv([lambda: _SpaceStub(stack)])
    model = PPO("CnnPolicy", env, device=device, gamma=args.gamma, verbose=0)
    policy = model.policy

    if args.weight_scheme == "none":
        cls_w = np.ones(ACTION_SPACE.num_actions, dtype=np.float32)
    else:
        cls_w = class_weights(actions, ACTION_SPACE.num_actions, args.weight_scheme)
    print(f"class weights ({args.weight_scheme}): "
          + "  ".join(f"{ACTION_SPACE.names[i]}={cls_w[i]:.2f}" for i in np.argsort(-cls_w)[:6]) + " ...")

    print(f"\nPretraining actor+critic on {device}: epochs={args.epochs} "
          f"batch={args.batch_size} lr={args.lr} vf_coef={args.vf_coef} ent_coef={args.ent_coef}")
    train(policy, obs, actions, returns, device=device, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, vf_coef=args.vf_coef,
          ent_coef=args.ent_coef, cls_w=cls_w)

    out = args.out or os.path.join(OUT_DIR, f"level{args.level}_bc.zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    model.save(out)
    save_config_to_model(out, stack=stack, train_config={"pretrain": vars(args)})
    print(f"\nSaved warm-start checkpoint → {out}")
    print(f"Resume PPO with:  python ppo/train.py --config ppo/train_config/level{args.level}_win.yaml --resume {out}")


if __name__ == "__main__":
    main()
