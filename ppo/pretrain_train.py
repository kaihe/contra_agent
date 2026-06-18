"""Behavior-clone the PPO CnnPolicy actor on Contra win traces.

Usage:
    python3 ppo/pretrain_train.py --level 1 --epochs 30
    python3 ppo/pretrain_train.py --level 1 --resolution 192 --num-workers 4
    python3 ppo/pretrain_train.py --level 1 --resolution 192 --backbone rescnn --num-workers 4
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import pretrain_dataset as data  # noqa: E402
import pretrain_model as model_lib  # noqa: E402
from contra_wrapper import save_config_to_model  # noqa: E402

from contra.action_space import DEFAULT as ACTION_SPACE

OUT_DIR = "tmp/ppo/pretrain"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--reward-config", default="stable")
    parser.add_argument("--stack", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--resolution", type=int, default=84)
    parser.add_argument("--backbone", default="nature", choices=["nature", "rescnn"])
    parser.add_argument("--features-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument(
        "--weight-scheme",
        default="inv_sqrt",
        choices=["inv_sqrt", "inv", "none"],
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def checkpoint_path(args):
    if args.out:
        return args.out
    res_tag = "" if args.resolution == 84 else f"_r{args.resolution}"
    backbone_tag = "" if args.backbone == "nature" else f"_{args.backbone}"
    return os.path.join(OUT_DIR, f"level{args.level}_bc{res_tag}{backbone_tag}.zip")


def make_dataloaders(args):
    obs_path, meta = data.ensure_cache(
        level=args.level,
        resolution=args.resolution,
        reward_config=args.reward_config,
        stack=args.stack,
        gamma=args.gamma,
    )
    dataset = data.BCDataset(
        obs_path,
        meta["count"],
        meta["actions"],
        meta["returns"],
    )
    train_loader = data.make_loader(
        dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    eval_loader = data.make_loader(
        dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(
        f"dataset: {meta['count']} samples  "
        f"obs (mmap) {meta['resolution']}x{meta['resolution']}x{meta['stack']}"
    )
    return train_loader, eval_loader, meta


def evaluate(policy, loader, device):
    policy.set_training_mode(False)
    fire_bit = model_lib.FIRE_BIT.to(device)
    jump_bit = model_lib.JUMP_BIT.to(device)
    correct = total = 0
    fire_hit = fire_n = jump_hit = jump_n = 0

    with torch.no_grad():
        for obs, action, _ret in loader:
            action = action.to(device)
            logits = model_lib.forward_actor(policy, obs, device)
            pred = logits.argmax(-1)

            correct += (pred == action).sum().item()
            total += len(action)
            recall, n = model_lib.button_recall(pred, action, fire_bit)
            fire_hit += recall * n if n else 0
            fire_n += n
            recall, n = model_lib.button_recall(pred, action, jump_bit)
            jump_hit += recall * n if n else 0
            jump_n += n

    return {
        "acc": correct / total,
        "fire_recall": fire_hit / fire_n if fire_n else float("nan"),
        "jump_recall": jump_hit / jump_n if jump_n else float("nan"),
    }


def train(policy, train_loader, eval_loader, args, device, class_weights):
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    class_weights_t = torch.as_tensor(class_weights, device=device)
    n_samples = len(train_loader.dataset)

    for epoch in range(1, args.epochs + 1):
        policy.set_training_mode(True)
        actor_loss_sum = entropy_sum = 0.0

        for obs, action, _ret in train_loader:
            action = action.to(device)
            logits = model_lib.forward_actor(policy, obs, device)

            actor_loss = F.cross_entropy(logits, action, weight=class_weights_t)
            entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
            loss = actor_loss - args.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(action)
            actor_loss_sum += actor_loss.item() * batch_size
            entropy_sum += entropy.item() * batch_size

        metrics = evaluate(policy, eval_loader, device)
        print(
            f"  epoch {epoch:3d}  "
            f"actor_ce {actor_loss_sum / n_samples:6.3f}  "
            f"ent {entropy_sum / n_samples:5.3f}  | "
            f"acc {metrics['acc']:.3f}  "
            f"fire_recall {metrics['fire_recall']:.3f}  "
            f"jump_recall {metrics['jump_recall']:.3f}",
            flush=True,
        )
    return metrics


def main():
    args = parse_args()
    train_loader, eval_loader, meta = make_dataloaders(args)
    device = args.device if torch.cuda.is_available() else "cpu"

    model = model_lib.build_ppo_model(
        stack=meta["stack"],
        resolution=meta["resolution"],
        backbone=args.backbone,
        features_dim=args.features_dim,
        gamma=args.gamma,
        device=device,
    )
    policy = model.policy
    print(
        f"backbone: {args.backbone}  "
        f"features_dim={policy.features_dim}  "
        f"params={model_lib.count_params(policy):,}  "
        f"extractor_params={model_lib.count_params(policy.features_extractor):,}"
    )

    if args.weight_scheme == "none":
        weights = np.ones(ACTION_SPACE.num_actions, dtype=np.float32)
    else:
        weights = model_lib.class_weights(
            meta["actions"],
            ACTION_SPACE.num_actions,
            args.weight_scheme,
        )
    preview = "  ".join(
        f"{ACTION_SPACE.names[i]}={weights[i]:.2f}" for i in np.argsort(-weights)[:6]
    )
    print(f"class weights ({args.weight_scheme}): {preview} ...")

    print(
        f"\nPretraining actor on {device}: "
        f"epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
        f"ent_coef={args.ent_coef} workers={args.num_workers}"
    )
    train(policy, train_loader, eval_loader, args, device, weights)

    out = checkpoint_path(args)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    model.save(out)
    save_config_to_model(
        out,
        stack=meta["stack"],
        train_config={"pretrain": vars(args)},
    )
    print(f"\nSaved warm-start checkpoint -> {out}")
    print(
        "Resume PPO with: "
        f"python3 ppo/train.py --config ppo/train_config/level{args.level}_win.yaml "
        f"--resume {out}"
    )


if __name__ == "__main__":
    main()
