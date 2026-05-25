"""
GRPO post-training for ContraVLA.

Algorithm
---------
Group Relative Policy Optimisation (GRPO) fine-tunes the BC-initialised VLA
policy with online environment interaction.  It eliminates the critic network by
using G parallel rollouts from the same emulator state as an implicit baseline.

Outer loop
----------
For each iteration:
  1. Collect G rollouts from the same
     starting state.  For each group:
       a. Reset env, snapshot emulator state.
       b. For each of the G members:
            - restore snapshot, run for max_steps (or until done)
            - record transitions (obs, chunk, log_prob_old, log_prob_ref)
            - record discounted return
       c. Compute group-normalised advantages.
  2. Update the policy for  update_epochs  epochs on the rollout buffer using
     the clipped surrogate loss + approximate KL penalty.
  3. Log metrics to TensorBoard and save checkpoint.

Usage
-----
    python -m vla.post_training.grpo --config vla/post_training/grpo.yaml
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from vla.model import ContraVLA, ContraVLAConfig
from vla.datasets.dataset import LEVEL_TEXTS
from vla.post_training.env_wrappers import VLAEnv
from vla.post_training.rollout_buffer import RolloutBuffer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_model(checkpoint_path: str, device: torch.device,
                dropout: float = 0.0) -> ContraVLA:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model"]
    sd   = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
    cfg  = ContraVLAConfig(dropout=dropout)
    mdl  = ContraVLA(cfg)
    mdl.load_state_dict(sd, strict=True)
    return mdl.to(device)


def _tokenize(tokenizer, text: str, max_len: int, device: torch.device) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", padding="max_length",
                    max_length=max_len, truncation=True)
    return enc["input_ids"].to(device)   # [1, L]


@torch.no_grad()
def _chunk_logprob(
    model:     ContraVLA,
    input_ids: torch.Tensor,         # [1, L]
    images:    torch.Tensor,         # [1, 2, 3, H, W]
    proprio:   torch.Tensor,         # [1, 118]
    actions:   torch.Tensor,         # [1, T]  int64
) -> float:
    """Return sum_{t} log π(a_t | s) for one (obs, chunk) pair."""
    vlm_feats = model.forward_vlm_efficient(images, input_ids)
    logits    = model.action_transformer(vlm_feats, proprio)     # [1, T, 36]
    lp        = F.log_softmax(logits, dim=-1)                    # [1, T, 36]
    lp_acts   = lp.gather(-1, actions.unsqueeze(-1)).squeeze(-1) # [1, T]
    return lp_acts.sum().item()


@torch.no_grad()
def _sample_chunk(
    model:     ContraVLA,
    input_ids: torch.Tensor,    # [1, L]
    images:    torch.Tensor,    # [1, 2, 3, H, W]
    proprio:   torch.Tensor,    # [1, 118]
) -> tuple[torch.Tensor, float]:
    """
    Sample one action chunk from the policy.

    Returns
    -------
    actions  : [T] int64  (on CPU)
    log_prob : float      sum_{t} log π(a_t | s)
    """
    vlm_feats = model.forward_vlm_efficient(images, input_ids)
    logits    = model.action_transformer(vlm_feats, proprio)  # [1, T, 36]
    dist      = Categorical(logits=logits[0])                 # [T, 36]
    actions   = dist.sample()                                 # [T]
    log_prob  = dist.log_prob(actions).sum().item()           # scalar
    return actions.cpu(), log_prob


# ── GRPOTrainer ───────────────────────────────────────────────────────────────

class GRPOTrainer:
    """
    Attributes set from config
    --------------------------
    G                : int    group size (rollouts per starting state)
    T                : int    action chunk length
    gamma            : float  discount factor
    clip_eps         : float  PPO clipping epsilon
    beta_kl          : float  KL penalty coefficient
    adv_clip         : float  advantage clipping
    max_steps        : int    max agent steps per episode rollout
    update_epochs    : int    policy update passes over the buffer
    batch_size       : int    minibatch size
    lr               : float  learning rate
    weight_decay     : float
    grad_clip        : float  gradient norm clip
    max_num_rollout  : int    total episodes to collect before stopping
    """

    def __init__(self, cfg: dict, out_dir: str) -> None:
        self.cfg     = cfg
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # ── policy (trainable) ──
        print(f"Loading BC checkpoint: {cfg['load_bc']}")
        self.policy = _load_model(cfg["load_bc"], self.device, dropout=0.0)
        self.policy.train()
        if cfg.get("freeze_vlm", False):
            self.policy.freeze_vlm()
            print("  VLM frozen — training action transformer only")
        if cfg.get("grad_checkpoint", False) and hasattr(self.policy, "gradient_checkpointing_enable"):
            self.policy.gradient_checkpointing_enable()
            print("  Gradient checkpointing enabled")

        # ── reference model (frozen, CPU) ──
        print("Loading reference model (frozen, CPU) …")
        self.ref = _load_model(cfg["load_bc"], torch.device("cpu"), dropout=0.0).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)

        # ── tokenizer + fixed input_ids for Level1 ──
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.policy.config.vlm_model_name
        )
        self.input_ids = _tokenize(
            self.tokenizer, LEVEL_TEXTS[0], max_len=32, device=self.device
        )  # [1, L]
        self.input_ids_cpu = self.input_ids.cpu()

        # ── env ──
        self.env = VLAEnv(level="Level1")

        # ── optimiser ──
        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=cfg.get("lr", 2e-5),
            weight_decay=cfg.get("weight_decay", 1e-2),
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

        # ── tensorboard ──
        log_dir = cfg.get("log_dir", f"tmp/tf_logs/{cfg.get('name', 'grpo')}")
        self.writer = SummaryWriter(log_dir)

        # ── hyperparameters ──
        self.G                 = cfg.get("G",                 4)
        self.T                 = self.policy.config.num_actions   # chunk size from BC
        self.K                 = cfg.get("K",                 48)  # fixed action horizon
        self.gamma             = cfg.get("gamma",             0.99)
        self.clip_eps          = cfg.get("clip_eps",          0.2)
        self.beta_kl           = cfg.get("beta_kl",           0.05)
        self.adv_clip          = cfg.get("adv_clip",          5.0)
        self.update_epochs     = cfg.get("update_epochs",     4)
        self.batch_size        = cfg.get("batch_size",        24)
        self.grad_clip         = cfg.get("grad_clip",         1.0)
        self.max_num_rollout   = cfg.get("max_num_rollout",   6400)

        self.global_step = 0

    # ── rollout collection ────────────────────────────────────────────────────

    def _rollout_member(
        self, obs: dict
    ) -> tuple[list[dict], list[list[int]], float, str]:
        """
        Run one group member for K actions from the current env state.

        Returns
        -------
        transitions : list of kwarg dicts for buf.add_transition()
        chunk_acts  : list of action chunks [[a0,a1], ...] — used for commit replay
        total_return: discounted return over the rollout
        status      : "" | "DEAD" | "DONE"
        """
        transitions: list[dict]       = []
        chunk_acts:  list[list[int]]  = []
        total_return = 0.0
        gamma_t      = 1.0
        status       = VLAEnv.RUNNING

        for _ in range(self.K // self.T):
            images  = obs["images"].unsqueeze(0).to(self.device)
            proprio = obs["proprio"].unsqueeze(0).to(self.device)

            actions, lp_old = _sample_chunk(
                self.policy, self.input_ids, images, proprio
            )
            lp_ref = _chunk_logprob(
                self.ref,
                self.input_ids_cpu,
                obs["images"].unsqueeze(0),
                obs["proprio"].unsqueeze(0),
                actions.unsqueeze(0),
            )

            new_obs, reward, status = self.env.step(actions.tolist())

            total_return += gamma_t * reward
            gamma_t      *= self.gamma ** self.T

            transitions.append(dict(
                images       = obs["images"].numpy(),
                proprio      = obs["proprio"].numpy(),
                actions      = actions.numpy().astype(np.int64),
                log_prob_old = lp_old,
                log_prob_ref = lp_ref,
            ))
            chunk_acts.append(actions.tolist())

            obs = new_obs
            if status:
                break

        return transitions, chunk_acts, total_return, status

    def collect_rollouts(self) -> RolloutBuffer:
        """
        Collect G episodes from the same committed state, advancing a committed
        state through the level — identical to MC search's commit strategy.

        For each group:
          1. Run G members from the committed emulator state.
          2. Pick the member with the highest return as the best.
          3. Commit: replay the best member's actions from the committed state
             to advance it K steps forward.
          4. If the best member died, reset committed state to Level1.
        """
        buf      = RolloutBuffer()
        self.policy.eval()

        # committed state starts at Level1 and walks forward through the level
        self.env.reset()
        emu_state = self.env.snapshot()
        t0 = time.time()

        # single group: all G members branch from the same committed state
        members: list[tuple[list, list, float, str]] = []

        for g in range(self.G):
            obs = self.env.restore(emu_state)
            transitions, chunk_acts, total_return, status = self._rollout_member(obs)
            members.append((transitions, chunk_acts, total_return, status))
            elapsed = time.time() - t0
            print(
                f"  collect  member {g + 1:3d}/{self.G}"
                f"  {status or 'MAX '}"
                f"  chunks={len(chunk_acts):4d}"
                f"  ret={total_return:8.1f}"
                f"  {elapsed:.0f}s",
                flush=True,
            )

        # ── add all members to buffer (group_id=0) ──
        for transitions, _, total_return, _ in members:
            buf.begin_episode(0)
            for tr in transitions:
                buf.add_transition(**tr)
            buf.end_episode(total_return)

        # ── commit best member (highest return) ──
        best_g = max(range(self.G), key=lambda i: members[i][2])
        best_chunks, best_return, best_status = (
            members[best_g][1], members[best_g][2], members[best_g][3]
        )

        if best_status == VLAEnv.DEAD:
            commit_note = "RESET (best died)"
        else:
            self.env.restore(emu_state)
            for chunk in best_chunks:
                _, _, step_status = self.env.step(chunk)
                if step_status:
                    break
            commit_note = f"committed member {best_g + 1}  ret={best_return:.1f}"

        elapsed = time.time() - t0
        print(f"  commit   {commit_note}  {elapsed:.0f}s", flush=True)

        buf.compute_advantages(adv_clip=self.adv_clip)
        return buf

    # ── policy update ─────────────────────────────────────────────────────────

    def update_policy(self, buf: RolloutBuffer) -> dict[str, float]:
        """
        Run update_epochs passes over the rollout buffer.

        Returns a dict of scalar metrics averaged over all update steps.
        """
        self.policy.train()

        metrics = dict(loss=[], loss_clip=[], loss_kl=[], ratio=[], kl_approx=[], grad_norm=[])

        n_batches_per_epoch = max(1, buf.n_transitions // self.batch_size)
        t0 = time.time()
        update_step = 0

        for epoch in range(self.update_epochs):
            for batch in buf.batches(self.batch_size, self.device):
                B = batch["images"].shape[0]

                # tile input_ids to batch size
                input_ids_b = self.input_ids.expand(B, -1)  # [B, L]

                with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                    enabled=self.device.type == "cuda"):
                    # forward policy
                    vlm_feats = self.policy.forward_vlm_efficient(
                        batch["images"], input_ids_b
                    )
                    logits = self.policy.action_transformer(
                        vlm_feats, batch["proprio"]
                    )                                          # [B, T, 36]

                    # log π_θ(a | s)  for the stored actions
                    lp_new = (
                        F.log_softmax(logits, dim=-1)
                        .gather(-1, batch["actions"].unsqueeze(-1))
                        .squeeze(-1)
                        .sum(-1)                              # [B]
                    )

                    # importance ratio
                    ratio = torch.exp(lp_new - batch["log_prob_old"])  # [B]

                    adv = batch["advantages"]                  # [B]

                    # clipped surrogate loss
                    loss_clip = -torch.min(
                        ratio * adv,
                        ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv,
                    ).mean()

                    # approximate KL: E[log π_θ(a) - log π_ref(a)]
                    kl_approx = (lp_new - batch["log_prob_ref"]).mean()

                    loss = loss_clip + self.beta_kl * kl_approx

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.grad_clip
                ).item()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                metrics["loss"].append(loss.item())
                metrics["loss_clip"].append(loss_clip.item())
                metrics["loss_kl"].append(kl_approx.item())
                metrics["ratio"].append(ratio.mean().item())
                metrics["kl_approx"].append(kl_approx.item())
                metrics["grad_norm"].append(grad_norm)

                self.global_step += 1
                update_step      += 1

                if update_step % 10 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"  update  epoch {epoch + 1}/{self.update_epochs}"
                        f"  step {update_step:4d}/{self.update_epochs * n_batches_per_epoch}"
                        f"  loss {loss.item():.4f}"
                        f"  clip {loss_clip.item():.4f}"
                        f"  kl {kl_approx.item():.4f}"
                        f"  ratio {ratio.mean().item():.3f}"
                        f"  grad {grad_norm:.2f} ({grad_norm / self.grad_clip:.2f}x)"
                        f"  {elapsed:.0f}s",
                        flush=True,
                    )

                self.writer.add_scalar("update/loss",      loss.item(),      self.global_step)
                self.writer.add_scalar("update/loss_clip", loss_clip.item(), self.global_step)
                self.writer.add_scalar("update/kl_approx", kl_approx.item(),self.global_step)
                self.writer.add_scalar("update/ratio",     ratio.mean().item(), self.global_step)
                self.writer.add_scalar("update/grad_norm", grad_norm,        self.global_step)

        return {k: float(np.mean(v)) for k, v in metrics.items() if v}

    # ── checkpoint ────────────────────────────────────────────────────────────

    def _save(self, iteration: int, tag: str = "last") -> None:
        path = os.path.join(self.out_dir, f"{tag}.pt")
        torch.save({
            "iteration": iteration,
            "model":     self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    # ── main training loop ────────────────────────────────────────────────────

    def train(self) -> None:
        print(f"\nGRPO training  G={self.G}  T={self.T}  K={self.K}"
              f"  budget={self.max_num_rollout} rollouts")
        print(f"  device={self.device}  out={self.out_dir}\n")

        best_mean_return = -float("inf")
        total_rollouts   = 0
        it               = 0
        iter_times: list[float] = []

        while total_rollouts < self.max_num_rollout:
            it += 1
            t0  = time.time()
            remaining = self.max_num_rollout - total_rollouts
            print(f"\n{'─' * 70}")
            print(f"Iter {it}  rollouts {total_rollouts}/{self.max_num_rollout}"
                  f"  collecting G={self.G} episodes"
                  f"  K={self.K} actions ({self.K // self.T} chunks) each …")

            # ── collect ──
            buf = self.collect_rollouts()
            stats = buf.return_stats()
            total_rollouts += buf.n_episodes
            dt_collect = time.time() - t0

            # ── update ──
            t1 = time.time()
            print(f"Iter {it}  updating policy"
                  f" ({self.update_epochs} epochs × {max(1, buf.n_transitions // self.batch_size)} batches) …")
            update_metrics = self.update_policy(buf)
            dt_update = time.time() - t1

            # ── log ──
            dt_iter = time.time() - t0
            iter_times.append(dt_iter)
            remaining_after = self.max_num_rollout - total_rollouts
            eta_s   = remaining_after / self.G * (sum(iter_times) / len(iter_times))
            eta_str = f"{eta_s / 3600:.1f}h" if eta_s >= 3600 else f"{eta_s / 60:.0f}m"

            mean_ret = stats.get("mean", 0.0)
            print(
                f"{'─' * 70}\n"
                f"[iter {it}  rollouts {total_rollouts}/{self.max_num_rollout}]"
                f"  eps={buf.n_episodes}  trans={buf.n_transitions}"
                f"  ret={mean_ret:8.1f} ± {stats.get('std', 0):.1f}"
                f"  [min={stats.get('min', 0):.0f}  max={stats.get('max', 0):.0f}]\n"
                f"  loss={update_metrics.get('loss', 0):.4f}"
                f"  clip={update_metrics.get('loss_clip', 0):.4f}"
                f"  kl={update_metrics.get('kl_approx', 0):.4f}"
                f"  ratio={update_metrics.get('ratio', 0):.3f}"
                f"  grad={update_metrics.get('grad_norm', 0):.2f}"
                f"  collect={dt_collect:.0f}s  update={dt_update:.0f}s  eta={eta_str}"
            )

            self.writer.add_scalar("rollout/mean_return",   mean_ret,                            total_rollouts)
            self.writer.add_scalar("rollout/std_return",    stats.get("std", 0),                 total_rollouts)
            self.writer.add_scalar("rollout/min_return",    stats.get("min", 0),                 total_rollouts)
            self.writer.add_scalar("rollout/max_return",    stats.get("max", 0),                 total_rollouts)
            self.writer.add_scalar("rollout/n_transitions", buf.n_transitions,                   total_rollouts)
            self.writer.add_scalar("train/loss",            update_metrics.get("loss", 0),       total_rollouts)
            self.writer.add_scalar("train/loss_clip",       update_metrics.get("loss_clip", 0),  total_rollouts)
            self.writer.add_scalar("train/kl_approx",       update_metrics.get("kl_approx", 0), total_rollouts)
            self.writer.add_scalar("train/ratio",           update_metrics.get("ratio", 0),      total_rollouts)

            # ── checkpointing ──
            self._save(total_rollouts, tag="last")
            if mean_ret > best_mean_return:
                best_mean_return = mean_ret
                self._save(total_rollouts, tag="best")
                print(f"  ↳ best return {best_mean_return:.1f}")

            buf.clear()

        self.writer.close()
        self.env.close()
        print(f"Training complete — {total_rollouts} rollouts collected.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="GRPO post-training for ContraVLA")
    p.add_argument("--config", default="vla/post_training/grpo.yaml")
    p.add_argument("--out",    default=None,
                   help="Output directory for checkpoints (overrides config name)")
    # override any config key via CLI, e.g. --lr 1e-5 --G 8
    p.add_argument("--G",                 type=int,   default=None)
    p.add_argument("--lr",                type=float, default=None)
    p.add_argument("--max_num_rollout",   type=int,   default=None)
    p.add_argument("--batch_size",        type=int,   default=None)
    p.add_argument("--device",            default=None)
    p.add_argument("--load_bc",           default=None)
    args = p.parse_args()

    cfg = _load_config(args.config)

    # CLI overrides
    for key in ("G", "lr", "max_num_rollout", "batch_size", "device", "load_bc"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    out_dir = args.out or f"tmp/vla/{cfg.get('name', 'grpo')}"

    trainer = GRPOTrainer(cfg=cfg, out_dir=out_dir)
    trainer.train()


if __name__ == "__main__":
    main()
