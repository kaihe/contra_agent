"""Component 3c gate — do the reward + continue heads predict truth?

Trains the full world model (dynamics + reward + continue) on a mix of whole-level
traces (rich reward, levelup terminals) and env rollouts (deaths → cont=0), then
on held-out data checks:
  reward   — predicted vs true correlation (Pearson r) + scatter plot,
  continue — AUC for flagging terminals, and mean P(continue) at terminal vs
             surviving steps (should be ~0 vs ~1).

This is also the implicit test of whether the task gradient pulls entities into
the latent: reward needs progress (player x), continue needs death (enemy contact),
so heads can only succeed if the latent tracks them.

    python -m dreamer.verify_reward --train_traces 8 --env_steps 6000 --steps 6000
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from dreamer.buffer import ReplayBuffer, _fill_from_env
from dreamer.collect import fill_buffer_from_traces, trace_paths
from dreamer.world_model import WorldModel


def _build_buffer(paths, env_steps, size, seq_len, device, seed):
    lengths = [len(np.load(p, allow_pickle=True)["actions"]) for p in paths]
    buf = ReplayBuffer(capacity=sum(lengths) + env_steps + 32,
                       obs_shape=(size, size, 3), num_actions=21,
                       seq_len=seq_len, device=device)
    fill_buffer_from_traces(buf, paths, verbose=False)
    if env_steps:
        _fill_from_env(buf, env_steps, seed)
    return buf


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-8))


def _load_trace_ordered(path, size, device):
    """One whole trace as ordered (frames, onehot actions, true reward)."""
    n = len(np.load(path, allow_pickle=True)["actions"])
    buf = ReplayBuffer(capacity=n + 4, obs_shape=(size, size, 3),
                       num_actions=21, seq_len=2, device=device)
    fill_buffer_from_traces(buf, [path], verbose=False)
    N = buf.size
    frames = torch.as_tensor(buf.image[:N], dtype=torch.float32, device=device) / 255.0
    onehot = torch.nn.functional.one_hot(
        torch.as_tensor(buf.action[:N], dtype=torch.long, device=device), 21).float()
    return frames, onehot, buf.reward[:N].copy()


def _extract_death(eval_buf, device):
    """First death episode (negative terminal) from the buffer, in order."""
    N = eval_buf.size
    deaths = [t for t in np.where(eval_buf.is_terminal[:N])[0] if eval_buf.reward[t] < 0]
    if not deaths:
        return None
    ti = deaths[0]
    start = ti
    while start > 0 and not eval_buf.is_first[start]:
        start -= 1
    frames = torch.as_tensor(eval_buf.image[start:ti + 1], dtype=torch.float32, device=device) / 255.0
    onehot = torch.nn.functional.one_hot(
        torch.as_tensor(eval_buf.action[start:ti + 1], dtype=torch.long, device=device), 21).float()
    return frames, onehot, eval_buf.reward[start:ti + 1].copy()


def _predict_episode_reward(wm, frames, onehot, device):
    """Teacher-forced reward-head prediction over an ordered episode (chunked
    encoding so a 1600-step trace doesn't OOM)."""
    T = len(frames)
    embeds = torch.cat([wm.encoder(frames[j:j + 128].permute(0, 3, 1, 2))
                        for j in range(0, T, 128)], 0)
    first = torch.zeros(1, T, device=device); first[0, 0] = 1.0
    with torch.no_grad():
        posts, _ = wm.rssm.observe(embeds.unsqueeze(0), onehot.unsqueeze(0), first)
        r_pred, _ = wm.predict_heads(wm.rssm.get_feat(posts))
    return r_pred[0].cpu().numpy()


def _auc(scores, labels):
    """Rank-based AUC; labels: 1 = positive (terminal)."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    npos, nneg = labels.sum(), (1 - labels).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_traces", type=int, default=8)
    p.add_argument("--env_steps", type=int, default=6000)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--deter", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    all_paths = trace_paths(1)
    train_paths = all_paths[: args.train_traces]
    eval_paths = all_paths[args.train_traces: args.train_traces + 2]
    print(f"[3c] device={device} loading buffers (traces + env deaths)…")
    train_buf = _build_buffer(train_paths, args.env_steps, args.size, args.seq_len, device, args.seed)
    eval_buf = _build_buffer(eval_paths, args.env_steps // 3, args.size, args.seq_len, device, args.seed + 1)
    print(f"[3c] train={train_buf.size} ({int(train_buf.is_terminal.sum())} terminals)  "
          f"eval={eval_buf.size} ({int(eval_buf.is_terminal.sum())} terminals)")

    wm = WorldModel(size=args.size, deter=args.deter).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        loss, m, _ = wm.loss(train_buf.sample(args.batch))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()
        if step % 500 == 0 or step == 1:
            print(f"  step {step:5d}  recon {m['recon']:.1f}  kl {m['kl']:.2f}  "
                  f"reward {m['reward']:.4f}  cont {m['cont']:.4f}")

    # ── eval: predicted vs true reward / continue on held-out sequences ───────
    wm.eval()
    rp, rt, cp, ct = [], [], [], []
    with torch.no_grad():
        for _ in range(8):
            b = eval_buf.sample(32)
            posts, _ = wm.observe(b["image"], b["action"], b["is_first"])
            feat = wm.rssm.get_feat(posts)
            r_pred, c_pred = wm.predict_heads(feat)
            rp.append(r_pred.flatten().cpu()); rt.append(b["reward"].flatten().cpu())
            cp.append(c_pred.flatten().cpu()); ct.append(b["cont"].flatten().cpu())
    rp, rt = torch.cat(rp).numpy(), torch.cat(rt).numpy()
    cp, ct = torch.cat(cp).numpy(), torch.cat(ct).numpy()

    r_pearson = _pearson(rp, rt)
    is_term = (ct == 0).astype(float)
    auc = _auc(1 - cp, is_term)                      # score = P(terminal) = 1-P(cont)
    p_cont_term = cp[is_term == 1].mean() if is_term.sum() else float("nan")
    p_cont_surv = cp[is_term == 0].mean()

    print("\n  REWARD head:")
    print(f"    predicted-vs-true Pearson r = {r_pearson:.3f}  "
          f"(true reward range [{rt.min():.2f},{rt.max():.2f}])")
    print("  CONTINUE head:")
    print(f"    terminal-detection AUC = {auc:.3f}   ({int(is_term.sum())} terminal steps)")
    print(f"    mean P(continue): surviving={p_cont_surv:.3f}  terminal={p_cont_term:.3f}  "
          f"(want high vs low)")

    # reward scatter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dreamer import out_path
    plt.figure(figsize=(5, 5))
    plt.scatter(rt, rp, s=4, alpha=0.3)
    lo, hi = min(rt.min(), rp.min()), max(rt.max(), rp.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=1)
    plt.xlabel("true reward"); plt.ylabel("predicted reward")
    plt.title(f"reward head (r={r_pearson:.3f})"); plt.tight_layout()
    sp = out_path("reward_scatter.png")
    plt.savefig(sp, dpi=90)
    print(f"\n  scatter → {sp}")

    # ── per-step true vs predicted reward, one WIN trace + one LOSE episode ────
    win = _load_trace_ordered(eval_paths[0], args.size, device)
    lose = _extract_death(eval_buf, device)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    for ax, ep, name in [(axes[0], win, "WIN trace"), (axes[1], lose, "LOSE episode")]:
        if ep is None:
            ax.set_title(f"{name}: none found"); continue
        frames, onehot, true_r = ep
        pred_r = _predict_episode_reward(wm, frames, onehot, device)
        x = np.arange(len(true_r))
        ax.plot(x, true_r, color="tab:blue", lw=0.8, label="true reward")
        ax.plot(x, pred_r, color="tab:orange", lw=0.8, alpha=0.85, label="predicted")
        ax.axhline(0, color="0.8", lw=0.5)
        ax.set_yscale("symlog", linthresh=0.1)       # tiny progress + big spikes both visible
        ax.set_title(f"{name}  (r={_pearson(pred_r, true_r):.3f}, {len(true_r)} steps)", fontsize=10)
        ax.set_xlabel("step"); ax.set_ylabel("reward")
        ax.legend(fontsize=8); ax.tick_params(labelsize=7)
    fig.tight_layout()
    tp = out_path("reward_trace.png")
    fig.savefig(tp, dpi=90)
    print(f"  per-episode true-vs-pred → {tp}")
    print("  GATE: reward r well above 0, AUC near 1, P(cont) high on survive / low on terminal.")


if __name__ == "__main__":
    main()
