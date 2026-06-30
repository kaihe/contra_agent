"""Component 4 gate — the critic + λ-returns.

Two checks, cheap → expensive:

1. UNIT TEST the λ-return math against hand-worked cases (deterministic, no
   training): λ=1 collapses to the discounted return-to-go; λ=0 is a 1-step
   bootstrap; a terminal (cont=0) cuts the future. If these fail, nothing
   downstream can be trusted.

2. BEHAVIORAL — load the frozen world model, train the critic purely in
   imagination (real actions = the data policy, no actor yet), then on a held-out
   WIN trace and a DEATH episode compare the learned value v(s_t) to the true
   Monte-Carlo discounted return-to-go G_t computed from REAL rewards. The critic
   should track G: high early (lots of progress ahead), declining toward the
   boss, and dropping sharply just before a death.

    python -m dreamer.verify_critic --steps 4000
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from dreamer.critic import Critic, critic_train_step, lambda_return
from dreamer.collect import trace_paths
from dreamer.verify_reward import (_build_buffer, _extract_death,
                                    _load_trace_ordered, _pearson)
from dreamer.world_model import WorldModel
from dreamer import out_path


# ── 1. λ-return unit tests ────────────────────────────────────────────────────

def _unit_tests() -> bool:
    t = lambda x: torch.tensor([x], dtype=torch.float32)
    ok = True

    # Case A — λ=1, cont=1, γ=0.5: collapses to discounted return-to-go.
    #   ret[2]=1 ; ret[1]=1+0.5·1=1.5 ; ret[0]=1+0.5·1.5=1.75
    ret = lambda_return(t([1., 1., 1.]), t([9., 9., 9.]),       # value ignored at λ=1
                        t([0.5, 0.5, 0.5]), torch.tensor([0.]), lam=1.0)
    expect = [1.75, 1.5, 1.0]
    ok &= _check("A λ=1 return-to-go", ret[0].tolist(), expect)

    # Case B — λ=0, γ=0.5, value≡2: pure 1-step bootstrap r+γ·v = 1+0.5·2 = 2.
    ret = lambda_return(t([1., 1., 1.]), t([2., 2., 2.]),
                        t([0.5, 0.5, 0.5]), torch.tensor([2.]), lam=0.0)
    ok &= _check("B λ=0 one-step TD", ret[0].tolist(), [2.0, 2.0, 2.0])

    # Case C — terminal at state 2 (pcont[1]=0) cuts the future there.
    #   λ=1: ret[2]=1 ; ret[1]=1+0·…=1 ; ret[0]=1+0.5·1=1.5
    ret = lambda_return(t([1., 1., 1.]), t([9., 9., 9.]),
                        t([0.5, 0.0, 0.5]), torch.tensor([0.]), lam=1.0)
    ok &= _check("C terminal cuts future", ret[0].tolist(), [1.5, 1.0, 1.0])

    print(f"[unit] λ-return math: {'PASS' if ok else 'FAIL'}")
    return ok


def _check(name, got, expect, tol=1e-5) -> bool:
    good = all(abs(a - b) < tol for a, b in zip(got, expect))
    flag = "ok " if good else "BAD"
    print(f"  [{flag}] {name}: got {[round(x,4) for x in got]} want {expect}")
    return good


# ── 2. behavioral helpers ─────────────────────────────────────────────────────

def _critic_values(critic, wm, frames, onehot, device):
    """Closed-loop value v(s_t) at every state of an ordered episode."""
    T = len(frames)
    embeds = torch.cat([wm.encoder(frames[j:j + 128].permute(0, 3, 1, 2))
                        for j in range(0, T, 128)], 0)
    first = torch.zeros(1, T, device=device); first[0, 0] = 1.0
    with torch.no_grad():
        posts, _ = wm.rssm.observe(embeds.unsqueeze(0), onehot.unsqueeze(0), first)
        v = critic.value(wm.rssm.get_feat(posts))
    return v[0].cpu().numpy()


def _return_to_go(reward, gamma):
    """True discounted future reward at each state: G_t = Σ_{k>t} γ^{k-t-1} r_k.

    Uses the reward-INTO-state convention (reward[i] is collected entering s_i),
    so v(s_t) excludes the reward already banked at s_t — matching the critic."""
    G = np.zeros_like(reward, dtype=np.float64)
    for t in range(len(reward) - 2, -1, -1):
        G[t] = reward[t + 1] + gamma * G[t + 1]
    return G


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="tmp/dreamer/world_model.pt")
    p.add_argument("--train_traces", type=int, default=8)
    p.add_argument("--env_steps", type=int, default=6000)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--context", type=int, default=5)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--deter", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not _unit_tests():
        raise SystemExit("λ-return unit tests FAILED — fix the math before training.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # frozen world model (reward + continue heads already trained in C3b/C3c) ----
    wm = WorldModel(size=args.size, deter=args.deter).to(device)
    import os
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"no world model at {args.ckpt} — run `python -m dreamer.verify_rssm` first.")
    wm.load_state_dict(torch.load(args.ckpt, map_location=device))
    wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    print(f"[4] loaded frozen world model ← {args.ckpt}")

    all_paths = trace_paths(1)
    train_paths = all_paths[: args.train_traces]
    eval_paths = all_paths[args.train_traces: args.train_traces + 2]
    train_buf = _build_buffer(train_paths, args.env_steps, args.size, args.seq_len, device, args.seed)
    eval_buf = _build_buffer(eval_paths, args.env_steps // 3, args.size, args.seq_len, device, args.seed + 1)
    print(f"[4] train={train_buf.size} eval={eval_buf.size}  "
          f"context={args.context} horizon={args.horizon} γ={args.gamma} λ={args.lam}")

    critic = Critic(wm.feat_dim, gamma=args.gamma, lam=args.lam).to(device)
    opt = torch.optim.Adam(critic.net.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        loss, ret = critic_train_step(wm, critic, opt, train_buf.sample(args.batch),
                                      args.context, args.horizon)
        if step % 500 == 0 or step == 1:
            print(f"  step {step:5d}  critic_loss {loss:.4f}  "
                  f"λret[mean {ret.mean():.2f} min {ret.min():.2f} max {ret.max():.2f}]")

    # ── value vs. true return-to-go on held-out WIN trace + DEATH episode ──────
    critic.eval()
    win = _load_trace_ordered(eval_paths[0], args.size, device)
    lose = _extract_death(eval_buf, device)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    summary = []
    for ax, ep, name in [(axes[0], win, "WIN trace"), (axes[1], lose, "DEATH episode")]:
        if ep is None:
            ax.set_title(f"{name}: none found"); continue
        frames, onehot, true_r = ep
        v_pred = _critic_values(critic, wm, frames, onehot, device)
        G = _return_to_go(true_r, args.gamma)
        r = _pearson(v_pred, G)
        summary.append((name, r))
        x = np.arange(len(G))
        ax.plot(x, G, color="tab:blue", lw=1.0, label="true return-to-go G_t")
        ax.plot(x, v_pred, color="tab:orange", lw=1.0, alpha=0.85, label="critic v(s_t)")
        ax.axhline(0, color="0.8", lw=0.5)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(f"{name}  (Pearson r={r:.3f}, {len(G)} steps)", fontsize=10)
        ax.set_xlabel("step"); ax.set_ylabel("value")
        ax.legend(fontsize=8); ax.tick_params(labelsize=7)
    fig.tight_layout()
    cp = out_path("critic_value.png")
    fig.savefig(cp, dpi=90)
    print(f"\n  critic-value vs return-to-go → {cp}")
    for name, r in summary:
        print(f"    {name:14s} Pearson r(v, G) = {r:.3f}")
    print("  GATE: v tracks G (r well above 0); declines toward the boss / drops before death.")


if __name__ == "__main__":
    main()
