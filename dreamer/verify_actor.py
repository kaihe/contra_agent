"""Component 5 gate — the actor.

Train actor + critic together in imagination on a FROZEN world model, then judge
the actor where it actually matters: the REAL env, run reactively one action at a
time (encode → posterior → π → step). Imagined return is NOT the gate — a
death-blind world model lets the actor look great in dreams and die on the
cartridge, so we measure real progress / reward / survival vs a random baseline.

  GATE: trained actor beats the random policy on real-env progress (the
        imagination gradient points the right way), and imagined return rises.

Honest expectation: the world model here is frozen and trained mostly on expert
traces, so it models "move right + shoot" well but is blind to deaths. Expect the
actor to BEAT random on progress (it learns to advance) yet still die — closing
that gap is C6's job (on-policy data refines the model where the policy goes).

    python -m dreamer.verify_actor --steps 3000 --eval_episodes 8
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from dreamer.actor import Actor, actor_critic_train_step
from dreamer.critic import Critic
from dreamer.collect import trace_paths
from dreamer.envs import make_contra_env, NUM_ACTIONS
from dreamer.verify_reward import _build_buffer
from dreamer.world_model import WorldModel
from dreamer import out_path
from contra.reward import progress_coord


@torch.no_grad()
def eval_real(wm, actor, n_episodes, size, device, greedy=True, max_steps=1200, seed=0):
    """Run a policy reactively in the real env; return per-episode stats.

    actor=None → uniform random baseline. Maintains the RSSM latent across frames
    (closed-loop posterior), exactly the deployment inference loop. Progress and
    reward are tracked ONLINE (from RAM / per-step reward) so an episode that
    never dies — common for a weak policy that just sits near the start — still
    reports its TRUE progress instead of a placeholder zero."""
    env = make_contra_env(level=1, size=size)
    rng = np.random.default_rng(seed)
    out = []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            state = wm.rssm.initial(1, device)
            prev_action = torch.zeros(1, NUM_ACTIONS, device=device)
            start_prog = progress_coord(env.unwrapped.get_ram())
            max_prog, ep_reward, reason, nsteps = start_prog, 0.0, "max_steps", max_steps
            for t in range(max_steps):
                if actor is None:
                    a = int(rng.integers(NUM_ACTIONS))
                else:
                    frame = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    frame = frame.permute(2, 0, 1).unsqueeze(0) / 255.0
                    embed = wm.encoder(frame)
                    state, _ = wm.rssm.obs_step(state, prev_action, embed)
                    logits = actor(wm.rssm.get_feat(state))
                    a = int(logits.argmax(-1)) if greedy else int(Categorical(logits=logits).sample())
                obs, r, term, trunc, info = env.step(a)
                prev_action = F.one_hot(torch.tensor([a], device=device), NUM_ACTIONS).float()
                ep_reward += r
                max_prog = max(max_prog, progress_coord(env.unwrapped.get_ram()))
                if term or trunc:
                    reason = info.get("episode_end_reason", "?"); nsteps = t + 1
                    break
            out.append((float(max_prog - start_prog), ep_reward, reason, nsteps))
    finally:
        env.close()
    return out


def _summ(tag, rows):
    prog = np.mean([r[0] for r in rows]); rew = np.mean([r[1] for r in rows])
    steps = np.mean([r[3] for r in rows])
    reasons = {}
    for r in rows:
        reasons[r[2]] = reasons.get(r[2], 0) + 1
    print(f"  {tag:16s} progress {prog:7.1f}  reward {rew:8.2f}  steps {steps:6.0f}  {reasons}")
    return prog


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="tmp/dreamer/world_model.pt")
    p.add_argument("--train_traces", type=int, default=8)
    p.add_argument("--env_steps", type=int, default=6000)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--imag_batch", type=int, default=256)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--deter", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent_coef", type=float, default=3e-3)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    import os
    wm = WorldModel(size=args.size, deter=args.deter).to(device)
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"no world model at {args.ckpt} — run `python -m dreamer.verify_rssm` first.")
    wm.load_state_dict(torch.load(args.ckpt, map_location=device))
    wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    print(f"[5] loaded frozen world model ← {args.ckpt}")

    train_paths = trace_paths(1)[: args.train_traces]
    train_buf = _build_buffer(train_paths, args.env_steps, args.size, args.seq_len, device, args.seed)
    print(f"[5] train={train_buf.size}  horizon={args.horizon} γ={args.gamma} "
          f"λ={args.lam} ent_coef={args.ent_coef}")

    actor = Actor(wm.feat_dim, NUM_ACTIONS).to(device)
    critic = Critic(wm.feat_dim, gamma=args.gamma, lam=args.lam).to(device)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_critic = torch.optim.Adam(critic.net.parameters(), lr=args.critic_lr)

    # baseline FIRST (env closed before training; one emulator at a time) ────────
    print("[5] random-policy baseline (real env):")
    base = eval_real(wm, None, args.eval_episodes, args.size, device, seed=args.seed)
    base_prog = _summ("RANDOM", base)

    history = []
    for step in range(1, args.steps + 1):
        m = actor_critic_train_step(wm, actor, critic, opt_actor, opt_critic,
                                    train_buf.sample(args.batch), args.horizon,
                                    args.ent_coef, args.imag_batch)
        if step % 200 == 0 or step == 1:
            print(f"  step {step:5d}  actor {m['actor_loss']:+.4f}  critic {m['critic_loss']:.4f}  "
                  f"H(π) {m['entropy']:.3f}  imag_return {m['return']:.2f}")
            history.append((step, m["return"], m["entropy"]))
        if step % args.eval_every == 0:
            rows = eval_real(wm, actor, args.eval_episodes, args.size, device, seed=args.seed)
            _summ(f"actor@{step}", rows)

    print("\n[5] final actor (real env):")
    final = eval_real(wm, actor, args.eval_episodes * 2, args.size, device, seed=args.seed + 100)
    final_prog = _summ("ACTOR", final)

    # imagined-return trend plot ────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hs, rs, es = zip(*history)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hs, rs, color="tab:blue", label="imagined return")
    ax.set_xlabel("train step"); ax.set_ylabel("imagined λ-return", color="tab:blue")
    ax2 = ax.twinx(); ax2.plot(hs, es, color="tab:orange", alpha=0.7, label="entropy")
    ax2.set_ylabel("policy entropy", color="tab:orange")
    ax.set_title(f"actor imagination training (random prog={base_prog:.0f} → actor prog={final_prog:.0f})")
    fig.tight_layout()
    pth = out_path("actor_training.png")
    fig.savefig(pth, dpi=90)
    print(f"  trend → {pth}")
    verdict = "PASS" if final_prog > base_prog * 1.2 else "WEAK"
    print(f"\n  GATE [{verdict}]: actor progress {final_prog:.1f} vs random {base_prog:.1f} "
          f"(want actor clearly higher). Imagined return should also rise over training.")


if __name__ == "__main__":
    main()
