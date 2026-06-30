"""Component 6 — the full DreamerV3 training loop on Contra level 1.

Interleaves the three things the earlier gates verified in isolation:

  1. COLLECT   run the actor in the REAL env (stochastic, latent carried),
               append its episodes to the actor replay  ← gives the world model
               the DEATHS the expert traces never had (closes death-blindness)
  2. LEARN WM  train the world model on a mix of pinned win-traces + actor data
               (traces are NEVER evicted → keep boss/late-level/win coverage)
  3. LEARN AC  train actor + critic in imagination on the updated world model
               (anchors include trace states → the actor rehearses the boss in
               its head long before it can survive to reach it)

The bet: actor dies early → its deaths enter replay → WM learns to predict them
→ actor learns to dodge → advances further → new states enter replay → repeat.
A curriculum that bootstraps OUTWARD from the first screen.

Metrics (the whole point — printed every iter, plotted to train_metrics.png):
  progress     mean episode_progress (level-aware) — the HEADLINE: does it get further?
  win/death    fraction of episodes ending win / death(+game_over)
  survival     mean steps survived
  ep_reward    mean shaped return
  wm_*         recon / kl / reward / cont losses — world-model health
  imag_return  mean imagined λ-return  — actor is finding value
  entropy      policy entropy          — exploration vs commitment
x-axis is ENV DECISIONS collected (= actions; ×3 = NES frames), the real budget.

    python -m dreamer.train --smoke                 # quick wiring check (~mins)
    python -m dreamer.train --iters 60 --collect_eps 6
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from dreamer.actor import Actor, actor_critic_train_step
from dreamer.critic import Critic
from dreamer.buffer import ReplayBuffer
from dreamer.collect import fill_buffer_from_traces, trace_paths
from dreamer.envs import make_contra_env, NUM_ACTIONS
from dreamer.verify_actor import eval_real
from dreamer.world_model import WorldModel
from dreamer import out_path
from contra.reward import progress_coord


# ── replay: pinned traces + actor ring, mixed at a fixed ratio ────────────────

def mixed_sample(trace_buf, actor_buf, batch, trace_frac):
    """Draw a batch part from the (pinned) trace buffer, part from actor data.

    Until the actor buffer has enough data, draw everything from traces. Keeping
    a fixed trace fraction FOREVER is the anti-forgetting guarantee: the world
    model can't lose the boss/late-level coverage that only traces provide."""
    if not actor_buf.can_sample(1):
        return trace_buf.sample(batch)
    n_trace = max(1, round(batch * trace_frac))
    n_actor = max(1, batch - n_trace)
    bt, ba = trace_buf.sample(n_trace), actor_buf.sample(n_actor)
    return {k: torch.cat([bt[k], ba[k]], 0) for k in bt}


@torch.no_grad()
def collect_episodes(wm, actor, env, buf, n_episodes, size, device, max_steps,
                     greedy=False):
    """Run the actor reactively in the real env; append transitions to `buf`
    (reward-INTO-observation convention) and return per-episode stats.

    Stochastic by default (sampling = exploration); greedy for clean eval."""
    stats = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        state = wm.rssm.initial(1, device)
        prev_action = torch.zeros(1, NUM_ACTIONS, device=device)
        start_prog = progress_coord(env.unwrapped.get_ram())
        max_prog, ep_reward, reason, nsteps = start_prog, 0.0, "max_steps", max_steps
        carry_r, carry_term, is_first = 0.0, False, True
        for t in range(max_steps):
            frame = torch.as_tensor(obs, dtype=torch.float32, device=device)
            frame = frame.permute(2, 0, 1).unsqueeze(0) / 255.0
            state, _ = wm.rssm.obs_step(state, prev_action, wm.encoder(frame))
            logits = actor(wm.rssm.get_feat(state))
            a = int(logits.argmax(-1)) if greedy else int(Categorical(logits=logits).sample())
            buf.add(obs, a, carry_r, is_first, carry_term)   # store transition INTO obs
            is_first = False
            nobs, r, term, trunc, info = env.step(a)
            prev_action = F.one_hot(torch.tensor([a], device=device), NUM_ACTIONS).float()
            ep_reward += r
            max_prog = max(max_prog, progress_coord(env.unwrapped.get_ram()))
            if term or trunc:
                buf.add(nobs, 0, r, False, bool(term))       # terminal observation
                reason, nsteps = info.get("episode_end_reason", "?"), t + 1
                break
            obs, carry_r, carry_term = nobs, r, False
        stats.append((float(max_prog - start_prog), ep_reward, reason, nsteps))
    return stats


def _rates(stats):
    """(mean progress, mean reward, win_rate, death_rate, mean steps, reasons)."""
    prog = np.mean([s[0] for s in stats]); rew = np.mean([s[1] for s in stats])
    steps = np.mean([s[3] for s in stats])
    reasons = {}
    for s in stats:
        reasons[s[2]] = reasons.get(s[2], 0) + 1
    n = len(stats)
    win = reasons.get("win", 0) / n
    death = (reasons.get("death", 0) + reasons.get("game_over", 0)) / n
    return prog, rew, win, death, steps, reasons


def _plot(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H = {k: [h[k] for h in history] for k in history[0]}
    x = H["env_steps"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    ax[0, 0].plot(x, H["eval_prog"], "-o", color="tab:blue", label="eval (greedy)")
    ax[0, 0].plot(x, H["coll_prog"], "-", color="tab:cyan", alpha=0.6, label="collect (sampled)")
    ax[0, 0].set_title("progress (level-aware)"); ax[0, 0].set_xlabel("env decisions"); ax[0, 0].legend()
    ax[0, 1].plot(x, H["win"], "-o", color="tab:green", label="win");
    ax[0, 1].plot(x, H["death"], "-o", color="tab:red", label="death")
    ax[0, 1].set_title("outcome rate"); ax[0, 1].set_ylim(-0.05, 1.05); ax[0, 1].legend()
    ax[0, 2].plot(x, H["survival"], "-o", color="tab:purple")
    ax[0, 2].set_title("survival steps")
    ax[1, 0].plot(x, H["imag_return"], "-o", color="tab:blue")
    ax[1, 0].set_title("imagined λ-return")
    ax[1, 1].plot(x, H["entropy"], "-o", color="tab:orange")
    ax[1, 1].set_title("policy entropy")
    ax[1, 2].plot(x, H["wm_recon"], "-o", color="tab:brown", label="recon")
    ax[1, 2].plot(x, H["wm_reward"], "-o", color="tab:olive", label="reward")
    ax[1, 2].set_title("world-model loss"); ax[1, 2].legend()
    for a in ax.flat:
        a.set_xlabel("env decisions"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=85); plt.close(fig)


CONFIG_KEYS = ("ckpt", "train_traces", "trace_frac", "actor_cap", "seq_len", "batch",
               "iters", "collect_eps", "wm_updates", "ac_updates", "max_steps",
               "horizon", "imag_batch", "gamma", "lam", "ent_coef", "size", "deter",
               "wm_lr", "actor_lr", "critic_lr", "eval_every", "eval_eps", "ckpt_every",
               "logdir", "seed")


def _load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    # config file supplies the defaults; any CLI flag overrides its field.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="dreamer/train_config.yaml",
                     help="YAML of default hyperparameters (CLI flags override its fields)")
    pre_args, _ = pre.parse_known_args()
    cfg = _load_config(pre_args.config)

    p = argparse.ArgumentParser(parents=[pre],
                                description="Component 6 — full DreamerV3 training loop")
    p.add_argument("--ckpt", help="trace-pretrained world model to warm-start from")
    p.add_argument("--train_traces", type=int)
    p.add_argument("--iters", type=int)
    p.add_argument("--collect_eps", type=int, help="episodes collected per iter")
    p.add_argument("--wm_updates", type=int)
    p.add_argument("--ac_updates", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--seq_len", type=int)
    p.add_argument("--trace_frac", type=float, help="batch fraction from pinned traces")
    p.add_argument("--horizon", type=int)
    p.add_argument("--imag_batch", type=int)
    p.add_argument("--size", type=int)
    p.add_argument("--deter", type=int)
    p.add_argument("--gamma", type=float)
    p.add_argument("--lam", type=float)
    p.add_argument("--ent_coef", type=float)
    p.add_argument("--wm_lr", type=float)
    p.add_argument("--actor_lr", type=float)
    p.add_argument("--critic_lr", type=float)
    p.add_argument("--actor_cap", type=int, help="actor ring capacity (decisions)")
    p.add_argument("--max_steps", type=int, help="cap per collected episode")
    p.add_argument("--eval_every", type=int)
    p.add_argument("--eval_eps", type=int)
    p.add_argument("--ckpt_every", type=int)
    p.add_argument("--logdir", help="tensorboard log dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--smoke", action="store_true", help="tiny run to check wiring")
    p.set_defaults(**{k: cfg[k] for k in cfg if k in CONFIG_KEYS})
    args = p.parse_args()

    missing = [k for k in CONFIG_KEYS if getattr(args, k, None) is None]
    if missing:
        p.error(f"missing config keys (not in {pre_args.config} or CLI): {missing}")

    if args.smoke:
        args.iters, args.collect_eps, args.wm_updates, args.ac_updates = 3, 2, 20, 20
        args.eval_every, args.eval_eps = 1, 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ── pinned trace buffer (built once; env closed before collection opens) ──
    train_paths = trace_paths(1)
    if args.train_traces and args.train_traces > 0:      # 0/None → use ALL win traces
        train_paths = train_paths[: args.train_traces]
    tlen = sum(len(np.load(q, allow_pickle=True)["actions"]) for q in train_paths)
    trace_buf = ReplayBuffer(tlen + 64, (args.size, args.size, 3), 21, args.seq_len, device)
    fill_buffer_from_traces(trace_buf, train_paths, verbose=False)
    actor_buf = ReplayBuffer(args.actor_cap, (args.size, args.size, 3), 21, args.seq_len, device)
    print(f"[6] pinned traces={trace_buf.size}  actor ring cap={args.actor_cap}  "
          f"trace_frac={args.trace_frac}")

    # ── models: warm-start WM from the trace-pretrained checkpoint ───────────
    wm = WorldModel(size=args.size, deter=args.deter).to(device)
    if os.path.exists(args.ckpt):
        wm.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"[6] warm-started world model ← {args.ckpt}")
    else:
        print(f"[6] WARNING no checkpoint at {args.ckpt}; cold WM (much harder)")
    actor = Actor(wm.feat_dim, NUM_ACTIONS).to(device)
    critic = Critic(wm.feat_dim, gamma=args.gamma, lam=args.lam).to(device)
    opt_wm = torch.optim.Adam(wm.parameters(), lr=args.wm_lr)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    opt_critic = torch.optim.Adam(critic.net.parameters(), lr=args.critic_lr)

    from torch.utils.tensorboard import SummaryWriter
    run_dir = os.path.join(args.logdir, f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(run_dir)
    print(f"[6] tensorboard → {run_dir}   (view: tensorboard --logdir {args.logdir})")

    env = make_contra_env(level=1, size=args.size)        # ONE emulator for the whole session
    history, env_steps, best = [], 0, -1e9
    try:
        for it in range(1, args.iters + 1):
            t0 = time.time()
            # 1. COLLECT (stochastic) ──────────────────────────────────────────
            wm.eval()
            cstats = collect_episodes(wm, actor, env, actor_buf, args.collect_eps,
                                      args.size, device, args.max_steps, greedy=False)
            env_steps += sum(s[3] for s in cstats)
            c_prog, c_rew, c_win, c_death, c_steps, _ = _rates(cstats)

            # 2. LEARN WORLD MODEL ─────────────────────────────────────────────
            wm.train()
            wmm = None
            for _ in range(args.wm_updates):
                loss, wmm, _ = wm.loss(mixed_sample(trace_buf, actor_buf, args.batch, args.trace_frac))
                opt_wm.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0); opt_wm.step()

            # 3. LEARN ACTOR + CRITIC (imagination on the updated WM) ──────────
            wm.eval()
            acm = None
            for _ in range(args.ac_updates):
                acm = actor_critic_train_step(
                    wm, actor, critic, opt_actor, opt_critic,
                    mixed_sample(trace_buf, actor_buf, args.batch, args.trace_frac),
                    args.horizon, args.ent_coef, args.imag_batch)

            # ── tensorboard: per-iter collect + model-health scalars ──────────
            writer.add_scalar("collect/progress", c_prog, env_steps)
            writer.add_scalar("collect/death_rate", c_death, env_steps)
            writer.add_scalar("collect/reward", c_rew, env_steps)
            for k in ("loss", "recon", "kl", "reward", "cont"):
                writer.add_scalar(f"wm/{k}", wmm[k], env_steps)
            writer.add_scalar("ac/imag_return", acm["return"], env_steps)
            writer.add_scalar("ac/entropy", acm["entropy"], env_steps)
            writer.add_scalar("ac/critic_loss", acm["critic_loss"], env_steps)
            writer.add_scalar("ac/actor_loss", acm["actor_loss"], env_steps)

            # 4. EVAL (greedy) + record ────────────────────────────────────────
            if it % args.eval_every == 0 or it == args.iters:
                estats = collect_episodes(wm, actor, env, actor_buf, args.eval_eps,
                                          args.size, device, args.max_steps, greedy=True)
                # note: eval episodes ALSO feed the buffer (free on-policy data)
                env_steps += sum(s[3] for s in estats)
                e_prog, e_rew, e_win, e_death, e_steps, e_reasons = _rates(estats)
                best = max(best, e_prog)
                for tag, val in [("eval/progress", e_prog), ("eval/win_rate", e_win),
                                 ("eval/death_rate", e_death), ("eval/survival", e_steps),
                                 ("eval/reward", e_rew), ("eval/best_progress", best)]:
                    writer.add_scalar(tag, val, env_steps)
                history.append(dict(
                    iter=it, env_steps=env_steps, eval_prog=e_prog, coll_prog=c_prog,
                    win=e_win, death=e_death, survival=e_steps, ep_reward=e_rew,
                    imag_return=acm["return"], entropy=acm["entropy"],
                    wm_recon=wmm["recon"], wm_reward=wmm["reward"]))
                _plot(history, out_path("train_metrics.png"))
                dt = time.time() - t0
                print(f"[it {it:3d}] steps={env_steps:6d}  EVAL prog={e_prog:6.1f} "
                      f"win={e_win:.2f} death={e_death:.2f} surv={e_steps:5.0f}  "
                      f"| imagR={acm['return']:.2f} H={acm['entropy']:.2f} "
                      f"recon={wmm['recon']:.0f} rew={wmm['reward']:.3f}  "
                      f"| best={best:.1f}  ({dt:.0f}s)  {e_reasons}")
            else:
                print(f"[it {it:3d}] steps={env_steps:6d}  collect prog={c_prog:6.1f} "
                      f"death={c_death:.2f}  | imagR={acm['return']:.2f} H={acm['entropy']:.2f}")

            if it % args.ckpt_every == 0:
                for name, mod in [("wm", wm), ("actor", actor), ("critic", critic.net)]:
                    torch.save(mod.state_dict(), out_path(f"train_{name}.pt"))
    finally:
        env.close()
        writer.close()
        for name, mod in [("wm", wm), ("actor", actor), ("critic", critic.net)]:
            torch.save(mod.state_dict(), out_path(f"train_{name}.pt"))
        if history:
            _plot(history, out_path("train_metrics.png"))
        print(f"\n[6] done. best eval progress={best:.1f}  "
              f"metrics → {out_path('train_metrics.png')}  ckpts → tmp/dreamer/train_*.pt")


if __name__ == "__main__":
    main()
