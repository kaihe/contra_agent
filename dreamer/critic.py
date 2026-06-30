"""Component 4 — the critic (value head) + λ-returns.

The reward head (C3c) predicts the IMMEDIATE reward *into* a state; the critic
predicts the VALUE — the discounted sum of all FUTURE reward the current policy
collects from here on. Value is policy-dependent and is trained entirely in
imagination: roll the FROZEN world model forward from real anchor states, let it
dream rewards + continues, and regress the critic to the λ-return of that dream.
No new env interaction.

λ-return (DreamerV3) blends a 1-step bootstrap (λ=0, trust the critic) with the
full Monte-Carlo return (λ=1, trust the dreamed rewards):

    V^λ_t = r_t + γ·c_t·[ (1-λ)·v(s_{t+1}) + λ·V^λ_{t+1} ]

With λ=1 and c=1 it collapses to the plain discounted return-to-go
V_t = r_t + γ·V_{t+1} — exactly the curve `verify_reward_plots --value` drew.
That collapse is one of the unit-test gates in verify_critic.

Simplification vs. the paper: a scalar symlog-regression value head (like the
reward head), not the two-hot categorical head. Adequate for Contra's reward
scale and keeps the math transparent.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer.world_model import _mlp, symexp, symlog


def lambda_return(reward, value, pcont, bootstrap, lam):
    """DreamerV3 λ-return over an imagined trajectory.

    All tensors are (B, T), time-aligned so that for the transition t → t+1:
        reward[t]   reward obtained on that transition (reward *into* state t+1)
        value[t]    critic value of state t
        pcont[t]    γ · continue(state t+1)  — the discount, zeroed past a terminal
    `bootstrap` (B,) is the value of the state just past the horizon (state T).

    Returns ret (B, T): the λ-return target for value[t], for t = 0 .. T-1.

    Backward recursion:  ret[t] = reward[t] + pcont[t]·[(1-λ)·v(t+1) + λ·ret[t+1]]
    with ret[T] := bootstrap and v(t+1) taken from `value`/`bootstrap`.
    """
    T = reward.shape[1]
    # value of the NEXT state at each step: [value[1], …, value[T-1], bootstrap]
    next_value = torch.cat([value[:, 1:], bootstrap.unsqueeze(1)], dim=1)
    inputs = reward + pcont * next_value * (1 - lam)      # the non-recursive part
    rets = []
    last = bootstrap
    for t in reversed(range(T)):
        last = inputs[:, t] + pcont[:, t] * lam * last
        rets.append(last)
    return torch.stack(rets[::-1], dim=1)


class Critic(nn.Module):
    """Scalar value head v(feat) with a slow EMA target copy.

    The target net (a lagged copy of the online net) supplies the values used to
    BUILD the λ-return; the online net is regressed toward it. Decoupling the
    target from the net being trained is what stops the value chasing its own
    tail and diverging (the standard Dreamer / DQN trick)."""

    def __init__(self, feat_dim: int, hidden: int = 256, gamma: float = 0.997,
                 lam: float = 0.95, tau: float = 0.02):
        super().__init__()
        self.net = _mlp(feat_dim, hidden, 1)
        self.target = _mlp(feat_dim, hidden, 1)
        self.target.load_state_dict(self.net.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.gamma, self.lam, self.tau = gamma, lam, tau

    def value_symlog(self, feat):
        """Raw symlog-space output — the training LHS (regressed to symlog(ret))."""
        return self.net(feat).squeeze(-1)

    def value(self, feat):
        """Value in real reward units (online net)."""
        return symexp(self.net(feat).squeeze(-1))

    def target_value(self, feat):
        """Value in real reward units (slow target net)."""
        return symexp(self.target(feat).squeeze(-1))

    @torch.no_grad()
    def update_target(self):
        for p, tp in zip(self.net.parameters(), self.target.parameters()):
            tp.data.lerp_(p.data, self.tau)               # tp ← (1-τ)·tp + τ·p


def imagine_with_actions(wm, state, action_seq):
    """Roll the FROZEN world model forward under a given action sequence.

    Returns
        feats  (B, H+1, F)  features for states s_0 (anchor) .. s_H
        reward (B, H)       predicted reward *into* s_1 .. s_H (real units)
        cont   (B, H)       predicted P(continue) of s_1 .. s_H
    """
    feat0 = wm.rssm.get_feat(state).unsqueeze(1)          # s_0 (the anchor)
    feats_img = wm.rssm.imagine(state, action_seq)        # (B,H,F): s_1 .. s_H
    feats = torch.cat([feat0, feats_img], dim=1)          # (B,H+1,F)
    reward, cont = wm.predict_heads(feats_img)            # heads on s_1 .. s_H
    return feats, reward, cont


def critic_train_step(wm, critic, opt, batch, context: int, horizon: int):
    """One imagination critic update.

    Observe `context` real steps to land on an anchor state, imagine `horizon`
    steps forward under the data's own actions (the policy whose value we learn —
    no actor yet), build the λ-return from the TARGET critic + dreamed
    reward/continue, and regress the ONLINE critic to it. WM stays frozen.
    """
    H = horizon
    with torch.no_grad():
        posts, _ = wm.observe(batch["image"][:, :context + 1],
                              batch["action"][:, :context + 1],
                              batch["is_first"][:, :context + 1])
        state = {k: v[:, -1] for k, v in posts.items()}   # s_context
        action_seq = batch["action"][:, context:context + H]
        feats, reward, cont = imagine_with_actions(wm, state, action_seq)
        tvalue = critic.target_value(feats)               # (B,H+1) target values
        pcont = critic.gamma * cont                       # (B,H)
        ret = lambda_return(reward, tvalue[:, :H], pcont, tvalue[:, H], critic.lam)

    pred = critic.value_symlog(feats[:, :H])              # value of s_0 .. s_{H-1}
    loss = F.mse_loss(pred, symlog(ret))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.net.parameters(), 100.0)
    opt.step()
    critic.update_target()
    return loss.item(), ret.detach()
