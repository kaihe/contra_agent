"""Component 5 — the actor, trained by imagination (REINFORCE + entropy).

The actor is a reactive policy π(a|s) over the 21 discrete actions. It is trained
ENTIRELY in imagination: from real anchor states we roll the FROZEN world model
forward under the actor's OWN sampled actions, score the imagined trajectory with
the critic's λ-returns, and push action log-probs up where the return beat the
critic's value baseline (advantage). No new env interaction during training.

  actor loss   = −E[ stop_grad(Â)·log π(a|s) ]  −  η·E[ entropy(π) ]
  critic loss  = E[ (v(s) − stop_grad(λ-return))² ]          (on the SAME rollouts)

REINFORCE (score-function), not backprop-through-dynamics: the action is a
DISCRETE sample, so there's no differentiable path from the return back through
the sampled action into π. We instead weight log π(a|s) by the advantage — the
standard trick for discrete actors. Advantages are normalized per batch so the
gradient scale stays sane as returns grow.

The actor never sees a real frame here. Deployment (and C6's data collection) runs
it reactively one action at a time — see verify_actor.eval_real.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from dreamer.critic import lambda_return
from dreamer.world_model import _mlp, symlog


class Actor(nn.Module):
    """π(a|s): feature → logits over the discrete actions."""

    def __init__(self, feat_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.net = _mlp(feat_dim, hidden, num_actions)
        self.num_actions = num_actions

    def forward(self, feat):
        return self.net(feat)                            # logits

    def dist(self, feat):
        return Categorical(logits=self.net(feat))


def imagine_policy(wm, actor, state, horizon: int):
    """Roll the FROZEN world model open-loop under the actor's sampled actions.

    Returns (all detached except logp/ent, which carry the actor gradient):
        feats  (N, H+1, F)  states s_0 (anchor) .. s_H
        logp   (N, H)       log π(a_t|s_t)        — grad to actor
        ent    (N, H)       entropy of π at s_t   — grad to actor
        reward (N, H)       predicted reward into s_1 .. s_H
        cont   (N, H)       predicted P(continue) of s_1 .. s_H
    """
    s = {k: v.detach() for k, v in state.items()}
    feats = [wm.rssm.get_feat(s)]
    logps, ents = [], []
    for _ in range(horizon):
        feat = feats[-1]                                 # detached → grad enters only via actor
        dist = actor.dist(feat)
        a = dist.sample()                                # discrete, non-differentiable
        logps.append(dist.log_prob(a))
        ents.append(dist.entropy())
        with torch.no_grad():
            s = wm.rssm.img_step(s, F.one_hot(a, actor.num_actions).float())
            feats.append(wm.rssm.get_feat(s))
    feats = torch.stack(feats, 1)                        # (N,H+1,F)
    logp = torch.stack(logps, 1)                         # (N,H)
    ent = torch.stack(ents, 1)                           # (N,H)
    with torch.no_grad():
        reward, cont = wm.predict_heads(feats[:, 1:])    # heads on s_1 .. s_H
    return feats, logp, ent, reward, cont


def actor_critic_train_step(wm, actor, critic, opt_actor, opt_critic, batch,
                            horizon: int, ent_coef: float, imag_batch: int = 256):
    """One joint actor+critic imagination update. World model stays frozen.

    Anchors = every posterior state of the observed batch (flattened, subsampled
    to `imag_batch`); imagine `horizon` steps from each under the actor.
    """
    H = horizon
    with torch.no_grad():
        posts, _ = wm.observe(batch["image"], batch["action"], batch["is_first"])
        anchors = {k: v.reshape(-1, v.shape[-1]) for k, v in posts.items()}
        n = anchors["deter"].shape[0]
        if n > imag_batch:
            sel = torch.randperm(n, device=anchors["deter"].device)[:imag_batch]
            anchors = {k: v[sel] for k, v in anchors.items()}

    feats, logp, ent, reward, cont = imagine_policy(wm, actor, anchors, H)

    with torch.no_grad():
        tvalue = critic.target_value(feats)              # (N,H+1)
        pcont = critic.gamma * cont
        returns = lambda_return(reward, tvalue[:, :H], pcont, tvalue[:, H], critic.lam)
        adv = returns - tvalue[:, :H]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)    # per-batch normalization

    # actor: REINFORCE on normalized advantage + entropy bonus
    actor_loss = -(adv * logp).mean() - ent_coef * ent.mean()
    opt_actor.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
    opt_actor.step()

    # critic: regress value of s_0..s_{H-1} to the λ-return (feats are constants)
    pred = critic.value_symlog(feats[:, :H].detach())
    critic_loss = F.mse_loss(pred, symlog(returns))
    opt_critic.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.net.parameters(), 100.0)
    opt_critic.step()
    critic.update_target()

    return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
            "entropy": ent.mean().item(), "return": returns.mean().item(),
            "adv_std": (returns - tvalue[:, :H]).std().item()}
