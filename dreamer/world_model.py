"""Component 3b — the RSSM and the WorldModel that wraps encoder+RSSM+decoder.

RSSM (Recurrent State-Space Model) is Dreamer's latent simulator. Each step has:
  deter  h_t  — GRU recurrent state (the memory: folds in the past + last action)
  stoch  z_t  — a categorical latent (stoch groups × classes), the "what's here now"

Two ways to produce z:
  posterior  q(z_t | h_t, embed_t)  — uses the encoded frame (training / observing)
  prior      p(z_t | h_t)           — predicts z WITHOUT the frame (imagination)

Training the prior to match the posterior (the KL term) is what lets the model
dream forward with no pixels — which is the Component 3b gate.

Categorical sampling uses a manual straight-through estimator + unimix (1% uniform
mix) — version-proof and avoids torch.distributions dispatch quirks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer.models import ConvDecoder, ConvEncoder, EntityHead


def _mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                         nn.Linear(hidden, out))


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


class RSSM(nn.Module):
    def __init__(self, embed_dim: int, action_dim: int, deter: int = 256,
                 stoch: int = 32, classes: int = 32, hidden: int = 256,
                 unimix: float = 0.01):
        super().__init__()
        self.deter, self.stoch, self.classes = deter, stoch, classes
        self.zdim = stoch * classes
        self.unimix = unimix
        self.img_in = nn.Sequential(
            nn.Linear(self.zdim + action_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.cell = nn.GRUCell(hidden, deter)
        self.prior_net = _mlp(deter, hidden, self.zdim)
        self.post_net = _mlp(deter + embed_dim, hidden, self.zdim)

    # ── categorical latent helpers (manual straight-through) ─────────────────
    def _probs(self, logits):
        p = F.softmax(logits.reshape(*logits.shape[:-1], self.stoch, self.classes), -1)
        return (1 - self.unimix) * p + self.unimix / self.classes

    def _sample(self, logits):
        p = self._probs(logits)                              # (..., stoch, classes)
        idx = torch.multinomial(p.reshape(-1, self.classes), 1).reshape(p.shape[:-1])
        onehot = F.one_hot(idx, self.classes).float()
        st = onehot + p - p.detach()                         # straight-through
        return st.reshape(*logits.shape[:-1], self.zdim)

    def _kl(self, post_logits, prior_logits):
        pp, qp = self._probs(post_logits), self._probs(prior_logits)
        kl = (pp * (torch.log(pp + 1e-8) - torch.log(qp + 1e-8))).sum(-1).sum(-1)
        return kl                                            # (...,) over stoch+classes

    # ── state transitions ────────────────────────────────────────────────────
    def initial(self, batch, device):
        z = torch.zeros(batch, self.zdim, device=device)
        return {"deter": torch.zeros(batch, self.deter, device=device),
                "stoch": z, "logits": z.clone()}

    def img_step(self, prev, prev_action):
        x = self.img_in(torch.cat([prev["stoch"], prev_action], -1))
        deter = self.cell(x, prev["deter"])
        logits = self.prior_net(deter)
        return {"deter": deter, "stoch": self._sample(logits), "logits": logits}

    def obs_step(self, prev, prev_action, embed):
        prior = self.img_step(prev, prev_action)
        logits = self.post_net(torch.cat([prior["deter"], embed], -1))
        post = {"deter": prior["deter"], "stoch": self._sample(logits), "logits": logits}
        return post, prior

    def observe(self, embed_seq, action_seq, is_first_seq):
        """Run posterior+prior over a (B,L,...) sequence. is_first resets state."""
        B, L = embed_seq.shape[:2]
        device = embed_seq.device
        prev = self.initial(B, device)
        prev_action = torch.zeros(B, action_seq.shape[-1], device=device)
        posts, priors = [], []
        for t in range(L):
            keep = (1.0 - is_first_seq[:, t]).unsqueeze(-1)   # 0 where episode resets
            prev = {k: v * keep for k, v in prev.items()}
            prev_action = prev_action * keep
            post, prior = self.obs_step(prev, prev_action, embed_seq[:, t])
            posts.append(post); priors.append(prior)
            prev, prev_action = post, action_seq[:, t]
        stack = lambda lst, k: torch.stack([s[k] for s in lst], 1)
        keys = ["deter", "stoch", "logits"]
        return ({k: stack(posts, k) for k in keys}, {k: stack(priors, k) for k in keys})

    def imagine(self, state, action_seq):
        """Roll the PRIOR forward (no observations) over a (B,H,A) action seq."""
        feats, s = [], state
        for t in range(action_seq.shape[1]):
            s = self.img_step(s, action_seq[:, t])
            feats.append(self.get_feat(s))
        return torch.stack(feats, 1)

    def get_feat(self, state):
        return torch.cat([state["deter"], state["stoch"]], -1)

    def kl_loss(self, posts, priors, free=1.0):
        sg = lambda x: x.detach()
        dyn = self._kl(sg(posts["logits"]), priors["logits"])     # train prior→post
        rep = self._kl(posts["logits"], sg(priors["logits"]))     # train post→prior
        dyn = torch.clamp(dyn, min=free).mean()
        rep = torch.clamp(rep, min=free).mean()
        return 0.5 * dyn + 0.1 * rep, dyn, rep


class WorldModel(nn.Module):
    def __init__(self, size: int = 128, num_actions: int = 21, embed_dim: int = 1024,
                 deter: int = 256, stoch: int = 32, classes: int = 32, depth: int = 32,
                 entity_grid: int | None = None, entity_weight: float = 1.0):
        super().__init__()
        self.size = size
        self.embed_dim = embed_dim
        self.encoder = ConvEncoder(size, depth=depth, embed_dim=embed_dim)
        self.rssm = RSSM(embed_dim, num_actions, deter, stoch, classes)
        self.feat_dim = deter + stoch * classes
        self.decoder = ConvDecoder(size, depth=depth, feat_dim=self.feat_dim)
        # reward in symlog space (handles wide range); continue = 1-terminal as a logit.
        self.reward_head = _mlp(self.feat_dim, 256, 1)
        self.continue_head = _mlp(self.feat_dim, 256, 1)

        # Optional entity head on the RSSM feature. With a frozen encoder, z is
        # shaped mainly by (blurry) pixel recon, so entities the encoder captured
        # can be re-dropped inside (h,z). This head forces them to survive: its
        # target is the frozen encoder's OWN entity head on embed (distillation),
        # so no RAM/heatmap plumbing is needed in the buffer. See load_encoder.
        self.entity_grid = entity_grid
        self.entity_weight = entity_weight
        if entity_grid:
            self.enc_entity_head = EntityHead(embed_dim, 4, grid=entity_grid, depth=depth)
            self.feat_entity_head = EntityHead(self.feat_dim, 4, grid=entity_grid, depth=depth)

    def _encode_seq(self, image):
        B, L = image.shape[:2]
        x = image.permute(0, 1, 4, 2, 3).reshape(B * L, 3, self.size, self.size)
        return self.encoder(x).reshape(B, L, -1)

    def observe(self, image, action, is_first):
        embed = self._encode_seq(image)
        return self.rssm.observe(embed, action, is_first)

    def loss(self, batch):
        image, action, first = batch["image"], batch["action"], batch["is_first"]
        B, L = image.shape[:2]
        embed = self._encode_seq(image)                       # encoder runs once
        posts, priors = self.rssm.observe(embed, action, first)
        feat = self.rssm.get_feat(posts)
        recon = self.decoder(feat.reshape(B * L, -1)).reshape(B, L, 3, self.size, self.size)
        target = image.permute(0, 1, 4, 2, 3)
        # SUM the squared error over pixels (image NLL), mean over batch/time —
        # so reconstruction dominates and KL is a small regularizer. Meaning over
        # pixels instead makes recon ~50000x too weak vs KL → blurry dreams.
        recon_loss = ((recon - target) ** 2).sum(dim=(-3, -2, -1)).mean()
        kl, dyn, rep = self.rssm.kl_loss(posts, priors)

        # reward + continue heads — the task gradient that pulls entities into
        # the latent (reward ← progress/enemy-hit, continue ← death/levelup).
        reward_pred = self.reward_head(feat).squeeze(-1)
        reward_loss = F.mse_loss(reward_pred, symlog(batch["reward"]))
        cont_logit = self.continue_head(feat).squeeze(-1)
        cont_loss = F.binary_cross_entropy_with_logits(cont_logit, batch["cont"])

        loss = recon_loss + kl + reward_loss + cont_loss
        metrics = {"loss": loss.item(), "recon": recon_loss.item(), "kl": kl.item(),
                   "dyn": dyn.item(), "rep": rep.item(),
                   "reward": reward_loss.item(), "cont": cont_loss.item()}

        # entity head (optional): distill the frozen encoder's entity heatmaps from
        # embed into the RSSM feature, forcing entities to survive into (h,z).
        if self.entity_grid:
            with torch.no_grad():
                ent_target = self.enc_entity_head(embed.reshape(B * L, -1))
            ent_pred = self.feat_entity_head(feat.reshape(B * L, -1))
            entity_loss = F.mse_loss(ent_pred, ent_target)
            loss = loss + self.entity_weight * entity_loss
            metrics["entity"] = entity_loss.item()
            metrics["loss"] = loss.item()

        return loss, metrics, posts

    def predict_heads(self, feat):
        """For eval: (predicted reward in real units, P(continue))."""
        return (symexp(self.reward_head(feat).squeeze(-1)),
                torch.sigmoid(self.continue_head(feat).squeeze(-1)))

    def decode(self, feat):
        n = feat.shape[0] if feat.dim() == 2 else feat.shape[0] * feat.shape[1]
        return self.decoder(feat.reshape(n, -1)).reshape(*feat.shape[:-1], 3, self.size, self.size)

    def predict_entity(self, feat):
        """Predicted entity heatmaps from the RSSM feature (verification / probing)."""
        g = self.entity_grid
        n = feat.shape[0] if feat.dim() == 2 else feat.shape[0] * feat.shape[1]
        return self.feat_entity_head(feat.reshape(n, -1)).reshape(*feat.shape[:-1], 4, g, g)

    def load_encoder(self, path, device="cpu"):
        """Load a pretrained ConvEncoder (dreamer.pretrain_ae checkpoint) into this WM.

        If this WM was built with entity_grid, also loads the encoder's EntityHead as
        the (frozen) distillation target for feat_entity_head. The RSSM, decoder,
        reward/continue heads and feat_entity_head still train.
        """
        ckpt = torch.load(path, map_location=device)
        self.encoder.load_state_dict(ckpt["encoder"])
        if self.entity_grid:
            self.enc_entity_head.load_state_dict(ckpt["entity_head"])

    def freeze_encoder(self):
        """Freeze the encoder (and the encoder entity head, if present): no grad
        updates. Exclude from the optimizer via
        `filter(lambda p: p.requires_grad, wm.parameters())`."""
        mods = [self.encoder] + ([self.enc_entity_head] if self.entity_grid else [])
        for m in mods:
            for p in m.parameters():
                p.requires_grad_(False)
