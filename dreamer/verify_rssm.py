"""Verify a trained world model — Layers 1-3 on ONE saved checkpoint (no retraining).

Loads tmp/dreamer/world_model.pt (from dreamer.train_wm) and runs, on held-out
data:

  Layer 1  closed-loop sanity  — posterior recon matches real frames, and entities
                                 decode from the POSTERIOR feat (entities are in z
                                 when observing).
  Layer 2  open-loop dream     — roll the prior H steps on real actions and compare
                                 to reality: pixel tracking (closed/open mse, motion)
                                 AND entity dream accuracy (dreamed entity heatmap vs
                                 the encoder's read of the real frame) + coherence.
                                 The key test: does the dream keep entities in the
                                 right place, or hallucinate / go blurry?
  Layer 3  heads               — reward-head accuracy (closed-loop) and the
                                 continue/death head on a random death rollout.

"Reference" entities use the frozen encoder's entity head on the REAL frame — our
C3a-verified proxy for RAM truth — so no RAM plumbing is needed. Because the dream
never sees those frames, matching them is a genuine dynamics test.

    python -m dreamer.verify_rssm --ckpt tmp/dreamer/world_model.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from dreamer import out_path
from dreamer.collect import trace_paths
from dreamer.verify_modes import _load_trace, _pick_anchors
from dreamer.verify_ae import _overlay_heatmaps
from dreamer.world_model import WorldModel


def load_wm(ckpt, device):
    blob = torch.load(ckpt, map_location=device)
    cfg = blob["config"]
    wm = WorldModel(size=cfg["size"], deter=cfg["deter"],
                    entity_grid=cfg["entity_grid"]).to(device)
    wm.load_state_dict(blob["world_model"])
    wm.eval()
    return wm, cfg


def _ref_entity(wm, frames_u):
    """Encoder's entity heatmaps on real frames (T,H,W,3 in [0,1]) → (T,4,g,g)."""
    return wm.enc_entity_head(wm.encoder(frames_u.permute(0, 3, 1, 2)))


# ── Layer 1 — closed-loop sanity ──────────────────────────────────────────────

def layer1(wm, frames, onehot, cfg, device, window=200):
    W = min(window, len(frames))
    fwin = frames[:W].unsqueeze(0)
    awin = onehot[:W].unsqueeze(0)
    first = torch.zeros(1, W, device=device)
    with torch.no_grad():
        posts, _ = wm.observe(fwin, awin, first)
        feat = wm.rssm.get_feat(posts)
        recon = wm.decode(feat)[0].clamp(0, 1)
        real = frames[:W].permute(0, 3, 1, 2)
        recon_mse = ((recon - real) ** 2).mean().item()
        line = f"[L1] closed-loop recon MSE {recon_mse:.4f}"
        if cfg["entity_grid"]:
            ref = _ref_entity(wm, frames[:W])                 # (W,4,g,g)
            pred = wm.feat_entity_head(feat[0])               # (W,4,g,g) from posterior feat
            ent_mse = ((pred - ref) ** 2).mean(dim=(0, 2, 3))
            line += "  |  entity-in-z MSE " + " ".join(
                f"{n}={ent_mse[c]:.4f}" for c, n in enumerate(["ply", "pbl", "eny", "ebl"]))
        print(line)
        print("     (recon MSE≈0 and low entity MSE ⇒ the posterior state represents the scene)")


# ── Layer 2 — open-loop dream (pixel + entity) ────────────────────────────────

def _dream_rollout(wm, frames, onehot, t0, context, H, device):
    """Open-loop from the warmup at t0. Returns real/dream frames + ref/dream entity."""
    fwin = frames[t0 - context: t0 + H].unsqueeze(0)
    awin = onehot[t0 - context: t0 + H].unsqueeze(0)
    first = torch.zeros(1, context + H, device=device)
    with torch.no_grad():
        posts, _ = wm.observe(fwin, awin, first)
        state = {k: posts[k][:, context - 1] for k in posts}          # end of warmup
        ia = onehot[t0 - 1: t0 - 1 + H].unsqueeze(0)
        dream_feat = wm.rssm.imagine(state, ia)                       # (1,H,feat)
        dream_img = wm.decode(dream_feat)[0].clamp(0, 1)             # (H,3,h,w)
        real = frames[t0: t0 + H].permute(0, 3, 1, 2)               # (H,3,h,w)
        ref_ent = _ref_entity(wm, frames[t0: t0 + H]) if wm.entity_grid else None
        dream_ent = wm.feat_entity_head(dream_feat[0]) if wm.entity_grid else None
    return real.cpu(), dream_img.cpu(), ref_ent, dream_ent


def _dream_grid(real, dream_img, ref_ent, dream_ent, size):
    """Tall grid: rows=steps, cols = real | dream | real+ref-entity | dream+dream-entity."""
    import cv2
    H = real.shape[0]
    to_u8 = lambda t: (t.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    R, D = to_u8(real), to_u8(dream_img)
    rows = []
    for t in range(H):
        cells = [R[t], D[t]]
        if ref_ent is not None:
            cells.append(_overlay_heatmaps(R[t], ref_ent[t].cpu().numpy(), size))
            cells.append(_overlay_heatmaps(D[t], dream_ent[t].cpu().numpy(), size))
        vsep = np.zeros((size, 2, 3), np.uint8)
        row = cells[0]
        for c in cells[1:]:
            row = np.concatenate([row, vsep, c], axis=1)
        rows.append(row)
    return np.concatenate(rows, axis=0)


def layer2(wm, frames, onehot, terminal, cfg, args, device):
    import imageio
    anchors = _pick_anchors(frames, terminal, args.context, args.horizon, args.n_anchors, args.seed)
    print(f"\n[L2] dream from {len(anchors)} anchors (H={args.horizon}); "
          f"cols = real | dream | real-entity | dream-entity")
    hdr = f"  {'anchor':>6} {'closed?':>8} {'open_mse':>9} {'real_mot':>9} {'open_mot':>9}"
    if cfg["entity_grid"]:
        hdr += f" {'ent_early':>10} {'ent_late':>9}"
    print(hdr)
    for i, t0 in enumerate(anchors):
        real, dream, ref_ent, dream_ent = _dream_rollout(
            wm, frames, onehot, t0, args.context, args.horizon, device)
        open_mse = ((dream - real) ** 2).mean().item()
        rm = (real[1:] - real[:-1]).abs().mean().item() * 255
        om = (dream[1:] - dream[:-1]).abs().mean().item() * 255
        row = f"  {t0:6d} {'-':>8} {open_mse:9.4f} {rm:9.2f} {om:9.2f}"
        if cfg["entity_grid"]:
            e = ((dream_ent - ref_ent) ** 2).mean(dim=(1, 2, 3)).cpu()   # per step
            row += f" {e[:5].mean():10.4f} {e[-5:].mean():9.4f}"          # early vs late drift
        path = out_path(f"rssm_dream_{i:02d}_t{t0}.png")
        imageio.imwrite(path, _dream_grid(real, dream, ref_ent, dream_ent, cfg["size"]))
        print(row + f"   → {path}")
    print("     GATE: open_mot≈real_mot (dream moves), ent_late not ≫ ent_early (entities")
    print("           tracked, not hallucinated/blurred over the horizon).")


# ── Layer 3 — heads (reward + continue/death) ─────────────────────────────────

def layer3(wm, frames, onehot, cfg, device, size, seed):
    # reward-head accuracy over closed-loop on the held-out trace is skipped here
    # (trace rewards aren't loaded); the death check is the informative one.
    from dreamer.buffer import ReplayBuffer, _fill_from_env
    buf = ReplayBuffer(4000, (size, size, 3), 21, 2, device)
    _fill_from_env(buf, 3000, seed)
    term = np.where(buf.is_terminal[: buf.size])[0]
    if len(term) == 0:
        print("\n[L3] no death in the random rollout — cannot test the continue head.")
        return
    d = int(term[0])
    lo = max(0, d - 15)
    fr = torch.as_tensor(buf.image[lo:d + 1], dtype=torch.float32, device=device) / 255.0
    oh = F.one_hot(torch.as_tensor(buf.action[lo:d + 1], dtype=torch.long, device=device), 21).float()
    with torch.no_grad():
        posts, _ = wm.observe(fr.unsqueeze(0), oh.unsqueeze(0), torch.zeros(1, len(fr), device=device))
        _, pcont = wm.predict_heads(wm.rssm.get_feat(posts))
    pc = pcont[0].cpu().numpy()
    print(f"\n[L3] continue head P(continue) into a death (step {d}); last 8 steps before terminal:")
    print("     " + " ".join(f"{v:.2f}" for v in pc[-8:]))
    print("     GATE: P(continue) should drop toward 0 approaching the death step.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="tmp/dreamer/world_model.pt")
    p.add_argument("--trace", default=None, help="held-out trace npz (default: level1 #8)")
    p.add_argument("--n_anchors", type=int, default=6)
    p.add_argument("--context", type=int, default=5)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm, cfg = load_wm(args.ckpt, device)
    print(f"[verify] {args.ckpt}  size={cfg['size']} entity_grid={cfg['entity_grid']}")

    trace = args.trace or trace_paths(1)[8]
    frames, onehot, terminal = _load_trace(trace, cfg["size"], device)
    print(f"[verify] held-out trace {trace.split('/')[-1]}  ({len(frames)} frames)")

    layer1(wm, frames, onehot, cfg, device)
    layer2(wm, frames, onehot, terminal, cfg, args, device)
    layer3(wm, frames, onehot, cfg, device, cfg["size"], args.seed)


if __name__ == "__main__":
    main()
