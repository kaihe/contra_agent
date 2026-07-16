"""Train the world model on the frozen encoder — C3b/C3c, standalone.

Loads the pretrained + frozen encoder (dreamer.pretrain_ae), builds the RSSM +
decoder + reward/continue heads (+ optional entity head on `feat`), and trains the
world-model loss on mixed batches from dreamer.wm_data.WMDataGenerator:

  * whole-level win-trace frames  (agent never dies)
  * anchor-branched death rollouts (DEATH / off-policy states across the WHOLE level)

Saves tmp/dreamer/world_model.pt. Verify separately with dreamer.verify_rssm
(Layers 1-3 on this one checkpoint — no retraining).

    python -m dreamer.train_wm --steps 6000
    # keep the failure stream fresh: regenerate rollouts every 1000 steps
    python -m dreamer.train_wm --steps 6000 --refresh_every 1000
    # A/B baseline without the entity head
    python -m dreamer.train_wm --steps 6000 --no_entity_head --out world_model_noent.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from dreamer import out_path
from dreamer.wm_data import WMDataGenerator
from dreamer.world_model import WorldModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--enc_ckpt", default="tmp/dreamer/ae_pretrained.pt",
                   help="pretrained encoder (+entity head) to freeze; input size from its config")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--deter", type=int, default=256)
    # data generator (dreamer.wm_data)
    p.add_argument("--train_traces", type=int, default=8, help="traces used as buffer FRAMES")
    p.add_argument("--trace_frac", type=float, default=0.5)
    p.add_argument("--anchor_traces", type=int, default=16,
                   help="traces to snapshot save-state anchors from (cheap; can exceed train_traces)")
    p.add_argument("--anchor_stride", type=int, default=20, help="anchor every N actions")
    p.add_argument("--anchor_rollouts", type=int, default=300, help="branched rollouts per fill")
    p.add_argument("--anchor_max_steps", type=int, default=150, help="max steps per rollout")
    p.add_argument("--rand_cap", type=int, default=25000, help="failure-buffer capacity (ring)")
    p.add_argument("--refresh_every", type=int, default=0,
                   help="regenerate anchor rollouts every N steps (0 = fill once)")
    # model
    p.add_argument("--no_entity_head", action="store_true",
                   help="disable the feat entity head (for the A/B baseline)")
    p.add_argument("--entity_weight", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--out", default="world_model.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.enc_ckpt):
        raise SystemExit(f"--enc_ckpt not found: {args.enc_ckpt}")
    cfg = torch.load(args.enc_ckpt, map_location="cpu")["config"]
    size, grid = cfg["size"], cfg["grid"]
    entity_grid = None if args.no_entity_head else grid
    print(f"[wm] device={device}  enc={args.enc_ckpt}  size={size}  "
          f"entity_head={not args.no_entity_head} (grid={grid})")

    # ── mixed data generator: win-trace frames + anchor-branched death rollouts ─
    data = WMDataGenerator(
        level=1, size=size, seq_len=args.seq_len, device=device,
        train_traces=args.train_traces, trace_frac=args.trace_frac,
        anchor_traces=args.anchor_traces, anchor_stride=args.anchor_stride,
        anchor_rollouts=args.anchor_rollouts, anchor_max_steps=args.anchor_max_steps,
        rand_cap=args.rand_cap, seed=args.seed)

    # ── model: frozen encoder + trainable RSSM/decoder/heads(+entity) ─────────
    wm = WorldModel(size=size, deter=args.deter, entity_grid=entity_grid,
                    entity_weight=args.entity_weight).to(device)
    wm.load_encoder(args.enc_ckpt, device)
    wm.freeze_encoder()
    params = [q for q in wm.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    print(f"[wm] training {sum(q.numel() for q in params)/1e6:.1f}M params (encoder frozen)")

    wm.train()
    for step in range(1, args.steps + 1):
        loss, m, _ = wm.loss(data.sample(args.batch))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 100.0)
        opt.step()
        if args.refresh_every and step % args.refresh_every == 0:
            data.refresh()
        if step % 500 == 0 or step == 1:
            ent = f"  entity {m['entity']:.4f}" if "entity" in m else ""
            print(f"  step {step:5d}  loss {m['loss']:8.1f}  recon {m['recon']:8.1f}  "
                  f"kl {m['kl']:.2f}  reward {m['reward']:.3f}  cont {m['cont']:.3f}{ent}")

    ckpt = out_path(args.out)
    torch.save({"world_model": wm.state_dict(),
                "config": {"size": size, "deter": args.deter, "entity_grid": entity_grid}},
               ckpt)
    print(f"\n[wm] saved → {ckpt}")


if __name__ == "__main__":
    main()
