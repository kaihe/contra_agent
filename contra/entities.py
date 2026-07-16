"""contra/entities.py — read on-screen entity positions from Contra RAM.

Every on-screen actor stores its position in RAM as an NES PPU sprite coordinate
(x ∈ [0,255], y ∈ [0,239], y grows downward) — which is also its pixel position
on the 256×240 frame. So a single RAM snapshot yields the player, every live
enemy, every enemy bullet and every player bullet, ready to use for reward /
observation features (e.g. "distance to nearest incoming bullet") or to overlay
on a frame for debugging.

The reusable core (``scan`` and the accessors) is **numpy-only** — imaging deps
(PIL) are imported lazily inside :func:`annotate`, so training/reward code can
``from contra.entities import scan`` without pulling in a rendering stack.

See ``contra/ENTITIES.md`` for the full mechanism, address map and disassembly
references. Quick summary:

  player       : SPRITE_X_POS $0334, SPRITE_Y_POS $031a
  enemy slot i : live when ENEMY_ROUTINE $04b8+i != 0; position
                 ENEMY_X_POS $033e+i / ENEMY_Y_POS $0324+i;
                 ENEMY_TYPE $0528+i == 0x01 → enemy bullet, else an enemy.
  player bullet: slot active when PLAYER_BULLET_SLOT $0388+i != 0; position
                 PLAYER_BULLET_X_POS $03c8+i / _Y_POS $03b8+i; OWNER $0448+i (0=P1).
"""

from dataclasses import dataclass

import numpy as np

# ── RAM addresses ─────────────────────────────────────────────────────────────
ADDR_PLAYER_X, ADDR_PLAYER_Y = 0x0334, 0x031A
ADDR_ENEMY_ROUTINE, ADDR_ENEMY_TYPE = 0x04B8, 0x0528
ADDR_ENEMY_X, ADDR_ENEMY_Y = 0x033E, 0x0324
ADDR_PBULLET_SLOT, ADDR_PBULLET_X, ADDR_PBULLET_Y, ADDR_PBULLET_OWNER = \
    0x0388, 0x03C8, 0x03B8, 0x0448

N_SLOTS = 16               # both the enemy array and the player-bullet array hold 16 slots
ENEMY_TYPE_BULLET = 0x01   # an enemy slot of this type is enemy gunfire, not an enemy
PPU_W, PPU_H = 256, 240    # native NES frame; used to offset overlay markers if cropped


def _as_xy(points: list) -> np.ndarray:
    """Turn a list of (x, y) into an (n, 2) int16 array (shape (0, 2) if empty)."""
    return np.array(points, dtype=np.int16).reshape(-1, 2)


@dataclass(frozen=True)
class Entities:
    """On-screen positions in PPU pixel coords (x∈[0,255], y∈[0,239], y down-positive).

    Each field is an int16 array: ``player`` is (2,); the rest are (n, 2) and may be
    empty. Bullet/enemy arrays hold only *live* slots.
    """

    player: np.ndarray         # (2,)
    enemies: np.ndarray        # (N, 2) live non-bullet enemies
    enemy_bullets: np.ndarray  # (M, 2) enemy gunfire
    player_bullets: np.ndarray # (K, 2) the player's own bullets (P1)


def player_pos(ram: np.ndarray) -> np.ndarray:
    """Player-1 (x, y) as an int16 array of shape (2,)."""
    return np.array([ram[ADDR_PLAYER_X], ram[ADDR_PLAYER_Y]], dtype=np.int16)


def player_bullets(ram: np.ndarray, owner: int | None = 0) -> np.ndarray:
    """Active player-bullet (x, y) positions; default owner=0 (P1). None = both players."""
    out = []
    for i in range(N_SLOTS):
        if ram[ADDR_PBULLET_SLOT + i] == 0:
            continue
        if owner is not None and ram[ADDR_PBULLET_OWNER + i] != owner:
            continue
        out.append((int(ram[ADDR_PBULLET_X + i]), int(ram[ADDR_PBULLET_Y + i])))
    return _as_xy(out)


def scan(ram: np.ndarray, *, bullet_owner: int | None = 0) -> Entities:
    """Read all live entities from `ram` in one pass. The main entry point.

    ``bullet_owner`` filters player bullets (0 = P1, default). Enemy slots are
    split into ``enemies`` and ``enemy_bullets`` by ENEMY_TYPE.
    """
    enemies, ebullets = [], []
    for i in range(N_SLOTS):
        if ram[ADDR_ENEMY_ROUTINE + i] == 0:          # inactive slot
            continue
        xy = (int(ram[ADDR_ENEMY_X + i]), int(ram[ADDR_ENEMY_Y + i]))
        (ebullets if ram[ADDR_ENEMY_TYPE + i] == ENEMY_TYPE_BULLET else enemies).append(xy)
    return Entities(
        player=player_pos(ram),
        enemies=_as_xy(enemies),
        enemy_bullets=_as_xy(ebullets),
        player_bullets=player_bullets(ram, owner=bullet_owner),
    )


# ── Occupancy heatmaps (aux training target) ──────────────────────────────────

HEATMAP_CLASSES = ("player", "player_bullets", "enemies", "enemy_bullets")


def entity_heatmaps(ram: np.ndarray, grid: int = 32, sigma: float = 1.0,
                    screen_hw: tuple[int, int] = (224, 240)) -> np.ndarray:
    """(4, grid, grid) float32 occupancy heatmaps, one channel per HEATMAP_CLASSES.

    Positions from :func:`scan` are full-PPU coords (256×240). They are shifted
    into the stable_retro screen crop (`screen_hw`, default 224×240 = an 8px
    overscan crop each side) and normalised to `grid`, so the heatmap lines up
    with a frame that was ``cv2.resize``d from that screen to *any* square size —
    the grid is resolution-independent. Entities outside the visible crop are
    dropped. Overlapping blobs take the elementwise max (occupancy, not a sum),
    and each present entity contributes a Gaussian with peak 1.0.
    """
    H, W = screen_hw
    xoff = (PPU_W - W) // 2
    yoff = (PPU_H - H) // 2
    e = scan(ram)
    groups = (e.player.reshape(1, 2), e.player_bullets, e.enemies, e.enemy_bullets)
    hm = np.zeros((4, grid, grid), dtype=np.float32)
    yy, xx = np.mgrid[0:grid, 0:grid].astype(np.float32)
    inv = 1.0 / (2.0 * sigma * sigma)
    for c, pts in enumerate(groups):
        for ex, ey in pts.reshape(-1, 2):
            gx = (float(ex) - xoff) / W * grid
            gy = (float(ey) - yoff) / H * grid
            if not (0.0 <= gx < grid and 0.0 <= gy < grid):   # off the visible crop
                continue
            blob = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) * inv)
            np.maximum(hm[c], blob, out=hm[c])
    return hm


# ── Reward / observation helpers ──────────────────────────────────────────────

def _min_dist(points: np.ndarray, origin: np.ndarray) -> float:
    if len(points) == 0:
        return float("inf")
    d = points.astype(np.float32) - origin.astype(np.float32)
    return float(np.min(np.hypot(d[:, 0], d[:, 1])))


def nearest_enemy_bullet_dist(ram: np.ndarray) -> float:
    """Euclidean pixel distance from the player to the closest enemy bullet (inf if none)."""
    return _min_dist(scan(ram).enemy_bullets, player_pos(ram))


def nearest_enemy_dist(ram: np.ndarray) -> float:
    """Euclidean pixel distance from the player to the closest live enemy (inf if none)."""
    return _min_dist(scan(ram).enemies, player_pos(ram))


# ── Visualization (lazy imaging deps) ─────────────────────────────────────────

def annotate(frame: np.ndarray, ram: np.ndarray) -> np.ndarray:
    """Draw markers for player/enemies/bullets onto an RGB `frame`, return a new array.

    player=green box, enemy=red box, enemy bullet=yellow dot, player bullet=cyan dot.
    RAM holds full-PPU (256×240) coords, so if `frame` is cropped (stable_retro
    returns 240×224, an 8px overscan crop on every side) markers are shifted by the
    same crop to line up.
    """
    from PIL import Image, ImageDraw

    xoff = (PPU_W - frame.shape[1]) // 2
    yoff = (PPU_H - frame.shape[0]) // 2
    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img)
    e = scan(ram)

    px, py = int(e.player[0]) - xoff, int(e.player[1]) - yoff
    d.rectangle([px - 6, py - 10, px + 6, py + 10], outline=(0, 255, 0), width=2)
    for x, y in e.enemies:
        x, y = int(x) - xoff, int(y) - yoff
        d.rectangle([x - 7, y - 9, x + 7, y + 9], outline=(255, 0, 0), width=2)
    for x, y in e.enemy_bullets:
        x, y = int(x) - xoff, int(y) - yoff
        d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 255, 0), width=2)
    for x, y in e.player_bullets:
        x, y = int(x) - xoff, int(y) - yoff
        d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(0, 255, 255), width=2)
    return np.asarray(img)


# ── Demo (self-test / GIF) ────────────────────────────────────────────────────

def _demo(steps: int, stride: int) -> None:
    """Roll the Level1 state (hold right, pulse fire) and write an annotated GIF+PNG."""
    import os
    import imageio.v2 as imageio
    import stable_retro as retro
    from PIL import Image
    from contra.replay import step_env, GAME

    env = retro.make(game=GAME, state="Level1", use_restricted_actions=retro.Actions.ALL,
                     obs_type=retro.Observations.IMAGE, render_mode=None,
                     inttype=retro.data.Integrations.CUSTOM_ONLY)
    env.reset()
    frames, sample = [], None
    for t in range(steps):
        # hold right; pulse fire (standard gun is edge-triggered, so held B fires once)
        act = np.zeros(9, dtype=np.uint8); act[7] = 1; act[0] = (t % 2 == 0)
        step_env(env, act)
        ram = env.unwrapped.get_ram()
        marked = annotate(env.unwrapped.get_screen(), ram)
        if t % stride == 0:
            frames.append(marked)
        if sample is None and len(scan(ram).enemy_bullets):
            sample = marked
    env.close()

    os.makedirs("tmp", exist_ok=True)
    imageio.mimsave("tmp/overlay_entities.gif", frames, fps=20)
    print(f"Saved {len(frames)}-frame GIF -> tmp/overlay_entities.gif")
    if sample is not None:
        Image.fromarray(sample).save("tmp/overlay_entities_sample.png")
        print("Saved sample frame with a bullet -> tmp/overlay_entities_sample.png")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Demo: overlay entity positions on Level1 frames")
    ap.add_argument("--steps", type=int, default=200, help="decision steps to roll (default 200)")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame in the GIF")
    args = ap.parse_args()
    _demo(args.steps, args.stride)


if __name__ == "__main__":
    main()
