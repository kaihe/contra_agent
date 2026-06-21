# Level-2 Baseline Evaluation

- **Checkpoint:** `tmp/ppo/level2_e1_baseline/level2_e1_baseline_77000000_steps.zip` (77M steps)
- **Command:** `python ppo/eval_level2.py --model tmp/ppo/level2_e1_baseline/level2_e1_baseline_77000000_steps.zip --episodes 20`
- **Setup:** 20 stochastic episodes from each of the 6 level-2 anchor states (one per indoor room: `s0000` = first room … `s0005` = boss room). skip=3, stack=3, max_steps=4000.

## Results

| Anchor (room) | win% | mean reward | mean enemy hp | mean core broken |
|---|---|---|---|---|
| s0000 (entry) | 0%  | 204.8 | 60.9 | 3.00 |
| s0001         | 0%  | 180.2 | 61.5 | 2.55 |
| s0002         | 0%  | 141.1 | 58.3 | 1.80 |
| s0003         | 0%  | 91.8  | 54.5 | 1.00 |
| s0004         | 0%  | 14.8  | 33.8 | 0.00 |
| s0005 (boss)  | 90% | 116.8 | 70.0 | 0.00 |
| **Overall**   | **15%** | **124.9** | **56.5** | **1.39** |

## Interpretation

- **Boss yes, approach no.** From the boss room (`s0005`) the agent wins 90%, but from every earlier room the win rate is 0% — it never completes the full indoor run. The difficulty is the path to the boss, not the boss itself.
- **Core breaking tapers with distance.** From `s0000` it averages 3 cores broken before dying, dropping monotonically to 0 by `s0004`. Partial early progress, but it consistently dies before clearing the room sequence.
- **`s0004` is the worst non-boss room** (lowest reward, fewest enemy hits, 0 cores) — likely a hard death trap right before the boss.
- **Reward is dominated by combat/progress, not winning**, since only `s0005` ever triggers the levelup bonus.

Net: this baseline has effectively learned the boss fight but not the indoor traversal — the 15% overall win rate is driven entirely by boss-room starts.
