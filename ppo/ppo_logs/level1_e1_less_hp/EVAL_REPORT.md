# level1_9_states_less_hp — checkpoint eval report

**Checkpoint:** `tmp/ppo/checkpoints/level1_9_states_less_hp/level1_9_states_less_hp_59000000_steps.zip` (step 59M of a 64M budget)
**Eval:** 20 episodes per anchor × 9 anchors (`ppo/states/win_level1_202603301145_*`), stochastic, `skip=8`, `max_steps=2000`. Greedy re-run + GIFs/montages for spot-checks.
**Date:** 2026-06-14

## TL;DR

**The "agent farms enemy HP" hypothesis is _not_ supported by this checkpoint.** The signal that looked like farming — `mean_delta_x` going negative late in training — is a **measurement artifact**: `delta_x` is derived from `xscroll` (RAM 100/101), which resets to ~0 at the level→level transition, so **every win logs `delta_x ≈ −start_x`**. As `win_rate` climbed to ~0.53, the win-heavy anchors dragged mean `delta_x` negative. The agent actually advances **rightward** and reaches the end-of-level wall in essentially every episode.

What this run *did* do: it **fixed** the camping/farming that killed the prior `level1_10_states_one_live` run (`progress` 0.03→0.1, enemy cap 5.0→2.5, `max_steps` 3000→2000). `ep_len_mean` stayed modest (~246, not ballooning to ~1600) and `win_rate` reached 0.53.

The **real** weakness is a weapon dependency — see below.

## Eval results (20 eps/anchor, stochastic)

| anchor (start_x) | win | die | dx_all | **dx_win** | **dx_die** | hp_win | hp_die | steps_win | steps_die |
|---|---|---|---|---|---|---|---|---|---|
| s0000_x0000 | 10 | 10 | 1056 | **0** | 2112 | 26.1 | 17.8 | 499 | 301 |
| s0139_x0325 | 12 | 8 | 845 | **−325** | 2600 | 24.1 | 21.1 | 446 | 362 |
| s0278_x0696 | 12 | 8 | 420 | **−697** | 2094 | 19.7 | 16.3 | 396 | 292 |
| s0418_x1096 | 14 | 6 | −295 | **−1096** | 1573 | 17.6 | 14.7 | 344 | 222 |
| s0557_x1426 | 12 | 8 | −258 | **−1426** | 1494 | 16.5 | 15.2 | 296 | 216 |
| s0696_x1808 | 14 | 6 | −922 | **−1808** | 1146 | 12.3 | 11.2 | 243 | 164 |
| s0835_x2218 | **1** | **19** | 700 | −2218 | 854 | 8.5 | 8.5 | 218 | 137 |
| s0975_x2553 | 13 | 7 | −1478 | −2553 | 519 | 4.1 | 5.0 | 145 | 81 |
| s1114_x2869 | 11 | 9 | −1488 | −2870 | 202 | 2.7 | 2.5 | 102 | 36 |

**Overall: 99/180 = 55% win** (81 deaths, 0 timeouts, 0 game_over). Matches the train log's final `win_rate ≈ 0.53`. (Full numbers in `eval_per_state.csv`.)

## Why the negative delta_x is NOT farming

Two columns settle it:

1. **`dx_win ≈ −start_x` for every anchor** (0, −325, −697, −1096, −1426, −1808, … −2870). On a win the final `xscroll` is ~0 because the level counter (`RAM[48]`, `EV_LEVELUP`) increments at the transition and `xscroll` resets. So a winning episode mechanically records `delta_x = 0 − start_x`. The deeper the anchor, the more negative its win delta_x — nothing to do with retreating.
2. **`dx_die` is large and positive** (up to +2600). When the agent does *not* win, it dies while pushing **forward**, not camping backward.

Enemy-HP totals are just normal full-traverse kills: `hp_win` tops out at ~26 from x0000 ≈ the per-region cap (2.5) × ~10 regions. The cap is doing its job — there is no runaway HP accumulation.

GIF spot-checks (`eval_x1808_win.gif`, `eval_x2218_death.gif`, `eval_x0000_win.gif`, montages `montage_*.png`) confirm: the agent runs right through the jungle to the end-wall/fortress every time.

## The real failure mode: no win without the Spread gun

Greedy rollout, weapon held at reset vs. at episode end (`tmp/probe_weapon.py`):

| anchor | weapon @reset | weapon @end | result |
|---|---|---|---|
| s0000_x0000 | Default | **Spread** | win |
| s0139_x0325 | Machine | **Spread** | death* |
| s0278_x0696 | Machine | **Spread** | win |
| s0418_x1096 | Machine | **Spread** | win |
| s0557_x1426 | Machine | **Spread** | win |
| s0696_x1808 | Machine | **Spread** | win |
| s0835_x2218 | Machine | Machine | **death** |
| s0975_x2553 | Laser | Laser | **death** |
| s1114_x2869 | Laser | Laser | **death** |

- Every anchor that reaches the wall **with Spread wins**; every anchor that arrives with **Machine/Laser dies**.
- The Spread pickup sits **before ~x1808**. Anchors starting at **x2218 and later begin past it**, so the agent reaches the wall with a weak gun and dies — this is why x2218 wins only **1/20** even though it starts *closer* to the wall than x1808 (which wins 14/20).
- The agent never learned to clear the end-wall without Spread. It's a **policy gap conditioned on weapon state**, the opposite of HP-farming.

\* x0325 died once in the greedy single-episode probe despite reaching Spread — just per-episode variance; stochastically it wins 12/20.

(Note: stochastic eval gives x2553/x2869 ~12/20 wins while greedy dies — those anchors sit at/near the transition the user flagged as a "guaranteed-win" bad state, so random actions occasionally trip `EV_LEVELUP`. Their wins are not evidence of skill.)

## Recommendations

1. **Don't read negative `mean_delta_x` as farming.** Add a progress metric that is robust to the levelup reset, e.g. log **max `xscroll` reached during the episode** (or freeze delta_x at the pre-levelup frame), and log **per-anchor win-rate** so transition-anchor freebies don't inflate the aggregate.
2. **Address the weapon dependency.** Either (a) include anchors that start with Default/Machine *at the wall* so the agent must learn a no-Spread kill, or (b) accept the dependency but ensure a Spread pickup is reachable from every start region.
3. **Audit the late anchors.** x2218 (always dies with Machine) and x2553/x2869 (transition freebies) are not giving a clean training signal — they teach "die" or "win for free," not "beat the wall."

## Repro / notes

- The checkpoint was pickled under **numpy ≥ 2.0**; this env has **numpy 1.26**, so `PPO.load` fails (`numpy._core`, `_frombuffer`, `PCG64`). Eval scripts (`tmp/eval_less_hp.py`, `tmp/gif_less_hp.py`, `tmp/probe_weapon.py`) work around it by aliasing `numpy._core`→`numpy.core` and passing reconstructed spaces/schedules via `custom_objects` (channel-first obs space to match the trained CNN). `ppo/test.py` itself cannot load this checkpoint as-is.
- Artifacts in this folder: `eval_per_state.csv`, GIFs `eval_<anchor>_<outcome>.gif`, montages `montage_*.png`.
