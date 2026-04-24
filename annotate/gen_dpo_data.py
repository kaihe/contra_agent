"""
gen_dpo_data.py — Load a search graph, print statistics, validate traces.

Validates:
  1. Good trace reaches a win.
  2. Dead traces contain a player-death event (random sample of 5).
  3. Secondary traces accumulate less reward than the good continuation (random sample of 5).

Usage:
    python annotate/gen_dpo_data.py synthetic/mc_graph/graph_level1_<date>.pkl
"""

import os
import random
import re
import sys

import numpy as np
import stable_retro as retro

from contra.replay import rewind_state, step_env
from contra.events import (
    get_level, compute_reward,
    EV_PLAYER_DIE, EV_GAME_CLEAR, ADDR_LEVEL_ROUTINE,
)
from synthetic.mc_search_dpo import SearchNode, load_graph

_GAME           = "Contra-Nes"
_STATE_BY_LEVEL = {i: f"Level{i}" for i in range(1, 9)}


# ── graph helpers ──────────────────────────────────────────────────────────────

def _good_trace(root: SearchNode) -> list[SearchNode]:
    """Return all nodes in the committed path, root at index 0."""
    nodes = [root]
    n = root.next
    while n is not None:
        nodes.append(n)
        n = n.next
    return nodes


def _branch_actions(head: SearchNode) -> np.ndarray:
    acts = []
    n = head
    while n is not None:
        acts.append(n.action)
        n = n.next
    return np.stack(acts) if acts else np.empty((0, 9), dtype=np.uint8)


def _graph_stats(root: SearchNode) -> dict:
    good_nodes = _good_trace(root)
    dead_count = sum(len(n.dead)      for n in good_nodes)
    sec_count  = sum(len(n.secondary) for n in good_nodes)
    return dict(
        good_trace_len     = len(good_nodes) - 1,
        branch_nodes       = sum(1 for n in good_nodes if n.dead or n.secondary),
        dead_branches      = dead_count,
        secondary_branches = sec_count,
    )


def _all_paths(root: SearchNode) -> list[dict]:
    good_nodes = _good_trace(root)
    paths = [{"kind": "good", "branch_node": None, "length": len(good_nodes) - 1}]
    for i, node in enumerate(good_nodes):
        for head in node.dead:
            paths.append({"kind": "dead",      "branch_node": i, "length": len(_branch_actions(head))})
        for head in node.secondary:
            paths.append({"kind": "secondary", "branch_node": i, "length": len(_branch_actions(head))})
    return paths


# ── emulator helpers ───────────────────────────────────────────────────────────

def _make_env(level: int) -> retro.RetroEnv:
    state_label = _STATE_BY_LEVEL[level]
    use_spread  = level > 1
    env = retro.make(
        game=_GAME,
        state=retro.State.NONE if use_spread else state_label,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    if use_spread:
        env.load_state(f"spread_gun_state/{state_label}", retro.data.Integrations.CUSTOM_ONLY)
    env.reset()
    return env


def _replay_to(env, init_emu: bytes, good_nodes: list[SearchNode], up_to: int) -> None:
    """Rewind to init_emu then step through good_nodes[1:up_to+1]."""
    rewind_state(env, init_emu)
    for node in good_nodes[1 : up_to + 1]:
        step_env(env, node.action)


def _accumulate_reward(env, actions: list[np.ndarray]) -> float:
    total = 0.0
    for act in actions:
        pre_ram = env.unwrapped.get_ram().copy()
        step_env(env, act)
        curr_ram = env.unwrapped.get_ram()
        total += compute_reward(pre_ram, curr_ram)
    return total


def collect_dpo_pairs(root: SearchNode, init_emu: bytes, env: retro.RetroEnv = None, chunk_len: int = 128) -> list[dict]:
    """Build (good_trace, bad_trace) action pairs from the search tree."""
    good_nodes   = _good_trace(root)
    good_actions = np.array(
        [n.action for n in good_nodes[1:]], dtype=np.uint8
    ) if len(good_nodes) > 1 else np.empty((0, 9), dtype=np.uint8)

    emu_states = []
    if env is not None:
        rewind_state(env, init_emu)
        emu_states.append(init_emu)
        for act in good_actions:
            step_env(env, act)
            emu_states.append(env.em.get_state())

    def _pad(arr: np.ndarray) -> np.ndarray:
        if len(arr) < chunk_len:
            arr = np.concatenate([arr, np.zeros((chunk_len - len(arr), 9), dtype=arr.dtype)])
        return arr

    result = []
    for diverge_idx, node in enumerate(good_nodes):
        for kind, branches in (("dead", node.dead), ("secondary", node.secondary)):
            for head in branches:
                branch_acts = _branch_actions(head)

                good_reward = bad_reward = None
                if env is not None:
                    branch_emu = emu_states[diverge_idx]
                    good_cont  = good_actions[diverge_idx : diverge_idx + len(branch_acts)]
                    n_steps    = min(len(branch_acts), len(good_cont))

                    rewind_state(env, branch_emu)
                    bad_reward = _accumulate_reward(env, branch_acts[:n_steps])

                    rewind_state(env, branch_emu)
                    good_reward = _accumulate_reward(env, good_cont[:n_steps])

                    if kind == "secondary" and bad_reward >= good_reward:
                        continue

                L           = min(len(branch_acts), chunk_len)
                p           = min(chunk_len - L, diverge_idx)
                chunk_start = diverge_idx - p

                good_trace = good_actions[chunk_start : chunk_start + chunk_len].copy()
                branch_L   = branch_acts[:L]
                bad_trace  = (
                    np.concatenate([good_actions[chunk_start:diverge_idx], branch_L])
                    if p > 0 else branch_L
                )

                result.append({
                    "start_emu"   : init_emu,
                    "prefix_len"  : chunk_start,
                    "good_trace"  : _pad(good_trace),
                    "bad_trace"   : _pad(bad_trace),
                    "branch_pos"  : diverge_idx,
                    "pivot"       : p,
                    "kind"        : kind,
                    "good_reward" : good_reward,
                    "bad_reward"  : bad_reward,
                })
    return result

# ── validators ─────────────────────────────────────────────────────────────────

def verify_win(root: SearchNode, init_emu: bytes, level: int, goal: str) -> bool:
    """Replay the good trace and confirm a win occurs."""
    good_nodes = _good_trace(root)
    env = _make_env(level)
    rewind_state(env, init_emu)

    won = False
    for step, node in enumerate(good_nodes[1:]):
        pre_ram = env.unwrapped.get_ram().copy()
        step_env(env, node.action)
        curr_ram = env.unwrapped.get_ram()

        if goal == "game_clear" and EV_GAME_CLEAR.trigger(pre_ram, curr_ram):
            print(f"  game clear at step {step + 1}")
            won = True
            break
        if int(curr_ram[ADDR_LEVEL_ROUTINE]) in (0x08, 0x09):
            new_level = get_level(curr_ram)
            print(f"  level transition {level} → {new_level} at step {step + 1}")
            won = True
            break

    env.close()
    return won


def validate_dead_traces(root: SearchNode, init_emu: bytes, level: int, n_sample: int = 5) -> None:
    """Check that sampled dead branches contain a player-death event."""
    good_nodes = _good_trace(root)
    all_dead   = [(i, head) for i, node in enumerate(good_nodes) for head in node.dead]

    if not all_dead:
        print("  no dead branches to validate")
        return

    sample = random.sample(all_dead, min(n_sample, len(all_dead)))
    env    = _make_env(level)

    print(f"  sampling {len(sample)} of {len(all_dead)} dead branches:")
    for branch_idx, head in sample:
        _replay_to(env, init_emu, good_nodes, branch_idx)

        found_death = False
        step        = 0
        n = head
        while n is not None:
            pre_ram = env.unwrapped.get_ram().copy()
            step_env(env, n.action)
            curr_ram = env.unwrapped.get_ram()
            step += 1
            if EV_PLAYER_DIE.trigger(pre_ram, curr_ram):
                found_death = True
                break
            n = n.next

        branch_len = len(_branch_actions(head))
        status     = f"✓ died@step{step}" if found_death else "✗ no death detected"
        print(f"    branch@{branch_idx:4d}  len={branch_len:3d}  {status}")

    env.close()


def validate_secondary_traces(root: SearchNode, init_emu: bytes, level: int, n_sample: int = 5) -> None:
    """Check that sampled secondary branches yield less reward than the good continuation."""
    good_nodes = _good_trace(root)
    all_sec    = [(i, head) for i, node in enumerate(good_nodes) for head in node.secondary]

    if not all_sec:
        print("  no secondary branches to validate")
        return

    sample = random.sample(all_sec, min(n_sample, len(all_sec)))
    env    = _make_env(level)

    print(f"  sampling {len(sample)} of {len(all_sec)} secondary branches:")
    for branch_idx, head in sample:
        _replay_to(env, init_emu, good_nodes, branch_idx)
        branch_emu = env.em.get_state()

        sec_acts  = list(_branch_actions(head))
        good_cont = [n.action for n in good_nodes[branch_idx + 1 : branch_idx + 1 + len(sec_acts)]]
        n_steps   = min(len(sec_acts), len(good_cont))

        rewind_state(env, branch_emu)
        reward_sec  = _accumulate_reward(env, sec_acts[:n_steps])

        rewind_state(env, branch_emu)
        reward_good = _accumulate_reward(env, good_cont[:n_steps])

        ok     = reward_sec < reward_good
        status = "✓ sec<good" if ok else "✗ sec≥good"
        print(f"    branch@{branch_idx:4d}  len={len(sec_acts):3d}  "
              f"sec={reward_sec:7.1f}  good={reward_good:7.1f}  {status}")

    env.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _infer_level_goal(path: str) -> tuple[int, str]:
    fname = os.path.basename(path)
    m = re.search(r"level(\d+)", fname)
    if m:
        return int(m.group(1)), "level_up"
    if "game" in fname:
        return 1, "game_clear"
    return 1, "level_up"


GRAPH_PATH = "synthetic/mc_graph/graph_level6_202604231600.pkl"


def main():
    level, goal = _infer_level_goal(GRAPH_PATH)

    print(f"loading graph: {GRAPH_PATH}")
    root, init_emu = load_graph(GRAPH_PATH)
    stats = _graph_stats(root)
    paths = _all_paths(root)
    non_good = [p for p in paths if p["kind"] != "good"]

    print(f"\n── graph statistics ────────────────────────────")
    print(f"  good trace    : {stats['good_trace_len']} steps")
    print(f"  branch nodes  : {stats['branch_nodes']}")
    print(f"  dead branches : {stats['dead_branches']}")
    print(f"  secondary     : {stats['secondary_branches']}")
    print()
    print(f"  {'#':>4}  {'branch':>6}  {'length':>7}  kind")
    print("  " + "-" * 30)
    for k, p in enumerate(paths):
        branch = "-" if p["branch_node"] is None else str(p["branch_node"])
        print(f"  {k:4d}  {branch:>6}  {p['length']:7d}  {p['kind']}")
    print()

    env = _make_env(level)
    pairs = collect_dpo_pairs(root, init_emu, env)
    env.close()
    print(f"── DPO pairs ───────────────────────────────────")
    print(f"  total pairs : {len(pairs)}")
    if pairs:
        print(f"\n  {'#':>3}  {'pivot':>5}  {'kind':>9}")
        print("  " + "-" * 22)
        for k, p in enumerate(pairs):
            print(f"  {k:3d}  {p['pivot']:5d}  {p['kind']:>9}")
    print()

    print(f"── validating good trace ───────────────────────")
    won = verify_win(root, init_emu, level, goal)
    print(f"  {'✓ win confirmed' if won else '✗ no win — graph may be corrupt'}")
    if not won:
        sys.exit(1)
    print()

    print(f"── validating dead traces ──────────────────────")
    validate_dead_traces(root, init_emu, level)
    print()

    print(f"── validating secondary traces ─────────────────")
    validate_secondary_traces(root, init_emu, level)
    print()


if __name__ == "__main__":
    main()
