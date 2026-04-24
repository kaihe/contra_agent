"""
DPOGraphSample — process one search graph into DPO training pairs.

Output NPZ keys:
  chosen_ram        : uint8   (N, T, 2048)   RAM snapshot before each action
  chosen_actions    : uint8   (N, T, 9)      pruned 9-bit NES actions
  rejected_ram      : uint8   (N, T, 2048)
  rejected_actions  : uint8   (N, T, 9)
  chosen_len        : int16   (N,)           real (unpadded) length of chosen trace
  rejected_len      : int16   (N,)           real (unpadded) length of rejected trace
  pivot             : int16   (N,)           shared prefix length within chunk
  kind              : uint8   (N,)           0=dead  1=secondary
  good_reward       : float32 (N,)
  bad_reward        : float32 (N,)
  n_pairs           : int32   scalar
  level             : int32   scalar

T = chunk_len = 128 (padded with zeros if trace is shorter)
"""

import os
import re
import shutil

import numpy as np
import stable_retro as retro

from contra.replay import rewind_state, step_env
from contra.events import EV_GAME_CLEAR, ADDR_LEVEL_ROUTINE, compute_reward
from annotate.instance import prune_actions
from synthetic.mc_search_dpo import SearchNode, load_graph

_BAD_PRUNE_DIR = "synthetic/mc_graph_bad_prune"

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


def _infer_level_goal(path: str) -> tuple[int, str]:
    fname = os.path.basename(path)
    m = re.search(r"level(\d+)", fname)
    if m:
        return int(m.group(1)), "level_up"
    if "game" in fname:
        return 1, "game_clear"
    return 1, "level_up"


def _accumulate_reward(env, actions) -> float:
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


def _make_env(level: int) -> retro.RetroEnv:
    state_label = _STATE_BY_LEVEL[level]
    use_spread  = level > 1
    env = retro.make(
        game=_GAME,
        state=retro.State.NONE if use_spread else state_label,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.RAM,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    if use_spread:
        env.load_state(f"spread_gun_state/{state_label}", retro.data.Integrations.CUSTOM_ONLY)
    env.reset()
    return env


def _collect_emu_states(env, init_emu: bytes, good_actions: np.ndarray) -> list[bytes]:
    """Replay good actions and return emu state snapshots (index = step count from root)."""
    rewind_state(env, init_emu)
    states = [init_emu]
    for act in good_actions:
        step_env(env, act)
        states.append(env.em.get_state())
    return states


def _pad_to(arr: np.ndarray, length: int) -> np.ndarray:
    if len(arr) < length:
        pad = np.zeros((length - len(arr),) + arr.shape[1:], dtype=arr.dtype)
        arr = np.concatenate([arr, pad])
    return arr[:length]


def _replay_ram(env, start_emu: bytes, actions: np.ndarray) -> np.ndarray:
    """Rewind and replay, collecting pre-action RAM at each step. Returns (T, 2048) uint8."""
    rewind_state(env, start_emu)
    T       = len(actions)
    ram_buf = np.empty((T, 2048), dtype=np.uint8)
    for i, act in enumerate(actions):
        ram_buf[i] = env.unwrapped.get_ram()
        step_env(env, act)
    return ram_buf


def _verify_win(env, init_emu: bytes, actions: np.ndarray, goal: str) -> bool:
    """Replay actions and return True if a level-up or game-clear is detected."""
    rewind_state(env, init_emu)
    for act in actions:
        pre_ram = env.unwrapped.get_ram().copy()
        step_env(env, act)
        curr_ram = env.unwrapped.get_ram()
        if goal == "game_clear" and EV_GAME_CLEAR.trigger(pre_ram, curr_ram):
            return True
        if int(curr_ram[ADDR_LEVEL_ROUTINE]) in (0x08, 0x09):
            return True
    return False


class DPOGraphSample:
    """Process one search graph file into a DPO training NPZ."""

    def __init__(self, graph_path: str) -> None:
        self.graph_path       = graph_path
        self.level, self.goal = _infer_level_goal(graph_path)
        self.uuid             = os.path.splitext(os.path.basename(graph_path))[0]

    def process(self, out_dir: str) -> "str | None":
        """Load graph, replay traces, prune actions, collect RAM obs, save NPZ.

        The good path is pruned and replayed once; each chosen chunk is a slice.
        Each bad trace is unique and pruned/replayed individually.

        Returns the output path, or None if the graph yields no valid pairs.
        """
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{self.uuid}.npz")

        if os.path.isfile(out_path):
            print(f"  {self.uuid}: already exists, skipping")
            return out_path

        root, init_emu = load_graph(self.graph_path)
        env            = _make_env(self.level)

        # raw pairs: actions + rewards, no observations yet
        pairs = collect_dpo_pairs(root, init_emu, env)
        if not pairs:
            env.close()
            print(f"  {self.uuid}: no valid pairs")
            return None

        good_nodes   = _good_trace(root)
        good_actions = (
            np.array([n.action for n in good_nodes[1:]], dtype=np.uint8)
            if len(good_nodes) > 1
            else np.empty((0, 9), dtype=np.uint8)
        )

        # prune the entire good path once — all chosen chunks slice from this
        pruned_good = prune_actions(good_actions, init_emu, verbose=True, env=env)

        if not _verify_win(env, init_emu, pruned_good, self.goal):
            env.close()
            os.makedirs(_BAD_PRUNE_DIR, exist_ok=True)
            dest = os.path.join(_BAD_PRUNE_DIR, os.path.basename(self.graph_path))
            shutil.move(self.graph_path, dest)
            print(f"  {self.uuid}: pruned trace failed win check → moved to {_BAD_PRUNE_DIR}")
            return None

        # replay once to get RAM for every good-path step
        good_ram_all = _replay_ram(env, init_emu, pruned_good)            # (G, 2048)
        # emu snapshots at each good-path step, used to rewind for bad-trace pruning
        emu_states   = _collect_emu_states(env, init_emu, pruned_good)    # G+1 entries

        CHUNK = 128  # must match collect_dpo_pairs chunk_len

        chosen_rams,   chosen_acts   = [], []
        rejected_rams, rejected_acts = [], []
        chosen_lens, rejected_lens   = [], []
        pivots, kinds                = [], []
        good_rewards,  bad_rewards   = [], []

        for pair in pairs:
            cs = pair["prefix_len"]  # chunk_start index into good path

            # slice chosen arrays from the pre-computed good path (no re-pruning)
            good_act_slice = pruned_good[cs : cs + CHUNK]
            chosen_lens.append(len(good_act_slice))
            chosen_acts.append(_pad_to(good_act_slice, CHUNK))
            chosen_rams.append(_pad_to(good_ram_all[cs : cs + CHUNK], CHUNK))

            # bad trace is unique per pair — prune and replay individually
            chunk_start_emu = emu_states[cs]
            bad_pruned = prune_actions(pair["bad_trace"], chunk_start_emu, verbose=False, env=env)
            rejected_lens.append(len(bad_pruned))
            rejected_acts.append(_pad_to(bad_pruned, CHUNK))
            rejected_rams.append(_pad_to(_replay_ram(env, chunk_start_emu, bad_pruned), CHUNK))

            pivots.append(pair["pivot"])
            kinds.append(0 if pair["kind"] == "dead" else 1)
            good_rewards.append(pair.get("good_reward") or float("nan"))
            bad_rewards.append(pair.get("bad_reward")  or float("nan"))

        env.close()

        N = len(chosen_rams)
        np.savez_compressed(
            out_path,
            chosen_ram       = np.stack(chosen_rams).astype(np.uint8),
            chosen_actions   = np.stack(chosen_acts).astype(np.uint8),
            rejected_ram     = np.stack(rejected_rams).astype(np.uint8),
            rejected_actions = np.stack(rejected_acts).astype(np.uint8),
            chosen_len       = np.array(chosen_lens,   dtype=np.int16),
            rejected_len     = np.array(rejected_lens, dtype=np.int16),
            pivot            = np.array(pivots,        dtype=np.int16),
            kind             = np.array(kinds,         dtype=np.uint8),
            good_reward      = np.array(good_rewards,  dtype=np.float32),
            bad_reward       = np.array(bad_rewards,   dtype=np.float32),
            n_pairs          = np.array(N,             dtype=np.int32),
            level            = np.array(self.level,    dtype=np.int32),
        )
        print(f"  {self.uuid}: {N} pairs → {out_path}")
        return out_path

    def __repr__(self) -> str:
        return f"DPOGraphSample(uuid={self.uuid!r}, level={self.level})"
