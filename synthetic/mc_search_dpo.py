"""
mc_search_dpo.py — MC search that builds a directed graph for DPO pair extraction.

Each committed action is a SearchNode with `next` (committed continuation) and
`dead` (rewound branch) pointers.  After search the graph encodes:
  - Good trace:   follow `next` pointers from root → committed sequence
  - Bad branches: at any node where `dead` is set, follow that chain → the
                  sequence abandoned when the search rewound to that node

DPO pairs (extracted in a later pass) compare the committed replacement path
against each dead branch at the rewind branch node.

Usage:
    python synthetic/mc_search_dpo.py --level 1 --rollout-len 48
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional
import multiprocessing as mp
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import pickle

import numpy as np
import stable_retro as retro

from contra.replay import rewind_state, step_env
from contra.events import (
    compute_reward, scan_events, get_level,
    EV_PLAYER_DIE, EV_GAME_CLEAR, ADDR_LEVEL_ROUTINE,
)

from synthetic.mc_search import (
    GAME, SKIP, TRACE_DIR, DEFAULT_STATE_BY_LEVEL, FPS,
    run_random_rollout, save_trace, _load_bigram,
    _worker_init, _worker_rollout,
)

DPO_DIR = os.path.join(os.path.dirname(__file__), "mc_graph")
_DPO_CHUNK_LEN = 128
_NES_KEY_MAP = [(0, "f"), (4, "w"), (5, "s"), (6, "a"), (7, "d"), (8, "j")]


@dataclass
class SearchNode:
    """One committed step in the MC search graph.

    `next`      — committed continuation (good path forward).
    `dead`      — branches abandoned because all rollouts died (high-contrast rejection).
    `secondary` — branches abandoned because a better path won, but the branch survived
                  (low-contrast rejection; only recorded when second_best < best reward).
    """
    action: Optional[np.ndarray]  # None only for the root node
    reward: float                 # cumulative reward up to this step
    next: Optional[SearchNode]  = field(default=None,         repr=False)
    dead: list[SearchNode]      = field(default_factory=list, repr=False)
    secondary: list[SearchNode] = field(default_factory=list, repr=False)


def _graph_actions(root: SearchNode) -> list[np.ndarray]:
    """Return committed actions in order by following `next` from root."""
    actions = []
    node = root.next
    while node is not None:
        actions.append(node.action)
        node = node.next
    return actions


def _flatten_graph(root: SearchNode) -> list[dict]:
    """Iteratively flatten the tree into a list to prevent pickle RecursionError."""
    node_list = []
    id_to_idx = {id(root): 0}
    queue = [root]
    
    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        
        children = []
        if curr.next: children.append(curr.next)
        children.extend(curr.dead)
        children.extend(curr.secondary)
        
        for child in children:
            if id(child) not in id_to_idx:
                id_to_idx[id(child)] = len(queue)
                queue.append(child)
                
    for curr in queue:
        node_list.append({
            "action": curr.action,
            "reward": curr.reward,
            "next": id_to_idx[id(curr.next)] if curr.next else None,
            "dead": [id_to_idx[id(n)] for n in curr.dead],
            "secondary": [id_to_idx[id(n)] for n in curr.secondary],
        })
    return node_list


def _unflatten_graph(flat_nodes: list[dict]) -> SearchNode:
    """Iteratively reconstruct the tree from a flat list."""
    nodes = [SearchNode(action=fn["action"], reward=fn["reward"]) for fn in flat_nodes]
    for i, fn in enumerate(flat_nodes):
        if fn["next"] is not None:
            nodes[i].next = nodes[fn["next"]]
        nodes[i].dead = [nodes[j] for j in fn.get("dead", [])]
        nodes[i].secondary = [nodes[j] for j in fn.get("secondary", [])]
    return nodes[0]


def save_graph(root: SearchNode, path: str, init_emu: bytes) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flat_root = _flatten_graph(root)
    with open(path, "wb") as f:
        pickle.dump({"root": flat_root, "init_emu": init_emu}, f)


class _SearchNodeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "SearchNode":
            return SearchNode
        return super().find_class(module, name)


def load_graph(path: str) -> tuple[SearchNode, bytes]:
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 50_000))
    try:
        with open(path, "rb") as f:
            d = _SearchNodeUnpickler(f).load()
    finally:
        sys.setrecursionlimit(old_limit)
        
    root_data = d["root"]
    if isinstance(root_data, list):
        root = _unflatten_graph(root_data)
    else:
        root = root_data
        
    return root, d["init_emu"]


def search_and_build_graph(
    env,
    initial_emu_state: bytes,
    rollouts: int,
    rollout_len: int,
    max_time: int,
    level: int = 1,
    max_rewind: int = 30,
    max_actions: int = 4000,
    goal: str = "level_up",
    secondary_prob: float = 0.25,
    verbose: bool = True,
    pool=None,
) -> tuple[SearchNode, bool]:
    """Run MC search and return (graph_root, success).

    The root node holds the initial emulator state with action=None.
    Every committed step is a SearchNode linked via `.next`.
    Every rewind attaches the abandoned chain to the branch node's `.dead`.
    """
    root    = SearchNode(action=None, reward=0.0)
    chain:      list[SearchNode] = []  # committed nodes in order
    chain_emus: list[bytes]      = []  # emu state after each committed step
    current     = root
    current_emu = initial_emu_state    # emu state at current node
    done        = False

    t_start          = time.time()
    pending_events:  list[str] = []
    current_level    = level

    if verbose:
        print(f"\n  {'step':>4}  {'reward':>7}  {'death':>5}  {'time':>7}  event")
        print("  " + "-" * 50)

    while True:
        elapsed = time.time() - t_start
        if elapsed > max_time:
            if verbose:
                print(f"\n  ⏱ Time budget exhausted ({max_time:.0f}s)")
            break
        if len(chain) >= max_actions:
            if verbose:
                print(f"\n  ✂ Action limit reached ({max_actions}), abandoning trace")
            break
        if done:
            if verbose:
                print(f"\n  🏆 WIN!  time={elapsed:.1f}s  steps={len(chain)}")
            break

        # ── 0. Level transition → mark done immediately ───────────────────────
        ram_now = env.unwrapped.get_ram()
        if int(ram_now[ADDR_LEVEL_ROUTINE]) in (0x08, 0x09):
            done = True
            continue

        # ── 1. Monte Carlo lookahead ──────────────────────────────────────────
        task = (current_emu, rollout_len, current_level)
        if pool is not None:
            rollout_results = pool.map(_worker_rollout, [task] * rollouts)
        else:
            rollout_results = [
                run_random_rollout(env, current_emu, rollout_len, current_level)
                for _ in range(rollouts)
            ]
            rewind_state(env, current_emu)

        best_seq, best_reward, best_died = None, -float("inf"), True
        second_best_seq, second_best_reward = None, -float("inf")
        died_count = 0
        for seq, reward, died in rollout_results:
            if died:
                died_count += 1
            if reward > best_reward:
                second_best_seq, second_best_reward = best_seq, best_reward
                best_reward, best_seq, best_died = reward, seq, died
            elif reward > second_best_reward:
                second_best_reward, second_best_seq = reward, seq

        death_rate = died_count / rollouts

        # ── 2. Rewind if all rollouts die ─────────────────────────────────────
        if best_died:
            n         = len(chain)
            rewind_to = max(0, n - np.random.randint(1, min(max_rewind, max(n, 1)) + 1)) if n > 0 else 0
            if verbose:
                ev_col = " ".join(pending_events) if pending_events else ""
                print(f"  {n:4d}  {current.reward:7.1f}  {death_rate:5.2f}  {elapsed:6.1f}s  {ev_col}⏪ →{rewind_to}")
                pending_events.clear()

            # ── extend dead chain tail with the dying rollout ─────────────────
            prev = current
            for act in best_seq:
                node = SearchNode(action=act.copy(), reward=prev.reward)
                prev.next = node
                prev = node

            branch_node = root if rewind_to == 0 else chain[rewind_to - 1]
            if rewind_to < n:
                branch_node.dead.append(chain[rewind_to])
            elif current.next is not None:
                # n == 0: rollout hangs directly off root/current
                branch_node.dead.append(current.next)
            branch_node.next = None
            chain            = chain[:rewind_to]
            chain_emus       = chain_emus[:rewind_to]
            current          = branch_node
            current_emu      = initial_emu_state if rewind_to == 0 else chain_emus[rewind_to - 1]
            rewind_state(env, current_emu)
            continue

        # ── 3. Commit best rollout (partial) ─────────────────────────────────
        n        = len(best_seq)
        commit_n = np.random.randint(n // 2, n + 1) if n >= 2 else n

        # Attach second-best as a secondary branch (survived but scored less).
        # secondary_prob controls how many we keep — nearly every step qualifies.
        if (second_best_seq is not None and len(second_best_seq) > 0
                and second_best_reward < best_reward
                and np.random.random() < secondary_prob):
            head, prev_sb = None, None
            for act in second_best_seq:
                node = SearchNode(action=act.copy(), reward=current.reward)
                if head is None:
                    head = node
                if prev_sb is not None:
                    prev_sb.next = node
                prev_sb = node
            current.secondary.append(head)

        rewind_state(env, current_emu)
        prev = current
        for act in best_seq[:commit_n]:
            pre_ram = env.unwrapped.get_ram().copy()
            step_env(env, act)
            curr_ram = env.unwrapped.get_ram()

            if EV_PLAYER_DIE.trigger(pre_ram, curr_ram):
                raise RuntimeError(f"Death during commit at step {len(chain)}")

            node = SearchNode(
                action=act.copy(),
                reward=prev.reward + compute_reward(pre_ram, curr_ram),
            )
            prev.next = node
            chain.append(node)
            chain_emus.append(env.em.get_state())
            prev        = node
            current     = node
            current_emu = chain_emus[-1]

            new_level = get_level(curr_ram)
            if new_level != current_level:
                current_level = new_level
                if verbose:
                    pending_events.append(f"prior→Level{current_level}")
            if goal == "game_clear" and EV_GAME_CLEAR.trigger(pre_ram, curr_ram):
                done = True
            elif goal == "level_up" and new_level != level:
                done = True

            for ev in scan_events(pre_ram, curr_ram, len(chain)):
                tag = ev["tag"] + (f"({ev['detail']})" if ev["detail"] else "")
                pending_events.append(tag)

            if done:
                break

        step_num      = len(chain)
        prev_step_num = step_num - commit_n
        if verbose and ((step_num // 100) > (prev_step_num // 100) or done or pending_events):
            ev_col = " ".join(pending_events) if pending_events else ""
            print(f"  {step_num:4d}  {current.reward:7.1f}  {death_rate:5.2f}  {elapsed:6.1f}s  {ev_col}")
            pending_events.clear()

    return root, done


def _run_one_search(
    level, rollouts, rollout_len, max_time, max_rewind, max_actions,
    goal, workers, secondary_prob=0.25, verbose=False, instance_id=None,
):
    """Set up env+pool, run search, save trace if successful."""
    _load_bigram(level)
    if instance_id is not None:
        np.random.seed((os.getpid() + instance_id * 1337) % (2**32))

    state_label = DEFAULT_STATE_BY_LEVEL[level]
    use_spread  = level > 1
    prefix      = f"[i{instance_id}] " if instance_id is not None else ""

    pool = mp.Pool(workers, initializer=_worker_init,
                   initargs=(GAME, state_label, use_spread)) if workers > 1 else None
    env = retro.make(
        game=GAME,
        state=retro.State.NONE if use_spread else state_label,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )
    if use_spread:
        env.load_state(f"spread_gun_state/{state_label}", retro.data.Integrations.CUSTOM_ONLY)
    env.reset()
    initial_state = env.em.get_state()

    if prefix:
        print(f"{prefix}start  level={level}  workers={workers}", flush=True)

    root, done = search_and_build_graph(
        env, initial_state,
        rollouts=rollouts, rollout_len=rollout_len, max_time=max_time,
        level=level, max_rewind=max_rewind, max_actions=max_actions,
        goal=goal, secondary_prob=secondary_prob, verbose=verbose, pool=pool,
    )

    env.close()
    if pool:
        pool.close()
        pool.join()

    if not done:
        if prefix:
            print(f"{prefix}no win  steps={len(_graph_actions(root))}", flush=True)
        return None, None

    actions   = _graph_actions(root)
    date_str  = time.strftime("%Y%m%d%H%M%S" if instance_id is not None else "%Y%m%d%H%M")
    suffix    = f"_i{instance_id}" if instance_id is not None else ""
    level_tag = "game" if goal == "game_clear" else f"level{level}"

    trace_path = os.path.join(TRACE_DIR, f"win_{level_tag}_{date_str}{suffix}.npz")
    save_trace(initial_state, actions, trace_path, level=level)

    os.makedirs(DPO_DIR, exist_ok=True)
    graph_path = os.path.join(DPO_DIR, f"graph_{level_tag}_{date_str}{suffix}.pkl")
    save_graph(root, graph_path, initial_state)

    if prefix:
        print(f"{prefix}WIN  steps={len(actions)}  → {trace_path}", flush=True)
    print(f"graph: {len(actions)} committed + dead branches → {graph_path}")

    return trace_path, graph_path


def main():
    parser = argparse.ArgumentParser(description="MC search with graph-based DPO collection")
    parser.add_argument("--level",       type=int, default=1, choices=list(range(1, 9)))
    parser.add_argument("--rollouts",    type=int, default=128)
    parser.add_argument("--rollout-len", type=int, default=48)
    parser.add_argument("--max-rewind",  type=int, default=48)
    parser.add_argument("--max-time",    type=int, default=600)
    parser.add_argument("--workers",     type=int, default=os.cpu_count())
    parser.add_argument("--goal",        type=str, default="level_up",
                        choices=["level_up", "game_clear"])
    parser.add_argument("--max-actions",     type=int,   default=6000)
    parser.add_argument("--secondary-prob",  type=float, default=0.25,
                        help="Fraction of secondary branches to keep (0–1)")
    parser.add_argument("--dpo-dir",         type=str,   default=DPO_DIR)
    parser.add_argument("--no-verbose",      action="store_true", default=False)
    args = parser.parse_args()

    _load_bigram(args.level)
    np.random.seed(int(time.time() * 1000) % (2**32))

    verbose = not args.no_verbose
    if verbose:
        state_label = DEFAULT_STATE_BY_LEVEL[args.level]
        print("=" * 70)
        print("MC Search with Graph-based DPO Collection")
        print("=" * 70)
        print(f"  Game:           {GAME}")
        print(f"  Level:          {args.level}  ({state_label})")
        print(f"  Rollouts/Step:  {args.rollouts}")
        print(f"  Rollout Length: {args.rollout_len} actions ({args.rollout_len * SKIP} frames)")
        print(f"  DPO Chunk:      {_DPO_CHUNK_LEN}")
        print(f"  Workers:        {args.workers if args.workers > 1 else 1}")
        print(f"  Goal:           {args.goal}")
        print(f"  Max Actions:    {args.max_actions}")
        print(f"  Secondary Prob: {args.secondary_prob}")
        print(f"  Time Budget:    {args.max_time}s")
        print("=" * 70)

    _run_one_search(
        level=args.level, rollouts=args.rollouts, rollout_len=args.rollout_len,
        max_time=args.max_time, max_rewind=args.max_rewind, max_actions=args.max_actions,
        goal=args.goal, workers=args.workers, secondary_prob=args.secondary_prob,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
