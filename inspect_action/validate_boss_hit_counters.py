import numpy as np
import glob
import os
import matplotlib.pyplot as plt

TRACE_DIR = os.path.join(os.path.dirname(__file__), "human_trace")

def extract_boss_metrics(filepath):
    data = np.load(filepath)
    ram = data['ram']
    
    # Isolate Boss Fight
    if 'xscroll' in data:
        boss_idx = np.where(data['xscroll'] >= 3072)[0]
        if len(boss_idx) > 0:
            start_idx = min(boss_idx[0], len(ram) - 1)
            end_idx = boss_idx[-1]
        else:
            start_idx = max(0, len(ram) - 300)
            end_idx = len(ram) - 1
    else:
        start_idx = max(0, len(ram) - 300)
        end_idx = len(ram) - 1

    # Don't return extremely short fights (e.g., instant deaths)
    if end_idx <= start_idx + 10:
        return None

    boss_ram = ram[start_idx:end_idx+1]
    
    # -------------------------------------------------------------
    # True Bullet-by-Bullet Hit Counters (Drops by 1 per bullet)
    # -------------------------------------------------------------
    # We clean up the "240" dead state so scales correctly to 0.
    # 1412: Left Cannon (~16 hits)
    # 1415: Right Cannon (~16 hits) 
    # 1414: Main Core (~32 hits)
    hit_lc = np.where(boss_ram[:, 1412] == 240, 0, boss_ram[:, 1412])
    hit_rc = np.where(boss_ram[:, 1415] == 240, 0, boss_ram[:, 1415])
    hit_core = np.where(boss_ram[:, 1414] == 240, 0, boss_ram[:, 1414])
    
    # -------------------------------------------------------------
    # State-Based Structural HP (Drops 63 -> 15 -> 0)
    # -------------------------------------------------------------
    hp_lc = boss_ram[:, 1698]
    hp_rc = boss_ram[:, 1702]
    hp_core = boss_ram[:, 1706]
    
    return {
        'hits': (hit_lc, hit_rc, hit_core),
        'hp': (hp_lc, hp_rc, hp_core)
    }

def plot_health_metrics():
    win_files = glob.glob(os.path.join(TRACE_DIR, "win_D3072_*.npz"))
    lose_files = glob.glob(os.path.join(TRACE_DIR, "lose_D3072_*.npz"))

    # Plot Hit Counters (Bullet-by-Bullet)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for w_file in win_files:
        metrics = extract_boss_metrics(w_file)
        if not metrics: continue
        win_lc, win_rc, win_core = metrics['hits']
        frames_win = np.arange(len(win_lc))
        
        lbl_lc = "Left Cannon (1412)" if w_file == win_files[0] else None
        lbl_rc = "Right Cannon (1415)" if w_file == win_files[0] else None
        lbl_core = "Main Core (1414)" if w_file == win_files[0] else None

        ax1.plot(frames_win, win_lc, color="blue", linewidth=1.5, alpha=0.4, label=lbl_lc)
        ax1.plot(frames_win, win_rc, '--', color="orange", linewidth=1.5, alpha=0.4, label=lbl_rc)
        ax1.plot(frames_win, win_core, ':', color="red", linewidth=2.5, alpha=0.6, label=lbl_core)

    ax1.set_title(f"All WIN Traces: Bullet Hit Counters Drop Smoothly")
    ax1.set_xlabel("Frames Elapsed in Boss Fight")
    ax1.set_ylabel("Remaining Hits")
    ax1.set_ylim(-2, 35)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for l_file in lose_files:
        metrics = extract_boss_metrics(l_file)
        if not metrics: continue
        lose_lc, lose_rc, lose_core = metrics['hits']
        
        if np.min(lose_core) == 0: continue # Skip the anomaly where player killed boss but still died

        frames_lose = np.arange(len(lose_lc))
        lbl_lc = "Left Cannon (1412)" if l_file == lose_files[0] else None
        lbl_rc = "Right Cannon (1415)" if l_file == lose_files[0] else None
        lbl_core = "Main Core (1414)" if l_file == lose_files[0] else None

        ax2.plot(frames_lose, lose_lc, color="blue", linewidth=1.5, alpha=0.4, label=lbl_lc)
        ax2.plot(frames_lose, lose_rc, '--', color="orange", linewidth=1.5, alpha=0.4, label=lbl_rc)
        ax2.plot(frames_lose, lose_core, ':', color="red", linewidth=2.5, alpha=0.6, label=lbl_core)

    ax2.set_title(f"Typical LOSE Traces: Player dies instantly, Boss unharmed")
    ax2.set_xlabel("Frames Elapsed in Boss Fight")
    ax2.set_ylim(-2, 35)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "boss_analysis.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Unified analysis plot saved to {plot_path}")

if __name__ == "__main__":
    print("-" * 60)
    print("Contra Boss Health & Hit Counter Discovery")
    print("-" * 60)
    print("This script validates the memory addresses for the Level 1 Boss:")
    print("  Structural HP (63->15->0) : [1698, 1702, 1706]")
    print("  Bullet Hits (~32->0)      : [1412, 1415, 1414]")
    print("-" * 60)
    plot_health_metrics()
