import numpy as np
import glob
import os
import matplotlib.pyplot as plt

TRACE_DIR = os.path.join(os.path.dirname(__file__), "human_trace")

win_files = glob.glob(os.path.join(TRACE_DIR, "win_D3072_*.npz"))
lose_files = glob.glob(os.path.join(TRACE_DIR, "lose_D3072_*.npz"))

def extract_boss_hp(filepath):
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

    # Don't plot extremely short fights (e.g., instant deaths)
    if end_idx <= start_idx + 10:
        return None

    boss_ram = ram[start_idx:end_idx+1]
    
    # Extract Left Cannon(1698), Right Cannon(1702), Core(1706)
    left_cannon = boss_ram[:, 1698]
    right_cannon = boss_ram[:, 1702]
    core = boss_ram[:, 1706]
    
    return left_cannon, right_cannon, core

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Normalize line length by dividing by the max frames if we want, or just plot raw frames
for w_file in win_files:
    result = extract_boss_hp(w_file)
    if result is None: continue
    win_lc, win_rc, win_core = result
    frames_win = np.arange(len(win_lc))
    
    # We only need the label for the first line to prevent legend clutter
    label_lc = "Left Cannon (1698)" if w_file == win_files[0] else None
    label_rc = "Right Cannon (1702)" if w_file == win_files[0] else None
    label_core = "Main Core (1706)" if w_file == win_files[0] else None

    # Plot lines with some transparency
    ax1.plot(frames_win, win_lc, color="blue", linewidth=1.5, alpha=0.4, label=label_lc)
    ax1.plot(frames_win, win_rc, '--', color="orange", linewidth=1.5, alpha=0.4, label=label_rc)
    ax1.plot(frames_win, win_core, ':', color="red", linewidth=2.5, alpha=0.6, label=label_core)

ax1.set_title(f"All {len(win_files)} WIN Traces: HP consistently cascades to 0")
ax1.set_xlabel("Frames Elapsed in Boss Fight")
ax1.set_ylabel("HP Value")
ax1.set_ylim(-5, 70)
ax1.legend()
ax1.grid(True, alpha=0.3)


for l_file in lose_files:
    result = extract_boss_hp(l_file)
    if result is None: continue
    lose_lc, lose_rc, lose_core = result
    
    # Quick fix for the anomaly trace "S87" where player killed boss but still died
    if np.min(lose_core) == 0:
        print(f"Skipping anomaly LOSE trace where boss was actually killed: {os.path.basename(l_file)}")
        continue

    frames_lose = np.arange(len(lose_lc))
    label_lc = "Left Cannon (1698)" if l_file == lose_files[0] else None
    label_rc = "Right Cannon (1702)" if l_file == lose_files[0] else None
    label_core = "Main Core (1706)" if l_file == lose_files[0] else None

    # Plot lines
    ax2.plot(frames_lose, lose_lc, color="blue", linewidth=1.5, alpha=0.4, label=label_lc)
    ax2.plot(frames_lose, lose_rc, '--', color="orange", linewidth=1.5, alpha=0.4, label=label_rc)
    ax2.plot(frames_lose, lose_core, ':', color="red", linewidth=2.5, alpha=0.6, label=label_core)

ax2.set_title(f"All {len(lose_files)-1} Typical LOSE Traces: Player dies instantly, HP stays flat")
ax2.set_xlabel("Frames Elapsed in Boss Fight")
ax2.set_ylim(-5, 70)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), "boss_hp_all_games.png")
plt.savefig(save_path, dpi=200)
print(f"Plot saved successfully to {save_path}")
