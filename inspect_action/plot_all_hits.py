import numpy as np
import glob
import os
import matplotlib.pyplot as plt

TRACE_DIR = os.path.join(os.path.dirname(__file__), "human_trace")

win_files = glob.glob(os.path.join(TRACE_DIR, "win_D3072_*.npz"))
lose_files = glob.glob(os.path.join(TRACE_DIR, "lose_D3072_*.npz"))

def extract_boss_hit_counters(filepath):
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
    
    # Extract Hit Counters
    # 1412: Left Cannon (usually ~16 HP)
    # 1415: Right Cannon (usually ~16 HP) 
    # 1414: Main Core (usually ~32 HP)
    
    # Clean up the "240" dead state so scales correctly
    left_cannon = np.where(boss_ram[:, 1412] == 240, 0, boss_ram[:, 1412])
    right_cannon = np.where(boss_ram[:, 1415] == 240, 0, boss_ram[:, 1415])
    core = np.where(boss_ram[:, 1414] == 240, 0, boss_ram[:, 1414])
    
    return left_cannon, right_cannon, core

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Normalize line length by dividing by the max frames if we want, or just plot raw frames
for w_file in win_files:
    result = extract_boss_hit_counters(w_file)
    if result is None: continue
    win_lc, win_rc, win_core = result
    frames_win = np.arange(len(win_lc))
    
    # We only need the label for the first line to prevent legend clutter
    label_lc = "Left Cannon (1412)" if w_file == win_files[0] else None
    label_rc = "Right Cannon (1415)" if w_file == win_files[0] else None
    label_core = "Main Core (1414)" if w_file == win_files[0] else None

    # Plot lines with some transparency
    ax1.plot(frames_win, win_lc, color="blue", linewidth=1.5, alpha=0.4, label=label_lc)
    ax1.plot(frames_win, win_rc, '--', color="orange", linewidth=1.5, alpha=0.4, label=label_rc)
    ax1.plot(frames_win, win_core, ':', color="red", linewidth=2.5, alpha=0.6, label=label_core)

ax1.set_title(f"All {len(win_files)} WIN Traces: Hit Counters steadily drop to 0")
ax1.set_xlabel("Frames Elapsed in Boss Fight")
ax1.set_ylabel("Hits Remaining")
ax1.set_ylim(-2, 35)
ax1.legend()
ax1.grid(True, alpha=0.3)


for l_file in lose_files:
    result = extract_boss_hit_counters(l_file)
    if result is None: continue
    lose_lc, lose_rc, lose_core = result
    
    # Quick fix for the anomaly trace "S87" where player killed boss but still died
    if np.min(lose_core) == 0:
        continue

    frames_lose = np.arange(len(lose_lc))
    label_lc = "Left Cannon (1412)" if l_file == lose_files[0] else None
    label_rc = "Right Cannon (1415)" if l_file == lose_files[0] else None
    label_core = "Main Core (1414)" if l_file == lose_files[0] else None

    # Plot lines
    ax2.plot(frames_lose, lose_lc, color="blue", linewidth=1.5, alpha=0.4, label=label_lc)
    ax2.plot(frames_lose, lose_rc, '--', color="orange", linewidth=1.5, alpha=0.4, label=label_rc)
    ax2.plot(frames_lose, lose_core, ':', color="red", linewidth=2.5, alpha=0.6, label=label_core)

ax2.set_title(f"Typical LOSE Traces: Player dies instantly, Boss takes barely any hits")
ax2.set_xlabel("Frames Elapsed in Boss Fight")
ax2.set_ylim(-2, 35)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), "boss_hit_counters_all_games.png")
plt.savefig(save_path, dpi=200)
print(f"Plot saved successfully to {save_path}")
