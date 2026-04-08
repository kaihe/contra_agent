# NES buttons: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
#
# Two-head action space: agent outputs (dpad_idx, button_idx) independently.
# The NES action is the bitwise OR of the two selected rows.

# Head 1 — D-pad (9 options), matches DPAD_ENCODING in nes_actions.py
DPAD_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: none
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 1: Left
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 2: Right
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3: Up
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 4: Down
    [0, 0, 0, 0, 1, 0, 1, 0, 0],  # 5: Up+Left
    [0, 0, 0, 0, 1, 0, 0, 1, 0],  # 6: Up+Right
    [0, 0, 0, 0, 0, 1, 1, 0, 0],  # 7: Down+Left
    [0, 0, 0, 0, 0, 1, 0, 1, 0],  # 8: Down+Right
]
DPAD_NAMES = ["_", "L", "R", "U", "D", "UL", "UR", "DL", "DR"]

# Head 2 — Buttons (4 options), matches BUTTON_ENCODING in nes_actions.py
BUTTON_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: none
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 1: Jump (A)
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2: Fire (B)
    [1, 0, 0, 0, 0, 0, 0, 0, 1],  # 3: Fire+Jump
]
BUTTON_NAMES = ["_", "J", "F", "FJ"]

NUM_DPAD    = len(DPAD_TABLE)
NUM_BUTTONS = len(BUTTON_TABLE)
