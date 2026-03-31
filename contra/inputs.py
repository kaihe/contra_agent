# NES buttons: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
#
# Two-head action space: agent outputs (dpad_idx, button_idx) independently.
# The NES action is the bitwise OR of the two selected rows.

# Head 1 — D-pad (7 options)
DPAD_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: none
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 1: Right
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 2: Left
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3: Up
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 4: Down
    [0, 0, 0, 0, 1, 0, 0, 1, 0],  # 5: Up+Right
    [0, 0, 0, 0, 0, 1, 0, 1, 0],  # 6: Down+Right
]
DPAD_NAMES = ["_", "R", "L", "U", "D", "UR", "DR"]

# Head 2 — Buttons (4 options)
BUTTON_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: none
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: Fire (B)
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 2: Jump (A)
    [1, 0, 0, 0, 0, 0, 0, 0, 1],  # 3: Fire+Jump
]
BUTTON_NAMES = ["_", "F", "J", "FJ"]

NUM_DPAD    = len(DPAD_TABLE)
NUM_BUTTONS = len(BUTTON_TABLE)
