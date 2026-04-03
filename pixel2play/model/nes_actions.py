"""
NES action encoding: keyboard key strings → integer class indices.

D-pad (WASD) → 9 classes
Buttons (j=A/jump, f=B/fire) → 4 classes
"""

from typing import List, Tuple

# fmt: off
DPAD_ENCODING = {
    frozenset():            0,
    frozenset({'a'}):       1,  # left
    frozenset({'d'}):       2,  # right
    frozenset({'w'}):       3,  # up
    frozenset({'s'}):       4,  # down
    frozenset({'w', 'a'}): 5,  # up-left
    frozenset({'w', 'd'}): 6,  # up-right
    frozenset({'s', 'a'}): 7,  # down-left
    frozenset({'s', 'd'}): 8,  # down-right
}
BUTTON_ENCODING = {
    frozenset():            0,
    frozenset({'j'}):       1,  # A only  (jump)
    frozenset({'f'}):       2,  # B only  (fire)
    frozenset({'j', 'f'}): 3,  # A + B
}
# fmt: on

N_DPAD    = len(DPAD_ENCODING)   # 9
N_BUTTONS = len(BUTTON_ENCODING) # 4

DPAD_KEYS   = frozenset({'w', 'a', 's', 'd'})
BUTTON_KEYS = frozenset({'j', 'f'})


def encode(keys: List[str]) -> Tuple[int, int]:
    pressed = frozenset(keys)
    dpad   = DPAD_ENCODING.get(pressed & DPAD_KEYS,   0)
    button = BUTTON_ENCODING.get(pressed & BUTTON_KEYS, 0)
    return dpad, button
