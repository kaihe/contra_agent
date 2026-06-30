"""DreamerV3 for Contra (NES), built component-by-component.

See dreamer/README.md for the build ladder. Each component has a standalone
verification gate; nothing downstream is trusted until its gate is green.
"""

import os

OUT_DIR = "tmp/dreamer"


def out_path(name: str) -> str:
    """Path under tmp/dreamer/ for a saved artifact (creates the dir)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)
