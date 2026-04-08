import os
import stable_retro as retro

_here = os.path.dirname(os.path.abspath(__file__))
retro.data.add_custom_integration(os.path.join(_here, "integration"))
