"""
Fix Python path so that robocasa and robosuite packages resolve correctly
even when running from the project root directory.

The project root (/media/razor/Razer/HKU_Dissertation/) contains bare
robocasa/ and robosuite/ repo directories (without __init__.py) which
shadow the properly installed editable packages. This module fixes that.

Usage: import this BEFORE importing robocasa or robosuite.
    import src.path_fix  # noqa: F401
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Remove cwd and project root from sys.path to prevent shadowing
_clean = []
_shadowed = []
for p in sys.path:
    if p in ("", PROJECT_ROOT):
        _shadowed.append(p)
    else:
        _clean.append(p)

# Put shadowed entries at the END so site-packages take precedence
sys.path = _clean + _shadowed
