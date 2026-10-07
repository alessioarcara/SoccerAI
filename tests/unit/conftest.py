import sys
from pathlib import Path

# make `tests/unit/factories.py` importable as `factories`
sys.path.insert(0, str(Path(__file__).parent))
