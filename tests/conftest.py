import os
import sys
from pathlib import Path

# Tests always run offline and deterministic, regardless of local API keys.
os.environ["TE_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
