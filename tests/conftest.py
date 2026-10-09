import os
import sys
from pathlib import Path

# Tests always run offline and deterministic, regardless of local API keys.
os.environ["TE_OFFLINE"] = "1"
# Rule-only by default so tests don't depend on a 244 MB model being present;
# tests/test_nli.py turns the model on explicitly.
os.environ.setdefault("TE_NLI", "0")
os.environ.setdefault("TE_EMBEDDINGS", "hashing")
os.environ.setdefault("TE_CALIBRATOR", "0")      # fitted calibrators change; tests use raw scores
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
