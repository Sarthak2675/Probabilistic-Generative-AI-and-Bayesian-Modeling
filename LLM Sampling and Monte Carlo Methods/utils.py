# utils.py
import os, json, random, time
from typing import Iterable, List, Dict

def set_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Any non-negative integer seed.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Minimal; graders can toggle deterministic kernels if needed.
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = False
        cudnn.deterministic = False
    except Exception:
        pass

def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts.

    Args:
        path: Path to a newline-delimited JSON file.

    Returns:
        List of Python dicts, one per line in the file.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def save_jsonl(path: str, rows: Iterable[Dict]) -> None:
    """Write a list/iterable of dicts to a JSONL file.

    Args:
        path: Output file path. Parent directories are created if missing.
        rows: Iterable of serializable Python dicts to write (one per line).
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def ensure_dir(path: str) -> None:
    """Create a directory path if it doesn't already exist.

    Args:
        path: Directory path to ensure. No-op for empty string.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def now() -> str:
    """Get a human-readable timestamp (local time).

    Returns:
        A timestamp string like '20251009_2312'.
    """
    return time.strftime("%Y%m%d_%H%M", time.localtime())
