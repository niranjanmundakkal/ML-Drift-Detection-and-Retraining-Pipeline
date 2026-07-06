"""Path manager for project directories"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def data_dir(name: str) -> Path:
    return ROOT / "data" / name
