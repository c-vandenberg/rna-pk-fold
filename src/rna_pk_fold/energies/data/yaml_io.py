from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml


def read_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Read and parse a YAML file.
    """
    path_obj = Path(path)
    if path_obj.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError("Only YAML files are supported.")

    return yaml.safe_load(path_obj.read_text(encoding="utf-8")) or {}
