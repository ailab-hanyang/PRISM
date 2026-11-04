from pathlib import Path
import shutil
import importlib.util
import sys
from typing import Any, Dict, Optional

import yaml


def _serialize_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _serialize_config_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_config_value(v) for v in value]
    return repr(value)


def config_to_dict(config: Any) -> Dict[str, Any]:
    config_dict: Dict[str, Any] = {}

    for key in dir(config):
        if key.startswith("_"):
            continue

        value = getattr(config, key)
        if callable(value):
            continue

        config_dict[key] = _serialize_config_value(value)

    return config_dict


def save_config_files(save_dir: Path, timestamp: str, config_file_path: str):

    save_dir.mkdir(parents=True, exist_ok=True)

    if Path(config_file_path).exists():
        dest_path = save_dir / f"{timestamp}_{Path(config_file_path).name}"
        shutil.copy2(config_file_path, dest_path)
        print(f"Config saved: {dest_path}")
    else:
        print(f"Config file not found: {config_file_path}")

    print(f"All config files saved to: {save_dir}")

def import_module_from_file(module_name: str, file_path: str):

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module
