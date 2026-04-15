import json
import os
from pathlib import Path

from .config import get_providers

APP_NAME = "A5Tool"


def _key_name(provider: str) -> str:
    return f"{provider.upper()}_API_KEY"


def get_config_path() -> Path:
    return Path.home() / APP_NAME / "config.json"


def load_api_keys() -> dict[str, str] | None:
    keys: dict[str, str] = {}

    for provider in get_providers():
        env_val = os.getenv(_key_name(provider))
        if env_val:
            keys[_key_name(provider)] = env_val

    if keys:
        return keys

    config_path = get_config_path()
    if not config_path.exists():
        return None

    data = json.loads(config_path.read_text(encoding="utf-8"))
    keys.update(data)
    return keys


def save_api_keys(api_keys: dict[str, str]) -> None:
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(api_keys, indent=2), encoding="utf-8")