import json
import os
from pathlib import Path

APP_NAME = "A5Tool"

def get_config_path() -> Path:
    return Path.home() / APP_NAME / "config.json"

def load_api_keys() -> dict[str, str] | None:

    keys = {}
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        keys["OPENAI_API_KEY"] = env_key
    env_key = os.getenv("ANTHROPIC_API_KEY")
    if env_key:
        keys["ANTHROPIC_API_KEY"] = env_key
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        keys["GEMINI_API_KEY"] = env_key

    if keys:
        return keys

    config_path = get_config_path()
    if not config_path.exists():
        return None

    data = json.loads(config_path.read_text(encoding="utf-8"))
    keys.update(data)
    return keys

def save_api_key(api_key: str) -> None:
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = {"OPENAI_API_KEY": api_key}
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")