# config_loader.py — has TWO bugs
import asyncio, json, os

async def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data
    except:
        return None  # BUG 1: silently swallows ALL exceptions

async def get_db_config() -> dict:
    cfg = await load_config(os.getenv("CONFIG_PATH", "config.json"))
    return cfg.get("database", {})  # BUG 2: NoneType has no .get()
