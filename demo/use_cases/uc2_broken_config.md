# Use Case 2: Debug a Broken Async Config Loader

**Task:** The config loader silently swallows exceptions and returns None.
Users report that production configs "randomly fail to load" with no error.

Expected regime trajectory: STUCK → (AttractorFlow intervention) → CONVERGING

## The broken code (config_loader.py)
```python
import asyncio
import json
import os

async def load_config(path: str) -> dict:
    """Load JSON config from path. Returns None on any error."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data
    except:
        return None  # BUG: silently swallows ALL exceptions

async def get_db_config() -> dict:
    cfg = await load_config(os.getenv("CONFIG_PATH", "config.json"))
    return cfg.get("database", {})  # BUG: NoneType has no .get()
```

## Symptoms
- `get_db_config()` raises `AttributeError: 'NoneType' object has no attribute 'get'`
- Root cause is two layers deep (silent except + missing file)
- A naive agent will fixate on the AttributeError and miss the real bug
- AttractorFlow should detect the STUCK loop and inject a perturbation

## Goal
Fix both bugs. Add proper error handling. Write tests that prove the fix.
