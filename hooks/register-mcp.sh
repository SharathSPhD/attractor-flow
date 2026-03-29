#!/usr/bin/env bash
# Auto-registers attractorflow_mcp in ~/.claude/settings.json on every session start.
# Idempotent — safe to run multiple times.

SETTINGS="$HOME/.claude/settings.json"
PLUGIN_CACHE="$HOME/.claude/plugins/cache/attractor-flow"
MCP_CMD='PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-'"$PLUGIN_CACHE"'}" && cd "$PLUGIN_ROOT/attractorflow/mcp-server" && PATH="$PATH:$HOME/.local/bin" uv run server.py'

python3 - <<'PYEOF'
import json, os, sys

settings_path = os.path.expanduser("~/.claude/settings.json")
plugin_cache  = os.path.expanduser("~/.claude/plugins/cache/attractor-flow")
mcp_cmd       = (
    'PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-' + plugin_cache + '}" '
    '&& cd "$PLUGIN_ROOT/attractorflow/mcp-server" '
    '&& PATH="$PATH:$HOME/.local/bin" uv run --no-project server.py'
)

try:
    with open(settings_path) as f:
        d = json.load(f)
except Exception:
    sys.exit(0)  # can't read settings — skip silently

servers = d.setdefault("mcpServers", {})
if "attractorflow_mcp" in servers:
    sys.exit(0)  # already registered

servers["attractorflow_mcp"] = {
    "command": "sh",
    "args": ["-c", mcp_cmd],
    "type": "stdio"
}

with open(settings_path, "w") as f:
    json.dump(d, f, indent=2)

print("[attractor-flow] Registered attractorflow_mcp in ~/.claude/settings.json")
PYEOF
