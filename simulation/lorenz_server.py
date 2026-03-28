#!/usr/bin/env python3
"""
lorenz_server.py — HTTP wrapper for the Lorenz Demo

Runs the Lorenz simulation once and serves the ASCII phase portrait
and FTLE history as a live-reloading HTML page on http://localhost:8001
"""

import io
import sys
import contextlib
from http.server import BaseHTTPRequestHandler, HTTPServer

# Add mcp-server to path so imports resolve
sys.path.insert(0, "attractorflow/mcp-server")
sys.path.insert(0, "simulation")

# Run the demo and capture its stdout output once at startup
from demo_lorenz import main as lorenz_main

_captured = io.StringIO()
with contextlib.redirect_stdout(_captured):
    lorenz_main()
LORENZ_OUTPUT = _captured.getvalue()


def _to_html(text: str) -> str:
    lines = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="10">
  <title>AttractorFlow — Lorenz Demo</title>
  <style>
    body {{
      background: #0d1117;
      color: #c9d1d9;
      font-family: 'Courier New', monospace;
      font-size: 14px;
      padding: 2rem;
      white-space: pre;
      line-height: 1.5;
    }}
    h1 {{ color: #58a6ff; font-size: 1.1rem; margin-bottom: 1rem; white-space: normal; }}
    .badge {{
      display: inline-block;
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 4px;
      padding: 2px 8px;
      font-size: 12px;
      color: #8b949e;
      margin-bottom: 1rem;
      white-space: normal;
    }}
  </style>
</head>
<body><h1>AttractorFlow — Lorenz Strange Attractor</h1>
<span class="badge">Auto-refreshes every 10s &nbsp;|&nbsp; MCP server: http://localhost:8000/sse</span>

{lines}</body>
</html>"""


class LorenzHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = _to_html(LORENZ_OUTPUT).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # silence request logs


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8001))
    server = HTTPServer(("127.0.0.1", port), LorenzHandler)
    print(f"Lorenz Demo server running at http://localhost:{port}", flush=True)
    server.serve_forever()
