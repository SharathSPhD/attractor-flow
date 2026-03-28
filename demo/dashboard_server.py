#!/usr/bin/env python3
"""Serve the AttractorFlow proof-of-work dashboard on port 8002."""
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = os.path.join(HERE, "index.html")
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"AttractorFlow dashboard running at http://localhost:{port}", flush=True)
    server.serve_forever()
