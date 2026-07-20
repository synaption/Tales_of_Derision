#!/usr/bin/env python3
"""Static file server that enables cross-origin isolation (COOP/COEP).

pygbag/emscripten can only run audio on a worker thread (glitch-free) when the
page is *cross-origin isolated*, which the browser only grants when these
response headers are present:

    Cross-Origin-Opener-Policy:   same-origin
    Cross-Origin-Embedder-Policy: credentialless   (or require-corp)

Python's http.server does not send them, so this small server does. Local
preview only -- GitHub Pages can't set headers, so it needs a coi-serviceworker
instead (added separately once we confirm this helps).

Usage: python3 serve_coi.py <directory> <port>

COEP mode is overridable via env var, e.g. `COEP=require-corp python3 serve_coi.py ...`.
`credentialless` (default) keeps the remote pygbag CDN working; `require-corp`
is stricter and needs every cross-origin asset to send CORP/CORS.
"""

from __future__ import annotations

import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import sys


COEP_MODE = os.environ.get("COEP", "credentialless")


class COIHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", COEP_MODE)
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        # Avoid serving stale cached assets while iterating on the build.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        pass


def main() -> None:
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    handler = functools.partial(COIHandler, directory=directory)
    with ThreadingHTTPServer(("", port), handler) as httpd:
        print(f"COI server on http://localhost:{port}  (COEP={COEP_MODE}, dir={directory})")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
