"""Plumbing shared by the model backends: HTTP with deadlines, and child
servers whose lifetime we own.

Split out of :mod:`chat` so a second backend (Letta, in :mod:`letta_backend`)
can start and stop its own server without the two modules importing each other.

Two rules live here, and both exist because the alternative is a villager who
thinks forever:

* every read has a deadline -- generous until the first token arrives (weights
  may be loading off a cold SD card), tight afterwards;
* a server we started is ours to kill, on any exit path; a server that was
  already running is never touched.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


PROBE_TIMEOUT = 0.4      # how long auto-detect waits for a local server
CHECK_TIMEOUT = 4.0      # the startup "is the model really there?" call
FIRST_TOKEN_TIMEOUT = float(os.environ.get("FRUITBRAINS_LOAD_TIMEOUT", 120))
STALL_TIMEOUT = 25.0     # mid-reply silence that means something has gone wrong
BOOT_TIMEOUT = 30.0      # a cold `ollama serve` on a Pi takes its time


class NoLocalModel(RuntimeError):
    """Raised at startup when no local model is serving. Message is user-facing."""


def _headers(token: str = "", body: bool = False) -> dict:
    head = {"Accept": "application/json"}
    if body:
        head["Content-Type"] = "application/json"
    if token:
        head["Authorization"] = f"Bearer {token}"
    return head


def request_json(method: str, url: str, payload: dict | None = None,
                 timeout: float = CHECK_TIMEOUT, token: str = "") -> dict | list | None:
    """One JSON call. ``None`` means "nothing answered" -- not "it said no"."""
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, method=method,
                                     headers=_headers(token, body=data is not None))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
            return json.loads(raw) if raw.strip() else {}
    except (urllib.error.URLError, OSError, ValueError):
        return None


def request_json_strict(method: str, url: str, payload: dict | None = None,
                        timeout: float = CHECK_TIMEOUT, token: str = "") -> dict | list:
    """Like :func:`request_json`, but a failure raises with the server's own words.

    Servers explain themselves in the body of a 4xx; swallowing that and saying
    "HTTP 422" instead is how a config mistake turns into an afternoon.
    """
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, method=method,
                                     headers=_headers(token, body=data is not None))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400].strip()
        raise RuntimeError(f"{method} {url} -> {exc.code}: {detail or exc.reason}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"{method} {url} -> {type(exc).__name__}: {exc}") from None
    except ValueError as exc:
        raise RuntimeError(f"{method} {url} -> unreadable reply: {exc}") from None


def get_json(url: str, timeout: float = PROBE_TIMEOUT, token: str = "") -> dict | list | None:
    return request_json("GET", url, None, timeout, token)


def post_json(url: str, payload: dict, timeout: float, token: str = "") -> dict | list | None:
    return request_json("POST", url, payload, timeout, token)


def post_stream(url: str, payload: dict, token: str = "",
                first_timeout: float | None = None, accept: str = "application/json"):
    """Yield response lines as they arrive, with a deadline on each read.

    Streaming is what makes a slow model bearable: the first token proves it is
    alive, and a stall becomes an error instead of a villager who thinks
    forever.  The read timeout starts generous (weights may be loading from a
    cold disk) and tightens once tokens are flowing.
    """
    data = json.dumps(payload).encode()
    head = dict(_headers(token, body=True), Accept=accept)
    request = urllib.request.Request(url, data=data, headers=head)
    with urllib.request.urlopen(request, timeout=first_timeout or FIRST_TOKEN_TIMEOUT) as response:
        first = True
        for line in response:
            if first:
                first = False
                try:                       # tighten the clock now that it is talking
                    response.fp.raw._sock.settimeout(STALL_TIMEOUT)
                except (AttributeError, OSError):
                    pass
            line = line.strip()
            if line:
                yield line


def reachable(url: str, timeout: float = PROBE_TIMEOUT) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True                        # it answered; a 404 is still someone home
    except (urllib.error.URLError, OSError, ValueError):
        return False


def log_tail(path: Path, lines: int = 12) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


class ManagedServer:
    """A server process we started, and are therefore responsible for.

    A server that was already running is never wrapped in one of these -- the
    game must not shut down something it did not start.
    """

    def __init__(self, process: subprocess.Popen, url: str, label: str = "server") -> None:
        self.process = process
        self.url = url
        self.label = label
        self._stopped = False
        atexit.register(self.stop)          # belt and braces for a hard exit

    def stop(self, timeout: float = 6.0) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self.process.poll() is not None:
            return
        try:
            # The child gets its own session, so the whole group goes down with it.
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except (AttributeError, OSError, ProcessLookupError):
            self.process.terminate()
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except (AttributeError, OSError, ProcessLookupError):
                self.process.kill()


def launch(label: str, argv: list[str], env: dict, log: Path, ready_url: str,
           timeout: float = BOOT_TIMEOUT, note: str = "") -> ManagedServer:
    """Start a server, wait for it to answer, and hand back its kill switch.

    Anything that goes wrong before it answers -- a crash, a port already taken,
    a timeout -- stops the child and raises with the tail of its log, because a
    server that died at boot always says why in the log and never on stdout.
    """
    print(f"Fruit Brains: starting {label} (log: {log}){note}", flush=True)
    handle = open(log, "ab")
    process = subprocess.Popen(argv, stdout=handle, stderr=handle, env=env,
                               start_new_session=True)
    server = ManagedServer(process, ready_url, label)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if reachable(ready_url, 0.5):
            return server
        if process.poll() is not None:
            server.stop()
            raise NoLocalModel(f"`{' '.join(argv)}` exited immediately.\n\n{log_tail(log)}")
        time.sleep(0.25)
    server.stop()
    raise NoLocalModel(f"{label} did not come up within {timeout:.0f}s.\n\n{log_tail(log)}")
