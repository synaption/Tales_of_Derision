"""Two-way dialogue with a small local LLM -- one that fits on a Raspberry Pi.

Nothing here needs a GPU or a cloud key: the game talks HTTP to whatever local
server you already run (Ollama, llama.cpp's ``llama-server``, KoboldCpp,
oobabooga, LM Studio).

**The model is a hard requirement.**  Startup checks that a server is up *and*
that the model is actually loaded, and refuses to launch otherwise -- a silent
fallback to hand-written lines just looks like a model that ignores you.  If a
request fails mid-session the villager says nothing and the error is shown; no
canned line is ever passed off as a reply.

:class:`CannedBackend` is still here, but as a *deliberate* mode for scripted
content -- barks, tutorials, shopkeeper patter, anything you want deterministic
-- selected with ``FRUITBRAINS_LLM=canned`` or ``--canned``.

Backends come in two shapes, and the difference matters upstream:

* *stateless* (Ollama, OpenAI-compatible) -- the game owns the conversation and
  sends a system card plus the last few turns every time.  Cheap, and the
  villager forgets you when you quit.
* *stateful* (:mod:`letta_backend`) -- the villager is a long-lived agent with
  its own memory; the game sends only the new line and must **not** replay the
  history, or every memory lands twice.

:meth:`ChatService.ask` picks the right one off ``Backend.stateful``.

Sizing, because a Pi is the target:

* Pi 4 (4 GB):  a 0.5 B or 360 M instruct model at Q4_K_M -- ``qwen2.5:0.5b``,
  ``smollm2:360m``.  Roughly 300-500 MB resident, a couple of seconds a reply.
* Pi 5 (8 GB):  a 1 B model is comfortable -- ``llama3.2:1b``, ``qwen2.5:1.5b``.

Replies are capped at a sentence or two (``MAX_TOKENS``) because small models
ramble, and every request runs on a worker thread so the 60 fps loop never
stalls waiting on a token.

Environment overrides
    FRUITBRAINS_LLM        ollama | openai | letta | canned  (default: auto-detect)
    FRUITBRAINS_LLM_URL    e.g. http://127.0.0.1:8080
    FRUITBRAINS_LETTA_URL  e.g. http://127.0.0.1:8283
    FRUITBRAINS_MODEL      e.g. qwen2.5:0.5b
    FRUITBRAINS_DEBUG=1    print every prompt and reply to the console
"""

from __future__ import annotations

import json
import os
import queue
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import hardware
import letta_backend
from fruit import EMOTIONS, LINES
from letta_backend import LETTA_URL, LettaBackend, start_letta
from serve import (BOOT_TIMEOUT, CHECK_TIMEOUT, FIRST_TOKEN_TIMEOUT, PROBE_TIMEOUT,
                   STALL_TIMEOUT, ManagedServer, NoLocalModel, launch, log_tail)
from serve import get_json as _get_json
from serve import post_json as _post_json
from serve import post_stream as _post_stream
from serve import reachable as _reachable


MAX_TOKENS = 72          # enough to finish a sentence; still quick on a Pi
TEMPERATURE = 0.6        # small models wander off-topic when this runs hot
DEBUG = os.environ.get("FRUITBRAINS_DEBUG", "") not in ("", "0")
HISTORY_TURNS = 6        # messages kept per villager -- tiny models have tiny context
REPLY_CHARS = 180
KEEP_ALIVE = "30m"       # keep the weights resident between conversations
SERVER_LOG = Path(tempfile.gettempdir()) / "fruitbrains-ollama.log"
EMBEDDING_MODEL = letta_backend.EMBEDDING_MODEL   # Letta needs one for archival memory

OLLAMA_URL = "http://127.0.0.1:11434"
OPENAI_URLS = ("http://127.0.0.1:8080", "http://127.0.0.1:5001", "http://127.0.0.1:5000")
FALLBACK_MODEL = "qwen2.5:0.5b"       # if hardware detection comes up empty


def chosen_model() -> tuple[str, str]:
    """(model tag, why). Sized to the machine unless FRUITBRAINS_MODEL says otherwise."""
    override = os.environ.get("FRUITBRAINS_MODEL", "")
    if override:
        return override, "FRUITBRAINS_MODEL"
    try:
        machine = hardware.probe()
        return hardware.recommend_model(machine) or FALLBACK_MODEL, machine.label
    except Exception:                  # detection must never stop the game
        return FALLBACK_MODEL, "unknown hardware"

# Who each villager is.  Short, because every word costs tokens on a Pi.
PERSONAS: dict[str, str] = {
    "Apple": "cheerful, practical, always tidying the plaza",
    "Banana": "goofy, tells terrible jokes, slips into puns",
    "Cherry": "a pair of twins who say everything twice",
    "Grape": "gossipy, knows everyone's business, whispers a lot",
    "Kiwi": "shy and fuzzy, speaks softly, apologises too much",
    "Lemon": "sour and sarcastic, secretly fond of everyone",
    "Orange": "sporty and loud, always mid-workout",
    "Peach": "sweet and anxious, worries about the weather",
    "Pear": "posh, thinks the town needs more culture",
    "Pineapple": "the town party host, invites you to everything",
    "Strawberry": "a hopeless romantic, leaves flowers on doorsteps",
    "Watermelon": "laid back, mostly here for the sunshine and naps",
}
DEFAULT_PERSONA = "a friendly fruit who likes the quiet life"


def tidy(text: str) -> str:
    """Small models pad, narrate and over-run. Take the first sentence or two."""
    text = text.strip().strip('"').replace("\n", " ")
    text = re.sub(r"\*[^*]*\*", "", text)            # strip *action* narration
    text = re.sub(r"^\s*\w+\s*:\s*", "", text)       # strip a "Name:" prefix
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > REPLY_CHARS:
        cut = text[:REPLY_CHARS]
        stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = cut[:stop + 1] if stop > 40 else cut.rsplit(" ", 1)[0] + "..."
    return text or "..."


# --- backends ---------------------------------------------------------------

class Backend:
    """Turns a message list into one line of villager dialogue.

    ``on_token`` receives the reply-so-far as it streams, so the game can show
    words appearing instead of an indefinite "thinking" state.
    """

    name = "backend"
    online = False
    scripted = False
    stateful = False        # True when the backend keeps its own per-villager memory

    def reply(self, messages: list[dict], mood: str, on_token=None) -> str:
        raise NotImplementedError

    def converse(self, villager: str, persona: str, mood: str, player: str,
                 said: str, on_token=None) -> str:
        """Stateful route: one line in, one line out, memory kept server-side.

        Stateless backends never see this -- :meth:`ChatService.ask` sends them
        the assembled message list instead.
        """
        raise NotImplementedError

    def check(self) -> None:
        """Raise :class:`NoLocalModel` unless this backend is ready to serve."""

    def warm(self) -> None:
        """Nudge the weights into memory. Failure here is never fatal."""
        try:
            self.reply([{"role": "user", "content": "hi"}], "happy")
        except Exception:
            pass                              # a cold start failing costs a slow first chat

    def memory(self, villager: str, label: str = "human") -> str:
        """What the backend remembers about the player. Stateless: nothing."""
        return ""


class CannedBackend(Backend):
    """Scripted lines -- a deliberate mode, never a stand-in for the model.

    Useful for gameplay that wants determinism: barks, tutorial beats, shop
    patter, tests.  Selected explicitly with ``--canned``; it is never chosen
    for you, because a villager answering from a list looks exactly like a
    model that is ignoring what you typed.
    """

    name = "scripted lines"
    scripted = True

    def reply(self, messages: list[dict], mood: str, on_token=None) -> str:
        asked = messages[-1]["content"].lower() if messages else ""
        pool = list(LINES.get(mood, LINES["happy"]))
        if "?" in asked:
            pool += ["Hmm! I've wondered about that myself.",
                     "Ask me again once I've had a nap.",
                     "You know, I really couldn't say."]
        return random.choice(pool)


class OllamaBackend(Backend):
    """Ollama's native chat endpoint -- the easiest thing to run on a Pi."""

    online = True

    def __init__(self, url: str = OLLAMA_URL, model: str = FALLBACK_MODEL) -> None:
        self.url, self.model = url.rstrip("/"), model
        self.name = f"ollama {model}"

    def installed(self) -> list[str] | None:
        tags = _get_json(f"{self.url}/api/tags", CHECK_TIMEOUT)
        return None if tags is None else [m.get("name", "") for m in tags.get("models", [])]

    def has_model(self) -> bool:
        """Exact tag, not just the family.

        ``qwen2.5:0.5b`` being installed does not mean ``qwen2.5:7b`` is -- a
        looser match here means every request 404s at play time instead.
        """
        have = self.installed() or []
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return any(name == wanted or name == self.model for name in have)

    def check(self, pull: bool = True) -> None:
        have = self.installed()
        if have is None:
            raise NoLocalModel(_no_server_help(f"no Ollama server answering at {self.url}"))
        if self.has_model():
            return
        if not pull:
            raise NoLocalModel(
                f"Ollama at {self.url} does not have '{self.model}'.\n"
                f"  --no-pull was given, so nothing was downloaded.\n"
                f"  Pull it:      ollama pull {self.model}\n"
                f"  Or pick one:  FRUITBRAINS_MODEL=<name> "
                f"(installed: {', '.join(have) or 'none'})")
        pull_model(self.url, self.model)
        if not self.has_model():
            raise NoLocalModel(f"'{self.model}' still isn't available at {self.url} after pulling.")

    def reply(self, messages: list[dict], mood: str, on_token=None) -> str:
        """Stream the reply, feeding partial text to ``on_token`` as it lands."""
        parts: list[str] = []
        for line in _post_stream(f"{self.url}/api/chat", {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS, "num_ctx": 1024},
        }):
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            parts.append(chunk.get("message", {}).get("content", ""))
            if on_token is not None and parts[-1]:
                on_token("".join(parts))
            if chunk.get("done"):
                break
        return "".join(parts)


class OpenAIChatBackend(Backend):
    """The OpenAI-compatible route: llama-server, KoboldCpp, oobabooga, LM Studio."""

    online = True

    def __init__(self, url: str, model: str = "local-model") -> None:
        self.url, self.model = url.rstrip("/"), model
        self.name = f"openai-api {model}"

    def check(self) -> None:
        models = _get_json(f"{self.url}/v1/models", CHECK_TIMEOUT)
        if models is None:
            raise NoLocalModel(_no_server_help(f"nothing answering /v1/models at {self.url}"))
        if not models.get("data"):
            raise NoLocalModel(f"{self.url} is up but has no model loaded.")

    def reply(self, messages: list[dict], mood: str, on_token=None) -> str:
        parts: list[str] = []
        for line in _post_stream(f"{self.url}/v1/chat/completions", {
            "model": self.model,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "stop": ["\n", "Player:"],
            "stream": True,
        }):
            if not line.startswith(b"data:"):
                continue
            body = line[5:].strip()
            if body == b"[DONE]":
                break
            delta = json.loads(body)["choices"][0].get("delta", {}).get("content", "")
            if delta:
                parts.append(delta)
                if on_token is not None:
                    on_token("".join(parts))
        return "".join(parts)


def pull_model(url: str, model: str) -> None:
    """Download a model through the server's own API, with progress on stdout.

    Uses HTTP rather than the `ollama` CLI so this also works against a server
    we didn't start -- including one on another machine.
    """
    print(f"Fruit Brains: pulling {model} (one-off download)...", flush=True)
    last = 0.0
    try:
        for line in _post_stream(f"{url}/api/pull", {"model": model, "stream": True}):
            note = json.loads(line)
            if note.get("error"):
                raise NoLocalModel(f"Pulling '{model}' failed: {note['error']}\n"
                                   f"  Pick another: FRUITBRAINS_MODEL=qwen2.5:0.5b")
            done, total = note.get("completed", 0), note.get("total", 0)
            if total and time.monotonic() - last > 2.0:
                last = time.monotonic()
                print(f"  {note.get('status', 'downloading')}: "
                      f"{done / 1e9:.1f}/{total / 1e9:.1f} GB", flush=True)
    except NoLocalModel:
        raise
    except Exception as exc:
        raise NoLocalModel(f"Pulling '{model}' failed: {type(exc).__name__}: {exc}")
    print(f"Fruit Brains: {model} ready", flush=True)


def _no_server_help(why: str) -> str:
    return (f"Fruit Brains needs a local LLM and {why}.\n\n"
            f"  Start one:    ollama serve\n"
            f"  Get a model:  ollama pull {chosen_model()[0]}\n\n"
            "  Elsewhere:    FRUITBRAINS_LLM_URL=http://pi.local:11434\n"
            "  llama.cpp:    llama-server -m model.gguf --port 8080\n"
            "  No model:     run with --canned for scripted lines (dialogue will not\n"
            "                respond to what you type -- it is for scripted beats)")


def _log_tail(lines: int = 12) -> str:
    return log_tail(SERVER_LOG, lines)


def start_ollama(url: str, model: str) -> ManagedServer:
    """Launch ``ollama serve`` and wait for it to answer. The model check follows."""
    exe = shutil.which("ollama")
    if not exe:
        raise NoLocalModel(
            "Fruit Brains starts its own model server, but `ollama` is not installed.\n\n"
            "  Install:      curl -fsSL https://ollama.com/install.sh | sh\n"
            "  Or point at an existing server:  FRUITBRAINS_LLM_URL=http://host:11434\n"
            "  Or run without a model:          --canned  (scripted lines)")

    host = url.split("://", 1)[-1].rstrip("/")
    env = dict(os.environ, OLLAMA_HOST=host)
    return launch("ollama", [exe, "serve"], env, SERVER_LOG, f"{url}/api/tags", BOOT_TIMEOUT)


def warm_up(backend: Backend) -> None:
    """Nudge the model into memory in the background, so the first chat is quick."""
    threading.Thread(target=backend.warm, name="chat-warmup", daemon=True).start()


def _open_ollama(url: str, model: str, require: bool, autostart: bool,
                 pull: bool) -> tuple[OllamaBackend, ManagedServer | None]:
    """An Ollama with the model in it, started by us if nobody else has one."""
    base = (url or OLLAMA_URL).rstrip("/")
    if _reachable(f"{base}/api/tags"):
        backend = OllamaBackend(base, model)
        if require:
            backend.check(pull)        # a server we didn't start still needs the model
        return backend, None           # ...but its lifetime is not ours to end
    if not autostart:
        raise NoLocalModel(_no_server_help(f"nothing is listening on {base}"))
    server = start_ollama(base, model)
    backend = OllamaBackend(base, model)
    try:
        if require:
            backend.check(pull)
    except BaseException:
        server.stop()                  # never leave our own server orphaned behind an error
        raise
    return backend, server


def _open_letta(url: str, model: str, require: bool, autostart: bool,
                pull: bool) -> tuple[LettaBackend, list[ManagedServer]]:
    """Letta on top of Ollama: agents here, inference there.

    Both layers have to be up, so both get the same treatment -- attach to
    whatever is already running, otherwise start it and own it.  Letta also
    wants an embedding model for archival memory, which is small and pulled
    alongside the chat model.
    """
    servers: list[ManagedServer] = []
    letta_url = (os.environ.get("FRUITBRAINS_LETTA_URL", "") or LETTA_URL).rstrip("/")
    backend = LettaBackend(letta_url, model)
    try:
        if not backend.alive():
            if not autostart:
                raise NoLocalModel(letta_backend.no_letta_help(
                    f"nothing is listening on {letta_url}"))
            # Nothing is serving agents, so we are building the whole stack:
            # inference first, because Letta is useless without it.
            ollama, server = _open_ollama(url, model, require, autostart, pull)
            if server is not None:
                servers.append(server)
            if pull and not OllamaBackend(ollama.url, EMBEDDING_MODEL).has_model():
                pull_model(ollama.url, EMBEDDING_MODEL)   # small, and Letta needs one
            servers.append(start_letta(letta_url, ollama.url))
        if require:
            backend.check(pull)    # a Letta someone else started still has to know our model
    except BaseException:
        for server in reversed(servers):
            server.stop()
        raise
    return backend, servers


def open_backend(require: bool = True, autostart: bool = True,
                 pull: bool = True) -> tuple[Backend, list[ManagedServer]]:
    """Pick a backend, starting our own servers if nothing is listening.

    Returns the backend and -- only for the servers we launched -- the ones to
    shut down on the way out, innermost last.  An already-running server is used
    as-is and left alone.
    """
    choice = os.environ.get("FRUITBRAINS_LLM", "auto").lower()
    url = os.environ.get("FRUITBRAINS_LLM_URL", "")
    model, why = chosen_model()
    print(f"Fruit Brains: {why} -> model {model}", flush=True)

    if choice == "canned":
        return CannedBackend(), []
    if choice in ("openai", "llamacpp", "kobold"):
        backend = OpenAIChatBackend(url or OPENAI_URLS[0], model)
        if require:
            backend.check()
        warm_up(backend)
        return backend, []
    if choice == "letta":
        backend, servers = _open_letta(url, model, require, autostart, pull)
        return backend, servers

    # Auto: a Letta already serving is the richer town -- villagers there
    # remember you -- so it wins over a bare model server.
    if choice == "auto" and _reachable(f"{LETTA_URL}{letta_backend.HEALTH_PATH}"):
        backend = LettaBackend(LETTA_URL, model)
        if require:
            backend.check(pull)
        return backend, []

    found = _probe(url, model) if choice == "auto" else None
    if found is not None:
        if require:
            found.check(pull)
        warm_up(found)
        return found, []

    if autostart:
        backend, server = _open_ollama(url, model, require, True, pull)
        warm_up(backend)
        return backend, [server] if server else []
    if require:
        where = url or f"{OLLAMA_URL} or {', '.join(OPENAI_URLS)}"
        raise NoLocalModel(_no_server_help(f"nothing is listening on {where}"))
    return CannedBackend(), []


def detect_backend(require: bool = True, autostart: bool = False) -> Backend:
    """Backend only, for callers that don't want to own a server's lifetime."""
    return open_backend(require, autostart)[0]


def _probe(url: str, model: str) -> Backend | None:
    """Look for a server, nearest first. Returns None if the town is quiet."""
    candidates = []
    if url:
        candidates += [(OllamaBackend, url, "/api/tags"), (OpenAIChatBackend, url, "/v1/models")]
    candidates.append((OllamaBackend, OLLAMA_URL, "/api/tags"))
    candidates += [(OpenAIChatBackend, u, "/v1/models") for u in OPENAI_URLS]
    for factory, base, path in candidates:
        if _reachable(f"{base.rstrip('/')}{path}"):
            return factory(base, model)
    return None


# --- conversation -----------------------------------------------------------

@dataclass
class Conversation:
    """One villager's running chat: history in, one pending reply out."""

    villager: str
    persona: str
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_TURNS))

    def system_prompt(self, mood: str, player: str) -> str:
        feeling = EMOTIONS.get(mood, EMOTIONS["happy"]).label
        return (f"You are {self.villager}, a talking fruit in the village of Fruit Town. "
                f"Personality: {self.persona}. Mood: {feeling}. "
                f"You are chatting with your neighbour {player}.\n"
                "Rules: answer what they actually said, directly. One short sentence, "
                "20 words max, in English. Stay in character. No narration, no "
                "asterisks, no emoji.")

    def messages(self, mood: str, player: str, said: str) -> list[dict]:
        """System card, recent turns, then their line -- restated so a 0.5 B
        model keeps its attention on the question instead of its own persona."""
        msgs = [{"role": "system", "content": self.system_prompt(mood, player)}]
        msgs += list(self.history)
        msgs.append({"role": "user", "content":
                     f'{player} says: "{said}"\nReply as {self.villager}, answering them.'})
        return msgs


class ChatService:
    """Runs generation off the main thread so the game never stutters.

    The loop calls :meth:`poll` every frame; replies arrive whenever they arrive.
    """

    def __init__(self, backend: Backend | None = None, require: bool = True,
                 autostart: bool = True, pull: bool = True) -> None:
        if backend is not None:
            self.backend, self.servers = backend, []
        else:
            self.backend, self.servers = open_backend(require, autostart, pull)
        self.conversations: dict[str, Conversation] = {}
        self._results: queue.Queue = queue.Queue()
        self._pending: set[str] = set()
        self._partial: dict[str, str] = {}      # reply-so-far, updated as it streams
        self._started: dict[str, float] = {}
        self._cancelled: set[str] = set()
        self._memories: dict[str, str] = {}     # what a stateful villager recalls of you
        self._recalling: set[str] = set()

    @property
    def name(self) -> str:
        return self.backend.name

    @property
    def online(self) -> bool:
        return self.backend.online

    @property
    def scripted(self) -> bool:
        """True when dialogue is coming from the scripted-lines mode."""
        return self.backend.scripted

    @property
    def stateful(self) -> bool:
        """True when villagers remember you between sessions (the Letta backend)."""
        return self.backend.stateful

    @property
    def server(self) -> ManagedServer | None:
        """The outermost server we started, if any. ``None`` means we started none."""
        return self.servers[0] if self.servers else None

    def shutdown(self) -> None:
        """Stop the servers this session started -- and only those, innermost first."""
        for server in reversed(self.servers):
            print(f"Fruit Brains: stopping {server.label}", flush=True)
            server.stop()
        self.servers = []

    def conversation(self, villager: str) -> Conversation:
        chat = self.conversations.get(villager)
        if chat is None:
            chat = Conversation(villager, PERSONAS.get(villager, DEFAULT_PERSONA))
            self.conversations[villager] = chat
        return chat

    def busy(self, villager: str) -> bool:
        return villager in self._pending

    def partial(self, villager: str) -> str:
        """The reply as far as it has streamed in -- "" before the first token."""
        return self._partial.get(villager, "")

    def waited(self, villager: str) -> float:
        """Seconds this villager has been thinking, for the UI to show."""
        start = self._started.get(villager)
        return 0.0 if start is None else time.monotonic() - start

    def cancel(self, villager: str) -> None:
        """Give up on a reply. The worker notices and stops feeding tokens."""
        if villager in self._pending:
            self._cancelled.add(villager)
            self._pending.discard(villager)
            self._partial.pop(villager, None)
            self._started.pop(villager, None)

    def memory(self, villager: str) -> str:
        """What this villager remembers about the player -- "" until it's fetched."""
        return self._memories.get(villager, "")

    def recall(self, villager: str) -> None:
        """Ask the backend what it remembers, off the main thread.

        Only stateful backends have anything to say; for the rest this is a
        no-op, so the caller can fire it whenever a conversation opens.
        """
        if not self.backend.stateful or villager in self._recalling:
            return
        self._recalling.add(villager)

        def work() -> None:
            try:
                text = self.backend.memory(villager)
            except Exception:
                text = ""                 # a missing memory line is not worth an error
            if text:
                self._memories[villager] = text
            self._recalling.discard(villager)

        threading.Thread(target=work, name=f"recall-{villager}", daemon=True).start()

    def ask(self, villager: str, mood: str, player: str, said: str) -> bool:
        """Queue a reply. Returns False if that villager is already thinking."""
        if villager in self._pending:
            return False
        chat = self.conversation(villager)
        # A stateful backend is sent the line alone -- assembling a prompt for it
        # would only be misleading in the debug log.
        messages = [] if self.backend.stateful else chat.messages(mood, player, said)
        self._pending.add(villager)
        self._partial[villager] = ""
        self._started[villager] = time.monotonic()
        self._cancelled.discard(villager)

        class Cancelled(Exception):
            pass

        def on_token(sofar: str) -> None:
            if villager in self._cancelled:
                raise Cancelled()
            self._partial[villager] = sofar

        def work() -> None:
            if DEBUG:
                print(f"\n[chat] -> {villager} ({mood})\n"
                      + "\n".join(f"  {m['role']}: {m['content']}" for m in messages)
                      + (f"  (agent) {player}: {said}" if not messages else ""))
            try:
                if self.backend.stateful:
                    # The villager's agent already holds the history; sending it
                    # again would file every memory twice.
                    raw = self.backend.converse(villager, chat.persona, mood,
                                                player, said, on_token)
                else:
                    raw = self.backend.reply(messages, mood, on_token)
                text = tidy(raw)
            except Cancelled:
                return                       # the player walked off; nothing to report
            except Exception as exc:
                # No scripted stand-in: a failed request must look like a failure,
                # not like a villager who ignored you.
                self._results.put((villager, said, "", f"{type(exc).__name__}: {exc}"))
                return
            if DEBUG:
                print(f"[chat] <- {villager}: {text}")
            self._results.put((villager, said, text, None))

        threading.Thread(target=work, name=f"chat-{villager}", daemon=True).start()
        return True

    def poll(self) -> list[tuple[str, str, str | None]]:
        """Collect finished replies: (villager, reply, error or None)."""
        out = []
        while True:
            try:
                villager, said, text, error = self._results.get_nowait()
            except queue.Empty:
                return out
            self._pending.discard(villager)
            self._partial.pop(villager, None)
            self._started.pop(villager, None)
            if villager in self._cancelled:
                self._cancelled.discard(villager)
                continue                   # a reply we already stopped caring about
            if text:                       # only real replies become memory
                chat = self.conversation(villager)
                chat.history.append({"role": "user", "content": said})
                chat.history.append({"role": "assistant", "content": text})
                # A stateful villager may have just written something down about
                # you; ask again so the panel shows what changed.
                self.recall(villager)
            out.append((villager, text, error))
