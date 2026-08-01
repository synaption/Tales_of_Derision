"""Two-way dialogue with a small local LLM -- one that fits on a Raspberry Pi.

Nothing here needs a GPU or a cloud key.  The game talks HTTP to whatever local
server you already run (Ollama, llama.cpp's ``llama-server``, KoboldCpp,
oobabooga, LM Studio -- the same backends SillyTavern points at), and falls back
to hand-written lines when no server answers, so the prototype always runs.

Sizing, because a Pi is the target:

* Pi 4 (4 GB):  a 0.5 B or 360 M instruct model at Q4_K_M -- ``qwen2.5:0.5b``,
  ``smollm2:360m``.  Roughly 300-500 MB resident, a couple of seconds a reply.
* Pi 5 (8 GB):  a 1 B model is comfortable -- ``llama3.2:1b``, ``qwen2.5:1.5b``.

Replies are capped at a sentence or two (``MAX_TOKENS``) because small models
ramble, and every request runs on a worker thread so the 60 fps loop never
stalls waiting on a token.

Environment overrides
    FRUITBRAINS_LLM        ollama | openai | canned   (default: auto-detect)
    FRUITBRAINS_LLM_URL    e.g. http://127.0.0.1:8080
    FRUITBRAINS_MODEL      e.g. qwen2.5:0.5b
"""

from __future__ import annotations

import json
import os
import queue
import random
import re
import threading
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field

from fruit import EMOTIONS, LINES


MAX_TOKENS = 48          # a sentence or two; keeps a Pi responsive
TEMPERATURE = 0.85
HISTORY_TURNS = 6        # messages kept per villager -- tiny models have tiny context
REPLY_CHARS = 180
PROBE_TIMEOUT = 0.4      # how long auto-detect waits for a local server
REQUEST_TIMEOUT = 45.0   # a 0.5B model on a Pi 4 is slow, not broken

OLLAMA_URL = "http://127.0.0.1:11434"
OPENAI_URLS = ("http://127.0.0.1:8080", "http://127.0.0.1:5001", "http://127.0.0.1:5000")
DEFAULT_MODEL = "qwen2.5:0.5b"

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


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _reachable(url: str, timeout: float = PROBE_TIMEOUT) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


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
    """Turns a message list into one line of villager dialogue."""

    name = "backend"
    online = False

    def reply(self, messages: list[dict], mood: str) -> str:
        raise NotImplementedError


class CannedBackend(Backend):
    """No server? Villagers still answer, from the hand-written lines."""

    name = "offline lines"

    def reply(self, messages: list[dict], mood: str) -> str:
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

    def __init__(self, url: str = OLLAMA_URL, model: str = DEFAULT_MODEL) -> None:
        self.url, self.model = url.rstrip("/"), model
        self.name = f"ollama {model}"

    def reply(self, messages: list[dict], mood: str) -> str:
        data = _post_json(f"{self.url}/api/chat", {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS, "num_ctx": 1024},
        }, REQUEST_TIMEOUT)
        return data.get("message", {}).get("content", "")


class OpenAIChatBackend(Backend):
    """The OpenAI-compatible route: llama-server, KoboldCpp, oobabooga, LM Studio."""

    online = True

    def __init__(self, url: str, model: str = "local-model") -> None:
        self.url, self.model = url.rstrip("/"), model
        self.name = f"openai-api {model}"

    def reply(self, messages: list[dict], mood: str) -> str:
        data = _post_json(f"{self.url}/v1/chat/completions", {
            "model": self.model,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "stop": ["\n", "Player:"],
        }, REQUEST_TIMEOUT)
        return data["choices"][0]["message"]["content"]


def detect_backend() -> Backend:
    """Env override, then a quick probe of the usual local ports, then canned."""
    choice = os.environ.get("FRUITBRAINS_LLM", "auto").lower()
    url = os.environ.get("FRUITBRAINS_LLM_URL", "")
    model = os.environ.get("FRUITBRAINS_MODEL", DEFAULT_MODEL)

    if choice == "canned":
        return CannedBackend()
    if choice == "ollama":
        return OllamaBackend(url or OLLAMA_URL, model)
    if choice in ("openai", "llamacpp", "kobold"):
        return OpenAIChatBackend(url or OPENAI_URLS[0], model)

    if url:
        if _reachable(f"{url.rstrip('/')}/api/tags"):
            return OllamaBackend(url, model)
        if _reachable(f"{url.rstrip('/')}/v1/models"):
            return OpenAIChatBackend(url, model)
    if _reachable(f"{OLLAMA_URL}/api/tags"):
        return OllamaBackend(OLLAMA_URL, model)
    for candidate in OPENAI_URLS:
        if _reachable(f"{candidate}/v1/models"):
            return OpenAIChatBackend(candidate, model)
    return CannedBackend()


# --- conversation -----------------------------------------------------------

@dataclass
class Conversation:
    """One villager's running chat: history in, one pending reply out."""

    villager: str
    persona: str
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_TURNS))

    def system_prompt(self, mood: str, player: str) -> str:
        feeling = EMOTIONS.get(mood, EMOTIONS["happy"]).label
        return (f"You are {self.villager}, a talking fruit living in Fruit Town, "
                f"a cosy village. You are {self.persona}. Right now you feel {feeling}. "
                f"You are chatting with your neighbour {player}. "
                "Reply in ONE short friendly sentence, at most 20 words. "
                "Stay in character. Do not narrate actions or use emoji.")

    def messages(self, mood: str, player: str, said: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_prompt(mood, player)}]
        msgs += list(self.history)
        msgs.append({"role": "user", "content": said})
        return msgs


class ChatService:
    """Runs generation off the main thread so the game never stutters.

    The loop calls :meth:`poll` every frame; replies arrive whenever they arrive.
    """

    def __init__(self, backend: Backend | None = None) -> None:
        self.backend = backend if backend is not None else detect_backend()
        self.conversations: dict[str, Conversation] = {}
        self._results: queue.Queue = queue.Queue()
        self._pending: set[str] = set()

    @property
    def name(self) -> str:
        return self.backend.name

    @property
    def online(self) -> bool:
        return self.backend.online

    def conversation(self, villager: str) -> Conversation:
        chat = self.conversations.get(villager)
        if chat is None:
            chat = Conversation(villager, PERSONAS.get(villager, DEFAULT_PERSONA))
            self.conversations[villager] = chat
        return chat

    def busy(self, villager: str) -> bool:
        return villager in self._pending

    def ask(self, villager: str, mood: str, player: str, said: str) -> bool:
        """Queue a reply. Returns False if that villager is already thinking."""
        if villager in self._pending:
            return False
        chat = self.conversation(villager)
        messages = chat.messages(mood, player, said)
        self._pending.add(villager)

        def work() -> None:
            try:
                text = tidy(self.backend.reply(messages, mood))
            except Exception as exc:                     # a dead server must not kill the game
                text = tidy(CannedBackend().reply(messages, mood))
                self._results.put((villager, said, text, f"{type(exc).__name__}: {exc}"))
                return
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
            chat = self.conversation(villager)
            chat.history.append({"role": "user", "content": said})
            chat.history.append({"role": "assistant", "content": text})
            out.append((villager, text, error))
