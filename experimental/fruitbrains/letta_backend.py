"""Persistent villager personalities, backed by a local Letta server.

The other backends in :mod:`chat` are stateless: the game hands over a system
card plus the last few turns, and the model answers.  Close the game and the
villager forgets you.  Letta inverts that -- each villager is a long-lived
*agent* with its own memory blocks, living in Letta's database, and the game
sends nothing but the new line.  What a villager knows about you accumulates
across sessions, and the villager edits that memory themselves (Letta's base
tools include core-memory writes), which is the whole point: Peach should still
be worried about your roof next week.

    one villager  ==  one Letta agent, named `fruitbrains-Peach`, tagged
                      `fruitbrains`, with a `persona` block (who they are) and
                      a `human` block (what they've learnt about you).

Letta is an agent layer, not a model server -- it still needs something to do
the inference.  We point it at the same local Ollama the other backends use, so
the model choice from :mod:`hardware` carries over unchanged, plus a small
embedding model for archival memory.

Running it -- docker is the path that works out of the box, because Letta's
server stores everything in Postgres (``letta/server/db.py`` builds a Postgres
engine unconditionally; the sqlite setting only covers its desktop mode) and
the image ships one::

    docker run -d -p 8283:8283 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \\
        -v ~/.letta/.persist/pgdata:/var/lib/postgresql/data letta/letta:latest

``pip install letta`` also works, but only with a Postgres to point at::

    LETTA_PG_URI=postgresql://letta:letta@localhost:5432/letta   # then the game
                                                                # starts it itself

A Letta server that is already listening is used as-is and never shut down;
one we start is stopped on the way out, same rule as Ollama.

Environment
    FRUITBRAINS_LETTA_URL    default http://127.0.0.1:8283
    FRUITBRAINS_EMBEDDING    default nomic-embed-text
    LETTA_API_KEY            sent as a bearer token if the server wants one
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import quote

from serve import (BOOT_TIMEOUT, CHECK_TIMEOUT, ManagedServer, NoLocalModel, get_json,
                   launch, post_stream, reachable, request_json_strict)


LETTA_URL = os.environ.get("FRUITBRAINS_LETTA_URL", "") or "http://127.0.0.1:8283"
EMBEDDING_MODEL = os.environ.get("FRUITBRAINS_EMBEDDING", "") or "nomic-embed-text"
AGENT_PREFIX = "fruitbrains-"     # namespaced: someone else's "Peach" is not ours
AGENT_TAG = "fruitbrains"
HEALTH_PATH = "/v1/health/"
LETTA_LOG = Path(tempfile.gettempdir()) / "fruitbrains-letta.log"
LETTA_BOOT_TIMEOUT = float(os.environ.get("FRUITBRAINS_LETTA_BOOT", 180))
CREATE_TIMEOUT = 90.0             # first agent on a cold database is slow
MEMORY_TIMEOUT = 6.0


def _token() -> str:
    for name in ("FRUITBRAINS_LETTA_TOKEN", "LETTA_API_KEY", "LETTA_SERVER_PASSWORD"):
        value = os.environ.get(name, "")
        if value:
            return value
    return ""


def _handles(entries) -> set[str]:
    """Model handles out of a Letta model listing, across API vintages."""
    found = set()
    for item in entries or []:
        if not isinstance(item, dict):
            continue
        if item.get("handle"):
            found.add(item["handle"])
        kind, model = item.get("model_endpoint_type"), item.get("model")
        if kind and model:
            found.add(f"{kind}/{model}")
    return found


def _content_text(content) -> str:
    """Letta messages carry a string or a list of content parts, by version."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


# Whole-text: a JSON object keyed like a call, pretty-printed across lines or not.
TOOL_JSON = re.compile(r'^\s*\{\s*"(name|type|tool|tool_call|function)"\s*:')
# Per-line: `memory_insert(label="human", ...)` and friends.
TOOL_LINE = re.compile(r'^\s*([a-z_]{3,}\(\s*[a-z_]+\s*=|<(tool_call|function)[ >])',
                       re.IGNORECASE)


def spoken(text: str) -> str:
    """Whatever is left once tool-call syntax is taken out of the dialogue.

    Small models routinely write ``memory_insert(label="human", ...)`` as prose
    when the server does not parse it as a call -- qwen2.5 through Ollama's
    OpenAI-compatible endpoint returns no native tool calls at all.  A villager
    must never say that: strip it, and if nothing human is left, fail the turn
    so the game reports an error instead of speaking JSON at the player.
    """
    kept = ""
    if not TOOL_JSON.match(text):
        lines = [line for line in text.splitlines() if not TOOL_LINE.match(line)]
        kept = " ".join(" ".join(lines).split())
    if not kept and text.strip():
        raise RuntimeError("the model wrote a tool call instead of a reply")
    return kept


def _learnt(value: str) -> str:
    """The block minus the text the game seeded it with.

    Agents usually *append*, so the starting sentences survive alongside a real
    memory; subtracting them leaves either the fact they learnt or nothing at
    all, which is what the panel wants to show.
    """
    text = HUMAN_OPENING.sub(" ", value.replace(HUMAN_BOILERPLATE, " "))
    return " ".join(text.split())


def _raise_if_error(event: dict) -> None:
    """Turn Letta's failure events into an exception the game can show.

    Letta reports a dead model as an ``error_message`` event mid-stream and then
    ends the stream normally.  Skipping those (they carry no assistant text)
    produced an empty reply -- a villager who says nothing looks like a villager
    ignoring you, which is exactly what this game must never do.
    """
    kind = event.get("message_type")
    if kind == "error_message":
        detail = event.get("detail") or event.get("message") or "unknown error"
        raise RuntimeError(str(detail)[:300])
    if event.get("error"):
        raise RuntimeError(str(event["error"])[:300])


def _assistant_text(message: dict) -> str:
    """Only the villager's spoken line -- not its reasoning or tool traffic."""
    if not isinstance(message, dict):
        return ""
    if message.get("message_type") not in ("assistant_message", None):
        return ""
    if message.get("message_type") is None and message.get("role") != "assistant":
        return ""
    return _content_text(message.get("content", ""))


class LettaBackend:
    """One Letta agent per villager, found by name or created on first meeting.

    Implements the :class:`chat.Backend` protocol, but with ``stateful = True``:
    the service must send only the new line, because the history lives on the
    server.  Re-sending it would double every memory.
    """

    online = True
    scripted = False
    stateful = True

    def __init__(self, url: str = LETTA_URL, model: str = "qwen2.5:0.5b",
                 embedding: str = EMBEDDING_MODEL, token: str | None = None) -> None:
        self.url = url.rstrip("/")
        self.model = model if "/" in model else f"ollama/{model}"
        self.embedding = embedding if "/" in embedding else f"ollama/{embedding}"
        self.token = _token() if token is None else token
        self.name = f"letta {self.model.split('/')[-1]}"
        self._agents: dict[str, str] = {}      # villager -> agent id
        self._lock = threading.Lock()          # agent creation must happen once

    # -- server ---------------------------------------------------------------

    def alive(self) -> bool:
        return reachable(f"{self.url}{HEALTH_PATH}", CHECK_TIMEOUT)

    def _resolve(self, handle: str, listing: str, what: str) -> str:
        """Match a handle to one Letta actually offers, tag and all.

        Letta lists Ollama models by their full tag -- ``ollama/qwen2.5:7b``,
        ``ollama/nomic-embed-text:latest`` -- and rejects anything else with a
        400 at agent-creation time.  An untagged name is worth one guess at
        ``:latest`` before giving up.
        """
        listed = _handles(get_json(listing, CHECK_TIMEOUT, self.token))
        if not listed:
            return handle              # can't read the listing; let creation speak
        for candidate in (handle, f"{handle}:latest"):
            if candidate in listed:
                return candidate
        raise NoLocalModel(
            f"Letta at {self.url} does not offer {what} '{handle}'.\n"
            f"  It sees: {', '.join(sorted(listed)[:8])}\n"
            f"  Letta needs OLLAMA_BASE_URL set, to an address it can reach from\n"
            f"  wherever it runs -- in docker that is not 127.0.0.1, and Ollama must\n"
            f"  be listening on it (OLLAMA_HOST=0.0.0.0).\n"
            f"  Or pick one it has:  FRUITBRAINS_MODEL=<name>")

    def check(self, pull: bool = True) -> None:
        """Refuse to start unless Letta is up and can run both of our models."""
        health = get_json(f"{self.url}{HEALTH_PATH}", CHECK_TIMEOUT, self.token)
        if health is None:
            raise NoLocalModel(no_letta_help(f"no Letta server answering at {self.url}"))
        if isinstance(health, dict) and health.get("version"):
            self.name = f"letta {health['version']} {self.model.split('/')[-1]}"
        # Resolved now, at startup, rather than as a 400 the first time someone
        # tries to say hello.
        self.model = self._resolve(self.model, f"{self.url}/v1/models/", "model")
        self.embedding = self._resolve(self.embedding, f"{self.url}/v1/models/embedding",
                                       "embedding model")

    # -- agents ---------------------------------------------------------------

    def agent_name(self, villager: str) -> str:
        return f"{AGENT_PREFIX}{villager}"

    def find_agent(self, villager: str) -> str | None:
        """The agent id Letta already has for this villager, if any."""
        name = self.agent_name(villager)
        found = request_json_strict("GET", f"{self.url}/v1/agents/?name={quote(name)}",
                                    timeout=CHECK_TIMEOUT, token=self.token)
        rows = found.get("agents", []) if isinstance(found, dict) else found
        for row in rows or []:
            if isinstance(row, dict) and row.get("name") == name and row.get("id"):
                return row["id"]
        return None

    def create_agent(self, villager: str, persona: str, player: str) -> str:
        """Make the villager for the first time.

        The persona block is written once and never overwritten: agents are
        allowed to edit their own memory, and a game that rewrote it every
        launch would be a game where nothing persists.
        """
        payload = {
            "name": self.agent_name(villager),
            "description": f"Fruit Brains villager {villager}",
            "tags": [AGENT_TAG],
            "model": self.model,
            "embedding": self.embedding,
            "include_base_tools": True,
            "memory_blocks": [
                {"label": "persona", "value": persona_block(villager, persona)},
                {"label": "human", "value": human_block(player)},
            ],
        }
        made = request_json_strict("POST", f"{self.url}/v1/agents/", payload,
                                   timeout=CREATE_TIMEOUT, token=self.token)
        agent_id = made.get("id") if isinstance(made, dict) else None
        if not agent_id:
            raise RuntimeError(f"Letta created no agent for {villager}: {str(made)[:200]}")
        print(f"Fruit Brains: {villager} is a new Letta agent ({agent_id})", flush=True)
        return agent_id

    def agent_for(self, villager: str, persona: str, player: str) -> str:
        """Find-or-create, once per villager per session, under a lock.

        Two villagers can be mid-reply at once; two *creations* of the same
        villager would leave a duplicate agent in the database forever.
        """
        with self._lock:
            known = self._agents.get(villager)
            if known:
                return known
            agent_id = self.find_agent(villager) or self.create_agent(villager, persona, player)
            self._agents[villager] = agent_id
            return agent_id

    def forget(self, villager: str | None = None) -> list[str]:
        """Delete our agents, so the town can meet you fresh. Returns what went."""
        listed = request_json_strict("GET", f"{self.url}/v1/agents/?tags={AGENT_TAG}&limit=200",
                                     timeout=CHECK_TIMEOUT, token=self.token)
        rows = listed.get("agents", []) if isinstance(listed, dict) else (listed or [])
        wanted = self.agent_name(villager) if villager else None
        gone = []
        for row in rows:
            name = row.get("name", "")
            if not name.startswith(AGENT_PREFIX) or (wanted and name != wanted):
                continue
            request_json_strict("DELETE", f"{self.url}/v1/agents/{row['id']}",
                                timeout=CHECK_TIMEOUT, token=self.token)
            self._agents.pop(name[len(AGENT_PREFIX):], None)
            gone.append(name)
        return gone

    def memory(self, villager: str, label: str = "human") -> str:
        """What this villager has *learnt*, for the game to show.

        A block still carrying its starting text has nothing to say, so it
        reads as empty rather than putting boilerplate on screen.  Best effort
        throughout: an unreadable block is a missing line in the chat panel,
        never an error in the player's face.
        """
        agent_id = self._agents.get(villager)
        if not agent_id:
            # Not spoken to yet *this* session -- but the agent may have been
            # remembering you since last week, which is the point.
            try:
                agent_id = self.find_agent(villager)
            except Exception:
                return ""
            if not agent_id:
                return ""
            with self._lock:
                self._agents.setdefault(villager, agent_id)
        block = get_json(f"{self.url}/v1/agents/{agent_id}/core-memory/blocks/{label}",
                         MEMORY_TIMEOUT, self.token)
        if isinstance(block, dict) and block.get("value"):
            return _learnt(block["value"])
        whole = get_json(f"{self.url}/v1/agents/{agent_id}/core-memory", MEMORY_TIMEOUT, self.token)
        blocks = whole.get("blocks", []) if isinstance(whole, dict) else []
        for entry in blocks:
            if isinstance(entry, dict) and entry.get("label") == label:
                return _learnt(entry.get("value") or "")
        return ""

    def remembers(self, villager: str) -> str:
        """One line for the panel: what they know about you, or how far back.

        A model that writes to its memory blocks gives us facts to show.  One
        that never calls a tool -- qwen2.5, the default -- still has every past
        conversation on the server, so say *that* rather than leaving the line
        reading as if nothing persisted.
        """
        facts = self.memory(villager)
        if facts:
            return facts
        agent_id = self._agents.get(villager)
        if not agent_id:
            return ""
        history = get_json(f"{self.url}/v1/agents/{agent_id}/messages?limit=200",
                           MEMORY_TIMEOUT, self.token)
        turns = sum(1 for m in history or []
                    if isinstance(m, dict) and m.get("message_type") == "user_message")
        if not turns:
            return ""
        return f"{turns} thing{'' if turns == 1 else 's'} you have said before"

    # -- talking --------------------------------------------------------------

    def converse(self, villager: str, persona: str, mood: str, player: str,
                 said: str, on_token=None) -> str:
        """Send one line to the villager's agent and stream the answer back."""
        agent_id = self.agent_for(villager, persona, player)
        text = f'[{villager} is feeling {mood}] {player} says: "{said}"'
        body = {"messages": [{"role": "user", "content": text}]}
        try:
            said_back = self._stream(agent_id, body, on_token)
        except _NoStreaming:
            said_back = self._blocking(agent_id, body, on_token)
        return spoken(said_back)

    def _stream(self, agent_id: str, body: dict, on_token) -> str:
        """SSE, one `data:` line per chunk. Token deltas arrive as assistant_message."""
        parts: list[str] = []
        events = 0
        stopped = ""
        payload = dict(body, stream_tokens=True)
        try:
            lines = post_stream(f"{self.url}/v1/agents/{agent_id}/messages/stream",
                                payload, self.token, accept="text/event-stream")
            for line in lines:
                if not line.startswith(b"data:"):
                    continue
                events += 1
                chunk = line[5:].strip()
                if not chunk or chunk == b"[DONE]":
                    continue
                try:
                    event = json.loads(chunk)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                _raise_if_error(event)
                if event.get("message_type") == "stop_reason":
                    # Arrives *before* the error event that explains it, so keep
                    # it and let the better message win if one follows.
                    stopped = event.get("stop_reason", "")
                delta = _assistant_text(event)
                if delta:
                    parts.append(delta)
                    if on_token is not None:
                        on_token("".join(parts))
        except RuntimeError:
            raise
        except Exception as exc:                    # older servers have no stream route
            if not parts and _looks_unsupported(exc):
                raise _NoStreaming from None
            raise
        if not events:
            # A 200 with nothing SSE-shaped in it means this server answers the
            # stream route in plain JSON.  Ask again, the blocking way -- an
            # empty reply here would be indistinguishable from a mute villager.
            raise _NoStreaming
        if not parts and stopped and stopped != "end_turn":
            raise RuntimeError(f"letta stopped: {stopped}")
        return "".join(parts)

    def _blocking(self, agent_id: str, body: dict, on_token) -> str:
        reply = request_json_strict("POST", f"{self.url}/v1/agents/{agent_id}/messages",
                                    body, timeout=CREATE_TIMEOUT, token=self.token)
        messages = reply.get("messages", []) if isinstance(reply, dict) else (reply or [])
        for message in messages:
            if isinstance(message, dict):
                _raise_if_error(message)
        text = "".join(_assistant_text(m) for m in messages)
        if text and on_token is not None:
            on_token(text)
        return text

    def reply(self, messages: list[dict], mood: str, on_token=None) -> str:
        """Stateless entry point, for callers that don't know about agents.

        Routed to a scratch agent named after nobody, so warm-ups and probes
        can't scribble on a real villager's memory.
        """
        said = messages[-1]["content"] if messages else "hello"
        return self.converse("Sprout", "a quiet seedling minding the greenhouse",
                             mood, "a neighbour", said, on_token)

    def warm(self) -> None:
        """Nothing to warm: creating an agent to say "hi" would persist forever."""


class _NoStreaming(Exception):
    """This server has no streaming route; fall back to one blocking call."""


def _looks_unsupported(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    return code in (404, 405, 501)


def persona_block(villager: str, persona: str) -> str:
    """The agent's own sense of self. Written once; the agent may edit it later.

    "I always speak English" is load-bearing: qwen2.5:7b drifts into Chinese
    without it.  The memory line is deliberately vague about *how* -- Letta's
    own system prompt already explains the memory tools, and spelling out the
    mechanism here got a 7 B model writing `memory_insert(...)` as dialogue.
    """
    return (f"My name is {villager}. I am a talking fruit living in Fruit Town.\n"
            f"I am {persona}.\n"
            "I always speak English, in one short sentence, 20 words at most, "
            "answering what my neighbour actually said. No narration, no asterisks, "
            "no emoji, never any code.\n"
            "I remember what my neighbours tell me about themselves, so I still know "
            "it the next time we meet.")


HUMAN_BOILERPLATE = ("I have only just met them, so I do not know much yet. "
                     "I should append what they tell me about themselves to this block.")
HUMAN_OPENING = re.compile(r"My neighbour \S+ lives in Fruit Town too\.")


def human_block(player: str) -> str:
    return f"My neighbour {player} lives in Fruit Town too.\n{HUMAN_BOILERPLATE}"


def no_letta_help(why: str) -> str:
    return (f"Fruit Brains is set to use Letta for persistent personalities, and {why}.\n\n"
            "  Docker:       docker run -d -p 8283:8283 \\\n"
            "                  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \\\n"
            "                  letta/letta:latest\n"
            "  Or from pip:  pip install letta  -- its server needs a Postgres, so also\n"
            "                LETTA_PG_URI=postgresql://letta:letta@localhost:5432/letta\n"
            "  Elsewhere:    FRUITBRAINS_LETTA_URL=http://pi.local:8283\n\n"
            "  Without Letta the villagers still talk -- they just forget you between\n"
            "  sessions.  Drop --letta to use the plain Ollama backend.")


def start_letta(url: str, ollama_url: str) -> ManagedServer:
    """Launch ``letta server`` pointed at our local Ollama, and wait for health.

    Letta's first boot builds a database and can take a minute or two; the
    timeout is generous for that reason and the log path is printed up front.
    """
    import shutil

    exe = shutil.which("letta")
    if not exe:
        raise NoLocalModel(no_letta_help("`letta` is not installed"))

    host, _, port = url.split("://", 1)[-1].rstrip("/").partition(":")
    env = dict(os.environ, OLLAMA_BASE_URL=ollama_url,
               LETTA_LLM_ENDPOINT_TYPE="ollama")
    argv = [exe, "server", "--host", host or "127.0.0.1", "--port", port or "8283"]
    started = time.monotonic()
    try:
        server = launch("letta", argv, env, LETTA_LOG, f"{url.rstrip('/')}{HEALTH_PATH}",
                        LETTA_BOOT_TIMEOUT, note=" -- first boot builds its database")
    except NoLocalModel as exc:
        # By far the most common cause, and the log says "Connect call failed
        # ('127.0.0.1', 5432)" rather than anything about Postgres being needed.
        raise NoLocalModel(f"{exc}\n\nIf that mentions port 5432: Letta's server keeps its "
                           "state in Postgres.\nEither give it one (LETTA_PG_URI=...) or run "
                           "the docker image, which\nbrings its own:\n\n"
                           "  docker run -d -p 8283:8283 \\\n"
                           "    -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \\\n"
                           "    letta/letta:latest") from None
    print(f"Fruit Brains: letta up in {time.monotonic() - started:.0f}s", flush=True)
    return server
