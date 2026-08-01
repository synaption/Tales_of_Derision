"""Headless smoke test for Fruit Brains -- no window is ever opened.

    python3 experimental/fruitbrains/headlesstest.py

Drives the simulation and the renderer against a dummy SDL video driver,
asserts the things that are easy to break (mood decay, collision, door
round-trips, zoom limits, mixed pixel scales) and drops a contact sheet at
``fruitbrains_headless.png`` so the art can be eyeballed without launching the
game.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("FRUITBRAINS_LLM", "canned")   # never probe the network in tests
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pygame

import chat as chat_mod
import fruitbrains
import hardware
import serve as serve_mod
from chat import CannedBackend, ChatService, _reachable, tidy
from fruit import EMOTIONS, Fruit
from render import MAX_ZOOM, MIN_ZOOM


SHOT = Path(__file__).resolve().parent / "fruitbrains_headless.png"


def step(game: fruitbrains.Game, frames: int, t0: float = 0.0) -> float:
    for i in range(frames):
        game.update(1 / 60, t0 + i / 60)
        game.draw(t0 + i / 60)
    return t0 + frames / 60


def test_moods_decay_to_happy(game: fruitbrains.Game) -> None:
    fruit = game.player
    fruit.feel("angry", 0.5)
    assert fruit.emotion == "angry"
    step(game, 60)
    assert fruit.emotion == "happy", f"mood stuck on {fruit.emotion}"


def test_every_mood_draws(game: fruitbrains.Game) -> None:
    for name in EMOTIONS:
        game.player.feel(name, 5.0)
        step(game, 6, 30)
    game.player.feel("happy", 0.1)


def test_solids_block(game: fruitbrains.Game) -> None:
    solid = game.scenes["town"].solids[0]
    fruit = game.player
    fruit.pos.update(solid.centerx, solid.centery - 200)
    for _ in range(240):                      # push straight into the wall
        fruit.velocity.y += 30
        fruit.update(1 / 60, 0, None, game.scenes["town"])
    assert not fruit.footprint().colliderect(solid), "walked through a solid prop"


def test_door_round_trip(game: fruitbrains.Game) -> None:
    door = game.scenes["town"].doors[0]
    game.player.pos.update(door.rect.centerx, door.rect.bottom + 70)
    t = 0.0
    for _ in range(240):
        game.player.velocity.y -= 26
        t = step(game, 1, t)
        if game.scene.name == "house":
            break
    assert game.scene.name == "house", "never got through the front door"
    assert game.scenes["house"].bounds.collidepoint(game.player.pos), "spawned outside the room"

    for _ in range(300):
        game.player.velocity.y += 26
        t = step(game, 1, t)
        if game.scene.name == "town":
            break
    assert game.scene.name == "town", "could not get back outside"


def test_zoom_is_bounded(game: fruitbrains.Game) -> None:
    for _ in range(40):
        game.cam.zoom_by(2.0)
    assert game.cam.target_zoom <= MAX_ZOOM
    for _ in range(80):
        game.cam.zoom_by(0.5)
    assert game.cam.target_zoom >= MIN_ZOOM
    game.cam.target_zoom = game.cam.zoom = 1.4


def test_mixed_pixel_scales(game: fruitbrains.Game) -> None:
    """Every sprite carries its own scale; 16 px tiles land on a 32 unit grid."""
    tile = game.assets.town["tile_path"]
    assert (tile.w, tile.h) == (32, 32), "tileset is not rendering at 2x"
    fruit_art = next(iter(game.assets.fruits.values()))
    assert fruit_art.w == 64 and fruit_art.art_scale == 2
    assert game.assets.flowers[0].art_scale == 2
    # A hand-made 1x sprite has to survive the same pipeline.
    from render import PixelArt, blit_art
    odd = PixelArt(pygame.Surface((9, 5), pygame.SRCALPHA), 1)
    assert (odd.w, odd.h) == (9, 5)
    blit_art(game.screen, game.cam, odd, game.player.pos.x, game.player.pos.y)


def test_talking(game: fruitbrains.Game) -> None:
    others = [f for f in game.here() if f is not game.player]
    other = min(others, key=lambda f: f.pos.distance_to(game.player.pos))
    other.pos.update(game.player.pos + pygame.Vector2(40, 0))
    game.talk()
    assert other.line, "villager said nothing"
    step(game, 10)


class ScriptedBackend(chat_mod.Backend):
    """A stand-in model: deterministic, optionally slow, never touches a socket."""

    name = "scripted"
    online = True

    def __init__(self, delay: float = 0.0, text: str = "Oh! Lovely to see you, neighbour.") -> None:
        self.delay, self.text = delay, text
        self.seen: list[list[dict]] = []

    def reply(self, messages: list[dict], mood: str, on_token=None) -> str:
        self.seen.append(messages)
        if self.delay:
            time.sleep(self.delay)
        if on_token is not None:
            on_token(self.text)
        return self.text


class BrokenBackend(chat_mod.Backend):
    name = "broken"
    online = True

    def reply(self, messages: list[dict], mood: str, on_token=None) -> str:
        raise ConnectionRefusedError("nothing listening on :11434")


def _ollama_pids(path_hint: str) -> set[int]:
    """PIDs of fake-ollama servers running out of this test's temp dir."""
    tmp = path_hint.split(os.pathsep)[0]
    found = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if tmp in (entry / "cmdline").read_bytes().decode(errors="replace"):
                found.add(int(entry.name))
        except OSError:
            continue
    return found


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def fake_ollama(models: list[str], content: str = "Hello!", token_delay: float = 0.0,
                stall_after: int | None = None):
    """A stand-in Ollama: /api/tags, and /api/chat streaming NDJSON like the real one.

    ``token_delay`` slows each token; ``stall_after`` stops sending mid-reply so
    the stall path can be exercised.
    """
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"models": [{"name": name} for name in models]})

        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            words = content.split(" ")
            try:
                for i, word in enumerate(words):
                    if stall_after is not None and i >= stall_after:
                        time.sleep(30)          # go quiet mid-sentence
                        return
                    chunk = word if i == 0 else " " + word
                    self.wfile.write(json.dumps({"message": {"content": chunk},
                                                 "done": False}).encode() + b"\n")
                    self.wfile.flush()
                    if token_delay:
                        time.sleep(token_delay)
                self.wfile.write(json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass                            # the client gave up; that's allowed

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.seen = seen
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def fake_letta(reply: str = "Good to see you again!", version: str = "0.16.8",
               handles: list[str] | None = None,
               embeddings: list[str] | None = None):
    """A stand-in Letta: agents that persist for the life of the server.

    Mirrors the v1 routes the backend actually calls -- health, model listing,
    find-by-name, create, streamed messages, core-memory blocks, delete -- so
    the agent-per-villager plumbing is exercised without installing Letta.
    """
    agents: dict[str, dict] = {}          # id -> agent record, i.e. Letta's database
    seen: list[dict] = []                 # every message body that arrived

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, payload, code: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _agent_id(self) -> str:
            return self.path.split("/v1/agents/", 1)[1].split("/")[0].split("?")[0]

        def do_GET(self):
            path = self.path
            if path.startswith("/v1/health"):
                return self._send({"version": version, "status": "ok"})
            if path.startswith("/v1/models/embedding"):
                # Real Letta lists Ollama models by full tag, `:latest` included.
                return self._send([{"handle": h} for h in
                                   (embeddings or ["ollama/nomic-embed-text:latest"])])
            if path.startswith("/v1/models"):
                return self._send([{"handle": h} for h in
                                   (handles or ["ollama/qwen2.5:0.5b", "ollama/qwen2.5:7b"])])
            if "/core-memory/blocks/" in path:
                agent = agents.get(self._agent_id(), {})
                label = path.rsplit("/", 1)[1]
                return self._send(agent.get("blocks", {}).get(label, {"label": label, "value": ""}))
            if path.startswith("/v1/agents"):
                query = path.split("?", 1)[1] if "?" in path else ""
                params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
                rows = list(agents.values())
                if "name" in params:
                    wanted = unquote(params["name"])
                    rows = [a for a in rows if a["name"] == wanted]
                if "tags" in params:
                    rows = [a for a in rows if params["tags"] in a.get("tags", [])]
                return self._send(rows)
            self._send({"detail": "not found"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"]) or 0) or b"{}")
            if self.path.rstrip("/") == "/v1/agents":
                agent_id = f"agent-{len(agents) + 1}"
                agents[agent_id] = {
                    "id": agent_id, "name": body["name"], "tags": body.get("tags", []),
                    "blocks": {b["label"]: dict(b) for b in body.get("memory_blocks", [])},
                }
                return self._send(agents[agent_id])
            if "/messages" in self.path:
                agent_id = self._agent_id()
                if agent_id not in agents:
                    return self._send({"detail": "no such agent"}, 404)
                seen.append({"agent": agents[agent_id]["name"], "body": body})
                # The agent learns something, the way a real one edits core memory.
                human = agents[agent_id]["blocks"].setdefault("human", {"label": "human", "value": ""})
                human["value"] = f"{human['value']} They said: {body['messages'][-1]['content']}".strip()
                if not self.path.endswith("/stream"):
                    return self._send({"messages": [
                        {"message_type": "reasoning_message", "content": "thinking"},
                        {"message_type": "assistant_message", "content": reply}]})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"message_type": "reasoning_message", '
                                 b'"content": "hmm"}\n\n')
                if reply == "__error__":
                    # Exactly how real Letta reports a model it cannot reach:
                    # a stop_reason, then an error event, then a clean [DONE].
                    for event in ({"message_type": "stop_reason",
                                   "stop_reason": "llm_api_error"},
                                  {"message_type": "error_message", "error_type": "llm_error",
                                   "detail": "INTERNAL_SERVER_ERROR: Failed to connect to "
                                             "OpenAI: Connection error."}):
                        self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
                for i, word in enumerate(reply.split(" ")):
                    chunk = word if i == 0 else " " + word
                    self.wfile.write(b"data: " + json.dumps(
                        {"message_type": "assistant_message", "content": chunk}).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            self._send({"detail": "not found"}, 404)

        def do_DELETE(self):
            agents.pop(self._agent_id(), None)
            self._send({"deleted": True})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.agents = agents
    server.seen = seen
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


@contextlib.contextmanager
def letta_town(**kwargs):
    """A fake Letta serving on a spare port, with the env pointed at it."""
    saved = dict(os.environ)
    server, port = fake_letta(**kwargs)
    try:
        os.environ["FRUITBRAINS_LLM"] = "letta"
        os.environ["FRUITBRAINS_LETTA_URL"] = f"http://127.0.0.1:{port}"
        os.environ["FRUITBRAINS_MODEL"] = "qwen2.5:0.5b"
        yield server
    finally:
        os.environ.clear()
        os.environ.update(saved)
        server.shutdown()


def _await_reply(service, tries: int = 800):
    for _ in range(tries):
        out = service.poll()
        if out:
            return out
        time.sleep(0.005)
    raise AssertionError("no reply came back")


def test_letta_gives_each_villager_one_agent(game: fruitbrains.Game) -> None:
    """One villager, one agent -- created once, found by name ever after."""
    with letta_town() as letta:
        service = ChatService()
        assert service.stateful and not service.scripted
        assert service.server is None, "adopted a Letta server it did not start"

        service.ask("Peach", "worried", "Apple", "is it going to rain?")
        assert _await_reply(service)[0][1] == "Good to see you again!"
        service.ask("Peach", "happy", "Apple", "how about tomorrow?")
        _await_reply(service)
        service.ask("Lemon", "grumpy", "Apple", "morning")
        _await_reply(service)

        names = sorted(a["name"] for a in letta.agents.values())
        assert names == ["fruitbrains-Lemon", "fruitbrains-Peach"], names

        # A second session is a second ChatService against the same database.
        again = ChatService()
        again.ask("Peach", "happy", "Apple", "back again")
        _await_reply(again)
        assert len(letta.agents) == 2, f"a new session made new agents: {letta.agents}"
        service.shutdown()
        again.shutdown()


def test_letta_is_sent_the_line_not_the_history(game: fruitbrains.Game) -> None:
    """The agent owns the history; replaying it would double every memory."""
    with letta_town() as letta:
        service = ChatService()
        service.ask("Pear", "happy", "Apple", "good morning")
        _await_reply(service)
        service.ask("Pear", "happy", "Apple", "and the concert?")
        _await_reply(service)

        for turn in letta.seen:
            sent = turn["body"]["messages"]
            assert len(sent) == 1, f"sent {len(sent)} messages, not just the new line"
            assert sent[0]["role"] == "user"
        assert "good morning" not in letta.seen[-1]["body"]["messages"][0]["content"], (
            "the first turn was replayed into the second")
        assert "and the concert?" in letta.seen[-1]["body"]["messages"][0]["content"]
        service.shutdown()


def test_letta_memory_reaches_the_panel(game: fruitbrains.Game) -> None:
    """What the villager wrote down about you is shown, not just logged."""
    with letta_town() as letta:
        game.chat = ChatService()
        nearest = min((f for f in game.here() if f is not game.player),
                      key=lambda f: f.pos.distance_to(game.player.pos))
        nearest.pos.update(game.player.pos + pygame.Vector2(50, 0))
        game.talk()
        other = game.talking
        assert other is not None, "nobody was close enough to talk to"

        game.chat.ask(other.name, "happy", game.player.name, "I keep bees")
        _await_reply(game.chat)
        for _ in range(400):                     # the recall runs off the main thread
            if game.chat.memory(other.name):
                break
            time.sleep(0.005)
        assert "I keep bees" in game.chat.memory(other.name), game.chat.memory(other.name)
        step(game, 3)                            # the panel draws the memory line
        game.end_chat()
        game.chat.shutdown()
        assert letta.agents, "the agent vanished"


def test_a_villager_never_speaks_a_tool_call(game: fruitbrains.Game) -> None:
    """Small models write `memory_insert(...)` as dialogue; that is not a line.

    Both shapes seen live from qwen2.5:7b through Letta, which passes them
    through as assistant text because Ollama's OpenAI endpoint returns no
    native tool calls for that model.
    """
    from letta_backend import spoken
    assert spoken("Lovely to see you!") == "Lovely to see you!"
    assert spoken('memory_insert(label="human", new_string="bees")\nHello!') == "Hello!"
    for call in ('{ "name": "memory_replace", "arguments": { "label": "human"',
                 '{\n  "name": "memory_insert",\n  "arguments": {\n    "label": "human"\n  }\n}'):
        try:
            spoken(call)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"a villager would have said this aloud: {call[:40]}")

    # ...and end to end, the game reports it rather than speaking it.
    with letta_town(reply='{"name": "memory_insert", "arguments": {"label": "human"}}'):
        service = ChatService()
        service.ask("Cherry", "happy", "Bob", "hello!")
        villager, text, error = _await_reply(service)[0]
        assert not text and error, (text, error)
        service.shutdown()


def test_letta_errors_are_not_silence(game: fruitbrains.Game) -> None:
    """Letta reports a dead model mid-stream and then ends the stream cleanly."""
    with letta_town(reply="__error__"):
        service = ChatService()
        service.ask("Grape", "happy", "Bob", "hello?")
        villager, text, error = _await_reply(service)[0]
        assert not text, f"spoke anyway: {text}"
        assert error and "Failed to connect" in error, error
        service.shutdown()


def test_forget_deletes_only_our_agents(game: fruitbrains.Game) -> None:
    with letta_town() as letta:
        service = ChatService()
        service.ask("Kiwi", "shy", "Apple", "hello")
        _await_reply(service)
        letta.agents["someone-else"] = {"id": "someone-else", "name": "research-bot", "tags": []}

        gone = service.backend.forget()
        assert gone == ["fruitbrains-Kiwi"], gone
        assert list(letta.agents) == ["someone-else"], "deleted an agent that wasn't ours"
        service.shutdown()


def test_letta_handles_are_resolved_to_full_tags(game: fruitbrains.Game) -> None:
    """Letta lists Ollama models by tag; an untagged handle is a 400 at hello.

    Found against a real Letta 0.16.8, which offers `ollama/nomic-embed-text:latest`
    and rejects `ollama/nomic-embed-text` when the agent is created.
    """
    with letta_town(handles=["ollama/qwen2.5:0.5b:latest"]):
        service = ChatService()
        assert service.backend.embedding == "ollama/nomic-embed-text:latest"
        assert service.backend.model == "ollama/qwen2.5:0.5b:latest", service.backend.model
        service.ask("Apple", "happy", "Bob", "morning!")
        _await_reply(service)
        service.shutdown()

    with letta_town(embeddings=["ollama/some-other-embedder:latest"]):
        try:
            ChatService(autostart=False)
        except chat_mod.NoLocalModel as exc:
            assert "embedding model" in str(exc), exc
        else:
            raise AssertionError("started with an embedding model Letta cannot use")


def test_letta_needs_a_model_it_can_reach(game: fruitbrains.Game) -> None:
    """A Letta started without OLLAMA_BASE_URL sees no models -- say so at startup."""
    with letta_town(handles=["openai/gpt-4o-mini"]):
        try:
            ChatService(autostart=False)
        except chat_mod.NoLocalModel as exc:
            assert "does not offer" in str(exc) and "OLLAMA_BASE_URL" in str(exc), exc
        else:
            raise AssertionError("started against a Letta that cannot run our model")


def test_prompt_is_in_character(game: fruitbrains.Game) -> None:
    backend = ScriptedBackend()
    service = ChatService(backend)
    service.ask("Lemon", "angry", "Apple", "how are you?")
    for _ in range(200):
        if service.poll():
            break
        time.sleep(0.005)
    system = backend.seen[0][0]["content"]
    assert "Lemon" in system and "sour" in system, system
    assert "angry" in system, "the villager's mood never reaches the model"
    assert service.conversation("Lemon").history, "history was not kept for the next turn"


def test_dialogue_round_trip(game: fruitbrains.Game) -> None:
    game.chat = ChatService(ScriptedBackend(text="Mind the puddles by the fountain!"))
    other = min((f for f in game.here() if f is not game.player),
                key=lambda f: f.pos.distance_to(game.player.pos))
    other.pos.update(game.player.pos + pygame.Vector2(50, 0))
    game.talk()
    assert game.talking is other, "conversation did not open"

    for ch in "hello there":
        game.handle(pygame.event.Event(pygame.TEXTINPUT, text=ch))
    game.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE, mod=0))
    assert game.typed == "hello ther"
    game.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
    assert game.typed == "" and game.chat.busy(other.name)

    for _ in range(240):
        step(game, 1, 40)
        if not game.chat.busy(other.name):
            break
    assert other.line == "Mind the puddles by the fountain!", other.line
    assert game.transcript[-1][0] == other.name
    step(game, 4, 41)                                   # draw the reply balloon

    game.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
    assert game.talking is None, "ESC did not close the conversation"


def test_walking_is_not_blocked_by_the_model(game: fruitbrains.Game) -> None:
    """A slow Pi must cost frames rate, not gameplay."""
    game.chat = ChatService(ScriptedBackend(delay=0.6, text="Sorry, I was daydreaming."))
    other = min((f for f in game.here() if f is not game.player),
                key=lambda f: f.pos.distance_to(game.player.pos))
    other.pos.update(game.player.pos + pygame.Vector2(50, 0))
    game.talk()
    game.chat.ask(other.name, other.emotion, game.player.name, "are you awake?")
    start = time.perf_counter()
    step(game, 12, 50)
    assert time.perf_counter() - start < 0.4, "the frame loop waited on the model"
    assert other.thinking, "no thinking indicator while the model works"
    for _ in range(400):
        step(game, 1, 52)
        if not game.chat.busy(other.name):
            break
    assert other.line == "Sorry, I was daydreaming."
    game.end_chat()


def test_nobody_wanders_off_mid_chat(game: fruitbrains.Game) -> None:
    game.chat = ChatService(ScriptedBackend(text="Stay a while!"))
    other = min((f for f in game.here() if f is not game.player),
                key=lambda f: f.pos.distance_to(game.player.pos))
    other.pos.update(game.player.pos + pygame.Vector2(60, 0))
    other.velocity.update(90, 40)                  # already strolling when you say hi
    game.talk()
    start_other = pygame.Vector2(other.pos)
    start_player = pygame.Vector2(game.player.pos)

    step(game, 600, 60)                            # ten seconds of conversation
    assert game.talking is other, "conversation dropped itself"
    assert other.pos.distance_to(start_other) < 24, (
        f"villager wandered {other.pos.distance_to(start_other):.0f} units away")
    assert game.player.pos.distance_to(start_player) < 24, "player drifted mid-chat"
    assert other.velocity.length() < 4, "villager never came to a stop"

    game.end_chat()
    assert not other.chatting and not game.player.chatting
    step(game, 240, 70)
    assert other.pos.distance_to(start_other) > 24, "villager never resumed wandering"


def test_dead_server_reports_instead_of_faking(game: fruitbrains.Game) -> None:
    """A failed request must read as a failure, never as a scripted line."""
    game.chat = ChatService(BrokenBackend())
    other = min((f for f in game.here() if f is not game.player),
                key=lambda f: f.pos.distance_to(game.player.pos))
    other.pos.update(game.player.pos + pygame.Vector2(50, 0))
    game.talk()
    other.line, other.line_left = "", 0.0      # drop the scripted hello-bark
    before = len(game.transcript)
    game.chat.ask(other.name, other.emotion, game.player.name, "hello?")
    for _ in range(400):
        step(game, 1, 80)
        if not game.chat.busy(other.name):
            break
    assert game.chat_error and "ConnectionRefused" in game.chat_error, game.chat_error
    assert len(game.transcript) == before, "a fake reply reached the transcript"
    assert not other.line, "villager spoke a line the model never produced"
    assert not game.chat.conversation(other.name).history, "a failure polluted the history"
    game.end_chat()


FAKE_OLLAMA = '''#!/usr/bin/env python3
"""A stand-in `ollama` binary: `serve` runs a tiny API, `pull` pretends to fetch."""
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

STORE = os.environ["FAKE_OLLAMA_STORE"]

def models():
    return json.load(open(STORE)) if os.path.exists(STORE) else []

if sys.argv[1] == "pull":
    json.dump(models() + [sys.argv[2]], open(STORE, "w"))
    print(f"pulled {sys.argv[2]}")
    sys.exit(0)

host, port = os.environ["OLLAMA_HOST"].split(":")

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _s(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body)
    def do_GET(self): self._s({"models": [{"name": n} for n in models()]})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/api/pull"):        # same API the real server exposes
            json.dump(models() + [body["model"]], open(STORE, "w"))
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({"status": "pulling", "completed": 1,
                                         "total": 2}).encode() + b"\\n")
            self.wfile.write(json.dumps({"status": "success"}).encode() + b"\\n")
            return
        self._s({"message": {"content": "Served by the fake ollama."}, "done": True})

HTTPServer((host, int(port)), H).serve_forever()
'''


FAKE_LETTA = '''#!/usr/bin/env python3
"""A stand-in `letta` binary: `server` runs just enough of the v1 API.

It answers the streaming route in plain JSON, the way an older Letta does, so
this also covers the backend falling back to a single blocking call.
"""
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1])
agents = {}
# Proof the game wired inference through to us, asserted by the test.
open(os.environ["FAKE_LETTA_NOTE"], "w").write(os.environ.get("OLLAMA_BASE_URL", ""))

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _s(self, payload, code=200):
        body = json.dumps(payload).encode()
        self.send_response(code); self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path.startswith("/v1/health"): return self._s({"version": "0.0-fake"})
        if self.path.startswith("/v1/models/embedding"):
            return self._s([{"handle": "ollama/nomic-embed-text:latest"}])
        if self.path.startswith("/v1/models"): return self._s([{"handle": "ollama/qwen2.5:0.5b"}])
        return self._s([a for a in agents.values()])
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.rstrip("/") == "/v1/agents":
            agents[body["name"]] = {"id": body["name"], "name": body["name"]}
            return self._s(agents[body["name"]])
        return self._s({"messages": [{"message_type": "assistant_message",
                                      "content": "Served by the fake letta."}]})

HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def _fake_letta_on_path(tmp: Path, env: dict) -> dict:
    """Add a fake `letta` to an existing fake-ollama environment."""
    exe = tmp / "letta"
    exe.write_text(FAKE_LETTA)
    exe.chmod(0o755)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return dict(env, FRUITBRAINS_LLM="letta",
                FRUITBRAINS_LETTA_URL=f"http://127.0.0.1:{port}",
                FAKE_LETTA_NOTE=str(tmp / "ollama-url.txt"))


def test_letta_stack_starts_and_stops_together(game: fruitbrains.Game) -> None:
    """--letta with nothing running brings up both layers, and takes both down."""
    saved = dict(os.environ)
    with tempfile.TemporaryDirectory() as tmp, _no_real_servers():
        try:
            env = _fake_letta_on_path(Path(tmp), _fake_ollama_on_path(Path(tmp), installed=[]))
            os.environ.update(env)
            service = ChatService()
            assert len(service.servers) == 2, f"started {len(service.servers)} servers, wanted 2"
            assert [s.label for s in service.servers] == ["ollama", "letta"]
            pids = [s.process.pid for s in service.servers]

            # The chat model and the embedding model both arrive, and Letta is
            # told where inference lives.
            pulled = json.load(open(env["FAKE_OLLAMA_STORE"]))
            assert pulled == ["qwen2.5:0.5b", chat_mod.EMBEDDING_MODEL], pulled
            assert Path(env["FAKE_LETTA_NOTE"]).read_text().endswith(
                env["FRUITBRAINS_LLM_URL"].split("//")[1]), "Letta was not pointed at Ollama"

            service.ask("Grape", "happy", "Apple", "any news?")
            assert _await_reply(service)[0][1] == "Served by the fake letta."

            service.shutdown()
            for _ in range(300):
                if not any(_alive(pid) for pid in pids):
                    break
                time.sleep(0.02)
            assert not any(_alive(pid) for pid in pids), "a server outlived the game"
        finally:
            os.environ.clear()
            os.environ.update(saved)


@contextlib.contextmanager
def _no_real_servers():
    """Point the default probe targets at dead ports.

    Without this, a real Ollama running on the developer's machine gets adopted
    and the autostart tests silently test nothing.
    """
    saved = chat_mod.OLLAMA_URL, chat_mod.OPENAI_URLS, chat_mod.LETTA_URL
    chat_mod.OLLAMA_URL, chat_mod.OPENAI_URLS = "http://127.0.0.1:9", ()
    chat_mod.LETTA_URL = "http://127.0.0.1:9"
    try:
        yield
    finally:
        chat_mod.OLLAMA_URL, chat_mod.OPENAI_URLS, chat_mod.LETTA_URL = saved


def _fake_ollama_on_path(tmp: Path, installed: list[str]) -> dict:
    """Env with a fake `ollama` first on PATH and a spare port to bind."""
    exe = tmp / "ollama"
    exe.write_text(FAKE_OLLAMA)
    exe.chmod(0o755)
    store = tmp / "models.json"
    store.write_text(json.dumps(installed))
    with socket.socket() as probe:                 # grab a free port, then let go
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return {"PATH": f"{tmp}{os.pathsep}{os.environ['PATH']}",
            "FAKE_OLLAMA_STORE": str(store),
            "FRUITBRAINS_LLM": "auto",
            "FRUITBRAINS_LLM_URL": f"http://127.0.0.1:{port}",
            "FRUITBRAINS_MODEL": "qwen2.5:0.5b"}


def test_game_starts_and_stops_its_own_server(game: fruitbrains.Game) -> None:
    saved = dict(os.environ)
    with tempfile.TemporaryDirectory() as tmp, _no_real_servers():
        try:
            os.environ.update(_fake_ollama_on_path(Path(tmp), installed=["qwen2.5:0.5b"]))
            service = ChatService()
            assert service.server is not None, "no server was started"
            assert not service.scripted and service.online
            pid = service.server.process.pid

            service.ask("Pear", "happy", "Apple", "are you there?")
            for _ in range(600):
                out = service.poll()
                if out:
                    break
                time.sleep(0.005)
            assert out[0][1] == "Served by the fake ollama.", out

            service.shutdown()
            for _ in range(200):                   # it should actually die
                if not _alive(pid):
                    break
                time.sleep(0.02)
            assert not _alive(pid), "the server we started outlived the game"
            assert service.server is None
        finally:
            os.environ.clear()
            os.environ.update(saved)


def test_a_server_we_did_not_start_is_left_alone(game: fruitbrains.Game) -> None:
    """Attaching to someone else's Ollama must not hand us a kill switch."""
    saved = dict(os.environ)
    server, port = fake_ollama(models=["qwen2.5:0.5b"], content="Already running here.")
    try:
        os.environ["FRUITBRAINS_LLM"] = "auto"
        os.environ["FRUITBRAINS_LLM_URL"] = f"http://127.0.0.1:{port}"
        os.environ["FRUITBRAINS_MODEL"] = "qwen2.5:0.5b"      # what this server has
        service = ChatService()
        assert service.server is None, "adopted a server it did not start"
        service.shutdown()
        assert _reachable(f"http://127.0.0.1:{port}/api/tags"), "shut down someone else's server"
    finally:
        os.environ.clear()
        os.environ.update(saved)
        server.shutdown()


def test_missing_model_is_pulled_once(game: fruitbrains.Game) -> None:
    saved = dict(os.environ)
    with tempfile.TemporaryDirectory() as tmp, _no_real_servers():
        try:
            env = _fake_ollama_on_path(Path(tmp), installed=[])      # nothing installed
            os.environ.update(env)
            service = ChatService()                                   # should pull, then run
            try:
                assert json.load(open(env["FAKE_OLLAMA_STORE"])) == ["qwen2.5:0.5b"]
            finally:
                service.shutdown()

            os.environ["FAKE_OLLAMA_STORE"] = str(Path(tmp) / "empty.json")
            before = _ollama_pids(env["PATH"])
            try:
                ChatService(pull=False).shutdown()
            except chat_mod.NoLocalModel as exc:
                assert "--no-pull" in str(exc), str(exc)
            else:
                raise AssertionError("--no-pull still downloaded a model")
            for _ in range(200):
                leaked = _ollama_pids(env["PATH"]) - before
                if not leaked:
                    break
                time.sleep(0.02)
            assert not leaked, f"a failed startup orphaned the server: {leaked}"
        finally:
            os.environ.clear()
            os.environ.update(saved)


def test_startup_requires_a_model(game: fruitbrains.Game) -> None:
    """No server, no launch -- with a message that says how to fix it."""
    saved = dict(os.environ)
    try:
        os.environ["FRUITBRAINS_LLM"] = "auto"
        os.environ["FRUITBRAINS_LLM_URL"] = "http://127.0.0.1:9"   # discard port
        try:
            chat_mod.detect_backend(require=True, autostart=False)
        except chat_mod.NoLocalModel as exc:
            assert "ollama serve" in str(exc) and "--canned" in str(exc), str(exc)
        else:
            raise AssertionError("detect_backend accepted a town with no model in it")
        # Scripted lines stay available, but only when asked for by name.
        os.environ["FRUITBRAINS_LLM"] = "canned"
        backend = chat_mod.detect_backend()
        assert backend.scripted and not backend.online
        with contextlib.redirect_stdout(io.StringIO()):
            assert fruitbrains.main(["--canned", "--help"]) == 0
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_startup_checks_the_model_is_loaded(game: fruitbrains.Game) -> None:
    """A running server with the wrong model is caught before a window opens."""
    server, port = fake_ollama(models=["llama3.2:1b"])
    try:
        backend = chat_mod.OllamaBackend(f"http://127.0.0.1:{port}", "qwen2.5:0.5b")
        try:
            backend.check(pull=False)
        except chat_mod.NoLocalModel as exc:
            assert "ollama pull qwen2.5:0.5b" in str(exc), str(exc)
        else:
            raise AssertionError("a missing model was not reported")
        chat_mod.OllamaBackend(f"http://127.0.0.1:{port}", "llama3.2:1b").check()   # present
        assert not chat_mod.OllamaBackend(f"http://127.0.0.1:{port}", "llama3.2:70b").has_model(), (
            "family match let a missing tag through -- every request would 404")
    finally:
        server.shutdown()


def test_live_http_round_trip(game: fruitbrains.Game) -> None:
    """The real request/response shape, against a stand-in Ollama."""
    server, port = fake_ollama(models=["qwen2.5:0.5b"],
                               content='  "Peachy! *waves* I napped in the sun."  ')
    try:
        os.environ["FRUITBRAINS_LLM"] = "auto"
        os.environ["FRUITBRAINS_LLM_URL"] = f"http://127.0.0.1:{port}"
        os.environ["FRUITBRAINS_MODEL"] = "qwen2.5:0.5b"
        backend = chat_mod.detect_backend()
        assert isinstance(backend, chat_mod.OllamaBackend)
        service = ChatService(backend)
        service.ask("Peach", "sleepy", "Apple", "what have you been up to?")
        replies = []
        for _ in range(600):
            replies = service.poll()
            if replies:
                break
            time.sleep(0.005)
        villager, text, error = replies[0]
        assert error is None and text == "Peachy! I napped in the sun.", (text, error)
        # The background warm-up talks to the same server, so find our own turn
        # rather than assuming it was the last thing to arrive.
        sent = next(s for s in server.seen
                    if "what have you been up to?" in s["messages"][-1]["content"])
        assert sent["options"]["num_predict"] == chat_mod.MAX_TOKENS
        assert "answering them" in sent["messages"][-1]["content"]
    finally:
        os.environ.pop("FRUITBRAINS_LLM_URL", None)
        os.environ["FRUITBRAINS_LLM"] = "canned"
        server.shutdown()


def test_reply_streams_word_by_word(game: fruitbrains.Game) -> None:
    """The player must see progress, not an opaque 'thinking' state."""
    server, port = fake_ollama(models=["m"], content="The tulips are coming along nicely.",
                               token_delay=0.02)
    try:
        game.chat = ChatService(chat_mod.OllamaBackend(f"http://127.0.0.1:{port}", "m"))
        other = min((f for f in game.here() if f is not game.player),
                    key=lambda f: f.pos.distance_to(game.player.pos))
        other.pos.update(game.player.pos + pygame.Vector2(50, 0))
        game.talk()
        game.chat.ask(other.name, other.emotion, game.player.name, "how's the garden?")

        seen_partial, seen_timer = "", 0.0
        for _ in range(900):
            step(game, 1, 90)
            seen_partial = seen_partial or game.chat.partial(other.name)
            seen_timer = max(seen_timer, game.chat.waited(other.name))
            if not game.chat.busy(other.name):
                break
        assert seen_partial and seen_partial != "The tulips are coming along nicely.", (
            f"never saw a partial reply ({seen_partial!r})")
        assert seen_timer > 0, "no elapsed-time signal for the UI"
        assert other.line == "The tulips are coming along nicely.", other.line
        assert game.chat.partial(other.name) == "", "partial text outlived the reply"
        game.end_chat()
    finally:
        server.shutdown()


def test_cancelling_a_slow_reply(game: fruitbrains.Game) -> None:
    """ESC out of a conversation and the pending reply is dropped, not spoken later."""
    server, port = fake_ollama(models=["m"], content="I will ramble on for quite a while yet",
                               token_delay=0.25)
    try:
        game.chat = ChatService(chat_mod.OllamaBackend(f"http://127.0.0.1:{port}", "m"))
        other = min((f for f in game.here() if f is not game.player),
                    key=lambda f: f.pos.distance_to(game.player.pos))
        other.pos.update(game.player.pos + pygame.Vector2(50, 0))
        game.talk()
        game.chat.ask(other.name, other.emotion, game.player.name, "say something long")
        step(game, 30, 100)
        assert game.chat.busy(other.name)

        game.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
        assert game.talking is None
        assert not game.chat.busy(other.name), "cancel did not clear the pending state"
        other.line, other.line_left = "", 0.0
        step(game, 180, 105)
        assert not other.line, "a cancelled reply was spoken anyway"
        assert not game.chat.conversation(other.name).history, "cancelled reply entered history"
    finally:
        server.shutdown()


def test_a_stalled_model_becomes_an_error(game: fruitbrains.Game) -> None:
    """A server that goes quiet mid-reply must not hang the conversation forever."""
    server, port = fake_ollama(models=["m"], content="one two three four", stall_after=2)
    saved = serve_mod.STALL_TIMEOUT          # the deadline lives with the HTTP plumbing
    serve_mod.STALL_TIMEOUT = 0.75           # don't make the suite wait 25 s
    try:
        game.chat = ChatService(chat_mod.OllamaBackend(f"http://127.0.0.1:{port}", "m"))
        other = min((f for f in game.here() if f is not game.player),
                    key=lambda f: f.pos.distance_to(game.player.pos))
        other.pos.update(game.player.pos + pygame.Vector2(50, 0))
        game.talk()
        other.line, other.line_left = "", 0.0
        game.chat.ask(other.name, other.emotion, game.player.name, "hello?")
        for _ in range(900):
            step(game, 1, 110)
            if not game.chat.busy(other.name):
                break
        assert not game.chat.busy(other.name), "a stalled reply never resolved"
        assert game.chat_error, "a stall was not reported as an error"
        game.end_chat()
    finally:
        serve_mod.STALL_TIMEOUT = saved
        server.shutdown()


def test_model_is_sized_to_the_machine(game: fruitbrains.Game) -> None:
    big = hardware.Hardware("RTX 4090", 24564, 65536)
    mid = hardware.Hardware("RTX 3060", 12288, 32768)
    desktop_cpu = hardware.Hardware("", 0, 16000)
    pi4 = hardware.Hardware("", 0, 3800, pi=True)
    tiny = hardware.Hardware("", 0, 900, pi=True)
    assert hardware.recommend_model(big) == "qwen2.5:14b"
    assert hardware.recommend_model(mid) == "qwen2.5:7b"
    assert hardware.recommend_model(desktop_cpu) == "qwen2.5:3b"
    assert hardware.recommend_model(pi4) == "qwen2.5:0.5b"
    assert hardware.recommend_model(tiny) == "smollm2:360m"
    # Bigger hardware never picks a smaller model.
    ladder = [hardware.recommend_model(hardware.Hardware("gpu", vram, 65536))
              for vram in (3600, 6100, 10500, 23000)]
    assert ladder == ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"], ladder

    saved = dict(os.environ)
    try:
        os.environ["FRUITBRAINS_MODEL"] = "smollm2:360m"
        assert chat_mod.chosen_model() == ("smollm2:360m", "FRUITBRAINS_MODEL")
        os.environ.pop("FRUITBRAINS_MODEL")
        model, why = chat_mod.chosen_model()
        assert model and why, (model, why)          # whatever this machine is
    finally:
        os.environ.clear()
        os.environ.update(saved)
    assert hardware.probe().ram_mb > 0, "no RAM detected at all"


def test_reply_tidying(game: fruitbrains.Game) -> None:
    assert tidy('  "Hello there!"  ') == "Hello there!"
    assert tidy("Lemon: *shrugs* fine, I suppose.") == "fine, I suppose."
    long = tidy("word " * 200)
    assert len(long) <= chat_mod.REPLY_CHARS + 3
    assert CannedBackend().reply([{"role": "user", "content": "how are you?"}], "sad")


def contact_sheet(game: fruitbrains.Game) -> None:
    """One PNG: town at a distance, a close-up, and the cottage interior."""
    shots = []
    game.cam.target_zoom = game.cam.zoom = 0.8
    game.player.pos.update(544, 560)
    game.cam.snap_to(game.player.pos)
    step(game, 90, 5)
    shots.append(game.screen.copy())

    game.cam.target_zoom = game.cam.zoom = 2.6
    game.cam.snap_to(game.player.pos)
    game.player.feel("love", 6.0)
    step(game, 40, 12)
    shots.append(game.screen.copy())

    game.enter(game.scenes["town"].doors[0])
    game.fade = 0.0
    game.cam.target_zoom = game.cam.zoom = 1.5
    game.cam.snap_to(game.player.pos)
    step(game, 90, 20)
    shots.append(game.screen.copy())

    w, h = shots[0].get_size()
    sheet = pygame.Surface((w, h * len(shots)))
    for i, shot in enumerate(shots):
        sheet.blit(shot, (0, i * h))
    pygame.image.save(sheet, SHOT)


def main() -> int:
    game = fruitbrains.Game()
    checks = (test_moods_decay_to_happy, test_every_mood_draws, test_solids_block,
              test_door_round_trip, test_zoom_is_bounded, test_mixed_pixel_scales, test_talking,
              test_prompt_is_in_character, test_dialogue_round_trip,
              test_walking_is_not_blocked_by_the_model, test_nobody_wanders_off_mid_chat,
              test_dead_server_reports_instead_of_faking,
              test_game_starts_and_stops_its_own_server,
              test_a_server_we_did_not_start_is_left_alone, test_missing_model_is_pulled_once,
              test_startup_requires_a_model, test_startup_checks_the_model_is_loaded,
              test_live_http_round_trip, test_reply_streams_word_by_word,
              test_cancelling_a_slow_reply, test_a_stalled_model_becomes_an_error,
              test_model_is_sized_to_the_machine, test_reply_tidying,
              test_letta_gives_each_villager_one_agent,
              test_letta_is_sent_the_line_not_the_history,
              test_letta_memory_reaches_the_panel, test_a_villager_never_speaks_a_tool_call,
              test_letta_errors_are_not_silence, test_forget_deletes_only_our_agents,
              test_letta_handles_are_resolved_to_full_tags,
              test_letta_needs_a_model_it_can_reach,
              test_letta_stack_starts_and_stops_together)
    # Nothing in here may touch a real model server on the developer's machine:
    # tests that need one stand up their own fake on a spare port.
    with _no_real_servers():
        for check in checks:
            check(game)
            print(f"  ok  {check.__name__}")
    contact_sheet(game)
    print(f"  ok  contact sheet -> {SHOT.name}")
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
