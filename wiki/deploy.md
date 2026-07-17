# Deployment Design: Commit-Addressable Browser Playtesting

## 1. Purpose

This document defines a deployment architecture that allows players to launch and play the Python roguelike from a web browser at any supported Git commit.

The game remains a terminal application. The browser displays that terminal through a web terminal frontend, while the selected commit runs inside an isolated server-side container.

The intended result is a URL-driven playtesting system such as:

```text
https://play.example.com/latest
https://play.example.com/branch/main
https://play.example.com/tag/v0.4.0
https://play.example.com/commit/8d36f23
```

Each URL resolves to an immutable game build and launches a temporary, isolated play session.

---

## 2. Goals

### Primary goals

- Make the current development build playable in a browser.
- Make historical commits playable by commit SHA.
- Preserve the existing terminal-oriented game loop and input model.
- Support the current ASCII renderer.
- Allow a terminal graphics renderer, such as SIXEL, to be added later.
- Run one isolated game process per player session.
- Avoid keeping every historical build continuously running.
- Make builds reproducible and attributable to exact commits.
- Support automated deployment from GitHub Actions.

### Secondary goals

- Provide branch, tag, and recent-commit selectors.
- Preserve save data when appropriate.
- Capture crash logs and basic session diagnostics.
- Allow testers to report the exact build they played.
- Support private or invitation-only testing before public release.
- Add branded pages and custom browser controls later without changing the game engine.

---

## 3. Non-goals

The initial system will not:

- Rewrite the game as a browser-native application.
- Replace the terminal renderer with Canvas or WebGL.
- Run Python directly in GitHub Pages.
- Keep one permanent server process alive for every commit.
- Expose a general shell to players.
- Provide multiplayer synchronization.
- Guarantee that every commit in repository history is playable.
- Guarantee compatibility for commits created before the deployment contract existed.

Historical commits are playable only when they can be built using the expected container and dependency configuration.

---

## 4. Architectural Principle

Every playable commit should produce an immutable artifact.

Containers are launched only when a player chooses a build.

```text
Git commit
    |
    v
GitHub Actions
    |
    v
Container image tagged with commit SHA
    |
    v
Container registry
    |
    v
Session launcher
    |
    v
Temporary isolated container
    |
    v
ttyd + Python game
    |
    v
Browser terminal
```

The system shares terminal behavior across local and browser play. It does not maintain a separate browser game renderer.

---

## 5. High-Level Architecture

```text
+------------------------+
| GitHub Repository      |
| source, Dockerfile, CI |
+-----------+------------+
            |
            | push / tag
            v
+------------------------+
| GitHub Actions         |
| test, build, publish   |
+-----------+------------+
            |
            | immutable image
            v
+------------------------+
| GitHub Container       |
| Registry (GHCR)        |
| game:<full-commit-sha> |
+-----------+------------+
            |
            | pull by SHA or digest
            v
+------------------------+       +------------------------+
| Launcher / Session API |<----->| Session Metadata DB    |
+-----------+------------+       +------------------------+
            |
            | create isolated runtime
            v
+------------------------+
| Docker / Machine Host  |
| one container/session  |
+-----------+------------+
            |
            | HTTPS + WebSocket proxy
            v
+------------------------+
| Browser                |
| ttyd / xterm.js client |
+------------------------+
```

---

## 6. Core Components

## 6.1 Game container

Each container image contains:

- The game source at one exact commit.
- The Python runtime.
- Locked Python dependencies.
- The terminal-to-browser server.
- A non-root runtime user.
- A fixed entrypoint that starts only the game.
- Build metadata containing the commit SHA.

Example runtime command:

```text
ttyd --writable --port 7681 python -m your_game
```

The container must not expose an interactive shell as its entrypoint.

### Required build metadata

Each image should include:

- Full Git commit SHA.
- Build timestamp.
- Repository identifier.
- Optional branch or tag name.
- Game version, when available.
- Dependency lockfile checksum.
- Container image digest after publication.

Expose the commit SHA inside the game, ideally in an About, Help, or Feedback screen.

---

## 6.2 Container registry

GitHub Container Registry stores built game images.

Recommended naming:

```text
ghcr.io/<owner>/<repository>/game:<full-commit-sha>
```

Optional convenience tags:

```text
ghcr.io/<owner>/<repository>/game:main
ghcr.io/<owner>/<repository>/game:latest
ghcr.io/<owner>/<repository>/game:v0.4.0
```

Convenience tags must not be used as the final runtime identity. The launcher should resolve them to a full commit SHA and preferably an immutable image digest before starting a session.

---

## 6.3 Launcher service

The launcher is a small web service responsible for:

- Resolving branch, tag, or short-SHA requests.
- Verifying that a requested commit belongs to the repository.
- Checking whether a playable image exists.
- Resolving the image to an immutable digest.
- Starting an isolated container.
- Assigning a random session identifier.
- Recording session metadata.
- Returning the browser play URL.
- Enforcing resource and inactivity limits.
- Stopping and deleting expired sessions.
- Preventing players from requesting arbitrary images or commands.

The launcher is part of the trusted control plane.

It should never accept raw Docker arguments, arbitrary image names, shell commands, mount paths, or entrypoint overrides from the browser.

---

## 6.4 Reverse proxy

A reverse proxy provides:

- HTTPS termination.
- WebSocket forwarding.
- Routing from a session URL to the correct container.
- Optional authentication.
- Rate limiting.
- Request size and timeout controls.
- Access logging.

Suitable implementations include Caddy, nginx, Traefik, or a platform-specific proxy.

A session may be routed using a path:

```text
https://play.example.com/sessions/<session-id>/
```

or a subdomain:

```text
https://<session-id>.play.example.com/
```

Path routing is simpler operationally. Subdomain routing can provide cleaner isolation but requires wildcard DNS and TLS support.

---

## 6.5 Browser terminal

The initial browser experience is the web client served by `ttyd`, which uses a terminal emulator in the browser.

The browser terminal should support:

- Keyboard input.
- Terminal resize events.
- ANSI color and cursor control.
- WebSocket reconnect behavior where possible.
- Full-screen or large-screen layout.
- A visible build identifier.
- A feedback link outside or inside the terminal.

The game remains responsible for terminal drawing and input semantics.

---

## 6.6 Session metadata store

A small database or durable key-value store should track:

```text
session_id
requested_ref
resolved_commit_sha
image_digest
created_at
last_activity_at
expires_at
runtime_id
runtime_host
status
player_id or anonymous token
save_id
exit_code
termination_reason
```

SQLite is sufficient for an initial single-host deployment.

PostgreSQL or a managed database becomes useful when the launcher runs on multiple hosts.

---

## 7. Commit Resolution

The launcher should accept the following reference types:

### Full commit SHA

```text
/commit/8d36f23c...
```

This is the preferred immutable public identifier.

### Short commit SHA

```text
/commit/8d36f23
```

The launcher must reject ambiguous short SHAs.

### Branch

```text
/branch/main
```

A branch is mutable. Resolve it to a full SHA when the player starts the session and record that SHA.

### Tag

```text
/tag/v0.4.0
```

Resolve the tag to its commit and then to the corresponding image digest.

### Latest

```text
/latest
```

Define this explicitly. Recommended meaning:

```text
latest = most recent successful playable build from the default branch
```

Do not define `latest` as the most recently pushed image of any branch.

---

## 8. Build Pipeline

A GitHub Actions workflow should run for relevant pushes and tags.

### Pipeline stages

1. Check out the exact commit.
2. Install build tools.
3. Run automated tests.
4. Verify the deployment contract.
5. Build the container image.
6. Start the container in CI.
7. Run a startup smoke test.
8. Push the image to GHCR.
9. Record the image digest.
10. Publish build metadata.
11. Optionally notify the launcher or update a build index.

### Deployment contract checks

A commit is considered playable only if it has:

- A valid container build configuration.
- A valid dependency lockfile.
- A known game startup command.
- A successful smoke test.
- A terminal size and encoding compatible with the browser terminal.
- No request for unsupported host capabilities.

A failed pipeline should leave the commit visible in history but mark it as not playable.

---

## 9. Example GitHub Actions Workflow

This is illustrative and should be pinned and reviewed before production use.

```yaml
name: Build playable commit

on:
  push:
    branches:
      - "**"
    tags:
      - "**"

permissions:
  contents: read
  packages: write

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/game

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Check out source
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push commit image
        id: build
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          labels: |
            org.opencontainers.image.revision=${{ github.sha }}
            org.opencontainers.image.source=${{ github.server_url }}/${{ github.repository }}

      - name: Record image digest
        run: |
          echo "${{ steps.build.outputs.digest }}"
```

Production workflows should pin third-party actions to immutable commit SHAs rather than floating major-version tags.

---

## 10. Example Container Contract

```dockerfile
FROM python:3.13-slim

ARG GAME_COMMIT=unknown

LABEL org.opencontainers.image.revision="${GAME_COMMIT}"

RUN useradd \
    --create-home \
    --uid 10001 \
    --shell /usr/sbin/nologin \
    game

WORKDIR /opt/game

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir .

COPY . .

ENV GAME_COMMIT="${GAME_COMMIT}"
ENV TERM="xterm-256color"
ENV PYTHONUNBUFFERED="1"

USER game

EXPOSE 7681

ENTRYPOINT [
  "ttyd",
  "--writable",
  "--port", "7681",
  "python", "-m", "your_game"
]
```

The exact dependency installer and `ttyd` installation method depend on the project and base image.

The production image should include a known, pinned `ttyd` version rather than relying on an unversioned system package.

---

## 11. Session Lifecycle

## 11.1 Create

The player selects a commit, branch, tag, or latest build.

The launcher:

1. Resolves the requested reference.
2. Confirms a successful playable build exists.
3. Resolves the image to an immutable digest.
4. Creates a random session ID.
5. Creates an isolated writable save location.
6. Starts a container with strict limits.
7. Waits for a health check.
8. Returns the play URL.

---

## 11.2 Active

While active, the system records:

- Last connection time.
- Last meaningful activity time.
- Container health.
- Process state.
- Optional terminal dimensions.
- Optional game heartbeat.
- Selected commit and image digest.

Activity should be based on more than open TCP connections when possible. A stale browser tab should not keep a session alive indefinitely.

---

## 11.3 Disconnect

Recommended initial behavior:

1. Mark the session disconnected.
2. Ask the game to save if the process supports a signal or control command.
3. Preserve the container for a short reconnect window.
4. Terminate the container after the reconnect window expires.
5. Retain or remove the save according to the save policy.

Suggested reconnect window:

```text
5 to 15 minutes
```

---

## 11.4 Expiration

A session should stop when any of the following occurs:

- The player exits the game.
- The game process crashes.
- The browser remains disconnected beyond the reconnect window.
- The session exceeds its maximum lifetime.
- The session exceeds its inactivity timeout.
- The player explicitly ends the session.
- An administrator terminates it.
- Resource limits are exceeded.

The runtime container should be deleted after termination.

---

## 12. Save Data

Save behavior must be explicit because different commits may use incompatible formats.

### Recommended save identity

```text
player or anonymous identity
repository
commit or save-compatibility version
save slot
```

### Save policies

#### Ephemeral

All save data is deleted when the session ends.

Best for:

- very early development;
- public anonymous testing;
- unstable save formats.

#### Commit-scoped

Saves are available only to the exact commit that created them.

Best for:

- reproducibility;
- debugging;
- historical playtesting.

#### Compatibility-scoped

The game declares a save format version. Commits with the same format version may share saves.

Best for:

- longer-running testers;
- controlled migrations.

### Initial recommendation

Use commit-scoped saves during early development.

Add explicit import or migration tooling later. Never silently load a save from an incompatible commit.

---

## 13. Terminal Rendering Strategy

## 13.1 ASCII baseline

ASCII is the universal compatibility mode and should remain available.

Recommended terminal environment:

```text
TERM=xterm-256color
UTF-8 locale
known minimum terminal dimensions
```

The game should respond safely to browser resize events.

---

## 13.2 Graphical tiles in the terminal

The target path for browser-compatible terminal graphics is a protocol supported by both:

- the selected local terminal;
- the browser terminal implementation.

SIXEL is a candidate when supported end-to-end by the terminal server and browser frontend.

The game should keep this behind a renderer interface:

```python
class Renderer:
    def render(self, frame) -> None:
        ...

class AsciiRenderer(Renderer):
    ...

class SixelRenderer(Renderer):
    ...
```

Renderer selection may be based on:

- configuration;
- environment variable;
- terminal capability detection;
- explicit command-line option.

The browser deployment should support an ASCII fallback when image protocol negotiation fails.

---

## 14. API Sketch

## 14.1 List builds

```http
GET /api/builds?branch=main&limit=50
```

Example response:

```json
{
  "builds": [
    {
      "commit": "8d36f23c...",
      "short_commit": "8d36f23",
      "message": "Add inventory sorting",
      "branch": "main",
      "created_at": "2026-07-17T20:15:00Z",
      "playable": true,
      "image_digest": "sha256:..."
    }
  ]
}
```

---

## 14.2 Resolve a reference

```http
GET /api/resolve/main
```

Example response:

```json
{
  "requested_ref": "main",
  "commit": "8d36f23c...",
  "playable": true
}
```

---

## 14.3 Create a session

```http
POST /api/sessions
Content-Type: application/json

{
  "ref": "8d36f23"
}
```

Example response:

```json
{
  "session_id": "random-unpredictable-id",
  "commit": "8d36f23c...",
  "play_url": "/sessions/random-unpredictable-id/",
  "expires_at": "2026-07-17T22:00:00Z"
}
```

---

## 14.4 Read session status

```http
GET /api/sessions/<session-id>
```

---

## 14.5 End a session

```http
DELETE /api/sessions/<session-id>
```

---

## 15. Security Model

The browser client is untrusted.

The selected game commit is also potentially untrusted because historical or feature-branch code can contain mistakes or malicious behavior.

Every game container must be treated as a sandboxed workload.

### Required controls

- Run as a non-root user.
- Never mount the Docker socket into a game container.
- Never mount the repository host filesystem.
- Use a read-only root filesystem where practical.
- Provide a dedicated writable save directory.
- Drop Linux capabilities.
- Disable privilege escalation.
- Apply CPU limits.
- Apply memory limits.
- Apply process-count limits.
- Apply disk quotas.
- Restrict network access.
- Use random, unguessable session IDs.
- Validate all commit references server-side.
- Permit only images from the configured repository.
- Use HTTPS and secure WebSockets.
- Apply rate limits to session creation.
- Expire inactive sessions.
- Do not expose shell access.
- Do not allow arbitrary command-line arguments.
- Keep launcher and runtime credentials outside game containers.

### Network policy

The game container should have no outbound internet access unless the game requires it.

If outbound access is required, allow only the minimum necessary destinations.

---

## 16. Resource Limits

Suggested initial limits per session:

```text
CPU: 0.25 to 1 core
Memory: 256 to 512 MiB
Processes: 64 or fewer
Writable storage: 10 to 100 MiB
Maximum session lifetime: 2 to 8 hours
Inactive timeout: 15 to 30 minutes
Reconnect grace period: 5 to 15 minutes
```

Tune these from observed game behavior.

The launcher must refuse new sessions when the host lacks sufficient capacity.

---

## 17. Build Retention

Keeping an image for every commit forever may consume significant registry storage.

### Recommended policy

Keep permanently:

- Tagged releases.
- Milestone builds.
- Commits explicitly marked for preservation.
- A chosen number of default-branch builds.

Keep temporarily:

- Feature-branch builds.
- Pull-request builds.
- Failed or superseded development builds.

Example policy:

```text
Tagged releases: permanent
Default branch: last 250 successful builds
Active feature branches: 30 to 90 days
Closed pull requests: 14 to 30 days
Failed builds: metadata only
```

### Rebuilding old commits

If an image has expired, the system may offer to rebuild the commit.

This is weaker than retaining the original image because:

- base image tags may have changed;
- package indexes may have changed;
- system packages may have changed;
- dependencies may have disappeared.

To improve rebuild reproducibility:

- lock Python dependencies;
- pin base images by digest;
- pin system packages where practical;
- pin CI actions;
- archive required assets;
- avoid downloading unversioned resources at build time.

---

## 18. GitHub Pages

GitHub Pages may host the static landing page and build selector.

It cannot host:

- Python game processes;
- PTYs;
- `ttyd`;
- session containers;
- the launcher API;
- the WebSocket-to-terminal backend.

Recommended split:

```text
GitHub Pages
    |
    | static build selector and documentation
    v
Launcher API on a runtime host
    |
    v
Temporary game container
```

Alternatively, serve the landing page directly from the launcher service and omit GitHub Pages in the initial deployment.

---

## 19. Hosting Options

## 19.1 Single Docker VPS

Recommended for the first implementation.

Components:

```text
Caddy or nginx
launcher service
SQLite
Docker Engine
game containers
```

Advantages:

- Simple mental model.
- Full control of WebSockets and process lifetimes.
- Low initial cost.
- Easy local reproduction.
- Straightforward logging.

Disadvantages:

- One host is a single point of failure.
- Capacity is limited.
- The launcher must manage container cleanup carefully.
- Docker control-plane access must be strongly protected.

---

## 19.2 Machine or container platform

A platform that supports long-lived WebSockets and on-demand isolated machines can replace the VPS runtime.

Advantages:

- Easier scaling.
- Stronger workload isolation options.
- Less host maintenance.

Disadvantages:

- More platform-specific code.
- Startup latency may be noticeable.
- Cost can be harder to predict.
- Persistent save storage requires explicit design.

---

## 19.3 Kubernetes

Not recommended initially.

It becomes reasonable when:

- many concurrent sessions are expected;
- multiple launcher replicas are required;
- workload scheduling spans multiple nodes;
- operational expertise already exists.

---

## 20. Observability

Record enough information to reproduce tester reports.

### Per build

- Commit SHA.
- Image digest.
- Build logs.
- Test results.
- Build timestamp.
- Dependency lock checksum.
- Playable status.

### Per session

- Session ID.
- Commit SHA.
- Image digest.
- Start and stop times.
- Exit code.
- Termination reason.
- Crash output.
- Resource usage.
- Save identifier.
- Browser-visible feedback identifier.

### Privacy

Avoid recording raw terminal input by default.

Terminal recordings may contain names, private test data, or secrets. If recording is enabled, disclose it clearly and define retention limits.

---

## 21. Feedback Workflow

Every play session should make it easy to report:

- Commit SHA.
- Session ID.
- Approximate time.
- What the player was doing.
- Expected behavior.
- Actual behavior.
- Optional screenshot.
- Optional save or replay identifier.

A simple in-game command or browser-side button can open a prefilled issue template.

Do not include security-sensitive session tokens in issue URLs.

---

## 22. Failure Handling

### Image not found

Display:

- requested commit;
- whether the build failed or expired;
- link to CI logs when appropriate;
- option to select another build.

### Container startup failure

Record:

- image digest;
- runtime error;
- health-check output;
- container logs.

Return a generic player-facing error without exposing host internals.

### Game crash

- Preserve stderr and exit code.
- Mark the session failed.
- Preserve the save when safe.
- Offer a feedback link containing the commit and session identifier.

### Proxy or WebSocket failure

- Preserve the runtime during the reconnect grace period.
- Allow the player to reconnect using the same session token.
- Stop the session after expiration.

---

## 23. Health Checks

The runtime should expose a health signal separate from terminal output where possible.

Possible checks:

- Container process is running.
- `ttyd` port is listening.
- HTTP endpoint responds.
- WebSocket upgrade succeeds.
- Game process is still alive.
- Optional game heartbeat file or local socket.

The launcher should not return the play URL until the runtime passes its startup check.

---

## 24. Local Development

The deployment architecture should be reproducible locally.

Suggested commands:

```text
make image
make play
make launcher
make integration-test
```

A local integration test should:

1. Build the current commit.
2. Start the launcher.
3. Request a session.
4. Verify the terminal page loads.
5. Verify the WebSocket connects.
6. Send a harmless input.
7. Confirm the game responds.
8. End the session.
9. Confirm the container is removed.

---

## 25. Phased Implementation

## Phase 1: Single playable current build

Deliver:

- Game container.
- `ttyd` browser terminal.
- Manual deployment to one VPS.
- HTTPS reverse proxy.
- One active build.
- Basic resource limits.

Success criterion:

```text
A tester can open a URL and play the current game build.
```

---

## Phase 2: Commit-addressable images

Deliver:

- GitHub Actions container build.
- Full-SHA image tags.
- GHCR publication.
- Build smoke test.
- Commit SHA visible in-game.

Success criterion:

```text
Every successful pushed commit produces a uniquely identifiable playable image.
```

---

## Phase 3: On-demand launcher

Deliver:

- Session API.
- Commit resolution.
- On-demand container startup.
- Random session URLs.
- Automatic cleanup.
- Reconnect grace period.
- SQLite session metadata.

Success criterion:

```text
A tester can choose a commit and receive an isolated play session.
```

---

## Phase 4: Build browser

Deliver:

- Recent commits list.
- Branch and tag selection.
- Build status.
- Play buttons.
- Friendly failure pages.
- Optional GitHub Pages frontend.

Success criterion:

```text
A tester can browse successful builds without manually entering a SHA.
```

---

## Phase 5: Saves and feedback

Deliver:

- Commit-scoped saves.
- Anonymous or authenticated player identity.
- Crash capture.
- Prefilled issue reporting.
- Optional session replay or event log.

Success criterion:

```text
Testers can return to sessions and report bugs with exact build context.
```

---

## Phase 6: Graphical terminal renderer

Deliver:

- Capability-based renderer selection.
- SIXEL or another supported terminal image protocol.
- Browser-terminal verification.
- ASCII fallback.
- Performance and bandwidth measurements.

Success criterion:

```text
The same terminal game can use graphical tiles locally and through the browser.
```

---

## 26. Initial Technology Recommendation

For the first production-quality prototype:

```text
Source control: GitHub
CI: GitHub Actions
Registry: GitHub Container Registry
Runtime host: one Linux VPS
Container runtime: Docker
Reverse proxy: Caddy or nginx
Terminal server: ttyd
Game runtime: Python
Launcher API: FastAPI or aiohttp
Session database: SQLite
Build selector: launcher-hosted static page or GitHub Pages
```

This keeps the initial system small while preserving a path to larger-scale infrastructure.

---

## 27. Key Design Decisions

### Decision: Use server-side containers

Reason:

The existing Python terminal application can run unchanged or with minimal adaptation.

### Decision: Build one immutable image per playable commit

Reason:

A player report can be tied to an exact artifact rather than a moving branch.

### Decision: Launch containers on demand

Reason:

Historical builds remain available without consuming runtime resources continuously.

### Decision: Use one container per player session

Reason:

Players must not share process state, terminal input, files, or saves.

### Decision: Keep ASCII as a fallback

Reason:

Terminal image protocols vary across terminal and browser implementations.

### Decision: Treat commit code as untrusted

Reason:

Feature branches and historical code may contain unsafe or accidental host interactions.

### Decision: Resolve mutable references before launch

Reason:

Branch, tag, and latest URLs must map to a recorded immutable SHA and image digest.

---

## 28. Open Questions

The implementation should resolve the following before public launch:

1. Which local terminal and browser-terminal graphics protocol will be supported?
2. Will anonymous users be allowed to launch sessions?
3. What is the maximum number of concurrent sessions?
4. What is the session inactivity timeout?
5. Should saves be ephemeral, commit-scoped, or compatibility-scoped?
6. How long should commit images be retained?
7. Should pull-request builds be available publicly?
8. Should containers have outbound network access?
9. How should old commits with obsolete build files be handled?
10. Will terminal sessions be recorded for debugging?
11. What information should be included in tester feedback?
12. Will the game support reconnecting to an existing process?
13. What terminal dimensions are officially supported?
14. How should mobile input be handled?
15. Which build is represented by `/latest`?

---

## 29. Acceptance Criteria

The system is ready for initial playtesting when:

- A pushed commit produces a container image tagged by full SHA.
- The image records the same SHA in its metadata.
- The image passes an automated startup smoke test.
- The launcher accepts a valid commit SHA.
- The launcher rejects unknown or unauthorized references.
- A new isolated game container starts for each session.
- The browser terminal connects over HTTPS and secure WebSockets.
- Two simultaneous players do not share state.
- The game displays the selected commit SHA.
- Containers have CPU, memory, process, and lifetime limits.
- A disconnected session is cleaned up automatically.
- A crashed session records its commit, image digest, and exit status.
- No player can access a shell or control the container runtime.
- The selected commit can be reproduced from the session metadata.

---

## 30. Summary

The deployment system should treat each successful Git commit as a versioned game artifact.

GitHub Actions builds and publishes the artifact. A launcher resolves the selected commit and starts a temporary isolated container. `ttyd` exposes the game terminal to the browser. A reverse proxy handles HTTPS and WebSocket routing. Session metadata records the exact commit and image digest used.

The core model is:

```text
commit -> immutable image -> temporary isolated session -> browser terminal
```

This preserves the command-line development workflow while making current and historical game builds continuously available to testers.