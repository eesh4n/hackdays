# RivalRuns

A two-player, two-laptop endless runner where your body is the controller — for both players.

**Player A** runs down a 3-lane track by moving, jumping, and ducking in front of a webcam.
**Player B** watches the track from a second laptop and places obstacles into A's lanes using
hand gestures, trying to bring the run to an end. Coins scattered along the track bank into a
shield that can block a hit.

Built on [Ursina](https://www.ursinaengine.org/) (Panda3D) for real 3D rendering and
[MediaPipe](https://developers.google.com/mediapipe) for pose and hand tracking — no
controllers or keyboard required for either player during a real match, though both sides
have a keyboard fallback for testing without a camera.

---

## How it works

Two laptops, one WebSocket connection between them:

| | Laptop 1 — "the host" | Laptop 2 — "the placer" |
|---|---|---|
| Runs | `server/main.py` | `client/main.py` |
| Camera watches | Player A's whole body | Player B's hands + body |
| Sends over the network | nothing | obstacle placements + shield requests |
| Reads | Player A's own pose (local, no network hop) | — |

Player A's tracking is **local** to the host laptop — there's no network round-trip between
"A just jumped" and "the game sees it," since that latency directly decides whether an
obstacle is dodged. Player B's placements travel over a WebSocket, since B is a genuinely
separate machine watching a live feed of the track (not shown here, but assumed screen-shared
or otherwise visible to B).

## Gameplay

- **3 lanes** — Player A's hip position (left / center / right) sets which lane they're
  running in.
- **3 obstacle types**, each needing a different move to survive:

  | Type | Looks like | Avoided by |
  |---|---|---|
  | `high` | a hazard beam hanging from the ceiling | **duck** only (jumping puts you into it) |
  | `medium` | a grey concrete wall at chest height | **jump or duck** (low enough to clear by jumping, high enough to clear by ducking) |
  | `low` | a striped ground hurdle | **jump** only (ducking puts you into it) |

- **Coins** spawn in Subway-Surfers-style trails — short taps, long straight runs, and
  zigzags that walk back and forth across lanes — and bank toward a shield.
- **Shield**: costs 100 coins to activate, lasts a fixed 2 seconds once triggered, and breaks
  *any* collision while it's up. It isn't consumed by absorbing a hit — only the timer ends
  it, so it can block multiple obstacles in a row if they arrive close together. It can be
  triggered two ways: Player B sending a shield-activation message, or Player A raising both
  arms above shoulder height (a guard pose) — whichever happens first.
- Score climbs the longer you survive, plus a bonus per coin. A session-best score is tracked
  live and called out with a "NEW BEST!" banner on death.
- Collision detection is exact and rule-based, not physics: each obstacle is checked once, on
  the precise frame it reaches the player, against a lookup table of lane + action + obstacle
  type. See [Architecture notes](#architecture-notes) for why.

## Controls

**Player A (the runner)**

| Action | How |
|---|---|
| Change lane | Step left / right in front of the camera |
| Jump | Jump |
| Duck | Duck / crouch |
| Shield | Raise both arms above shoulder height (a guard pose) |
| *(no camera available)* | Arrow keys / A-D to move, Space to jump, Down/S to duck |

**Player B (the placer)**

| Action | How |
|---|---|
| Grab an obstacle | Close a fist inside the top strip of the camera frame (the GRAB zone) |
| Choose its type | Carry it — fist still closed — down into one of the three zones below: left/center/right = low/medium/high |
| Place it | Open your hand while inside a zone |

Both `client/main.py` and `server/main.py` also support a manual keyboard fallback (see each
file's own docstring for the exact keys) for testing without reliable tracking, or as a live
backup if a camera acts up mid-demo.

**In the game window itself**

| Key | Action |
|---|---|
| `Enter` / `Space`, or click PLAY | Dismiss the title screen and start the countdown |
| `E` | Activate the shield manually (same 100-coin cost and rules as a real trigger) |
| `R` | Restart at any time, not just after a game over |
| `Esc` / `Q` | Quit |

## Project structure

```
RivalRuns/
├── server/                    Laptop 1 — the game host
│   ├── main.py                 entrypoint: wires the websocket server, Player A's camera
│   │                            thread, and the game loop together
│   ├── game.py                  the game itself: Ursina 3D scene, collision rules, HUD,
│   │                            scoring, title screen, countdown
│   ├── actors3d.py               procedurally-built 3D meshes (player rig, obstacles, coins)
│   │                            — no external model/texture files
│   ├── game_state.py             thread-safe state shared between the camera thread, the
│   │                            websocket thread, and the render loop
│   ├── websocket_server.py       accepts Player B's connection, routes incoming messages by
│   │                            shape (obstacle placement vs. shield request)
│   ├── player_a_tracking.py      Player A's pose tracker: lane, jump/duck, block/shield pose
│   ├── jump_duck_detector.py     the jump/duck motion-detection algorithm, tuned from
│   │                            recorded motion data
│   └── record_motion_data.py     dev tool used to capture that motion data
├── client/                    Laptop 2 — the obstacle placer
│   ├── main.py                  entrypoint: camera loop, websocket client, keyboard fallback
│   ├── player_b_tracker.py       hand + pose tracking, grab/drop gesture logic
│   └── websocket_client.py       auto-reconnecting websocket client
├── models/                    MediaPipe model files, checked in — no separate download step
│   ├── pose_landmarker_lite.task
│   └── hand_landmarker.task
└── requirements.txt
```

## Setup

Both laptops need:

- **Python 3.10, 3.11, or 3.12** — not 3.13+, MediaPipe doesn't reliably support it yet.
  Check what you have with `py -0p`.
- A working webcam.

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> Don't separately `pip install opencv-python` alongside the `opencv-contrib-python` that's
> already in `requirements.txt` — they conflict and break camera access.

## Running it

**Laptop 1 (host / Player A):**

```bash
cd server
python main.py
```

Goes fullscreen at native resolution by default — the correct behavior for a monitor
physically rotated to portrait via Windows Display Settings, since fullscreen then just picks
up the already-swapped resolution automatically.

| Flag | Effect |
|---|---|
| `--camera N` | Use camera index N if the default opens the wrong device |
| `--no-camera` | Skip Player A's webcam; use keyboard control instead (for testing) |
| `--windowed` | Disable fullscreen |
| `--portrait` | Windowed 1080×1920 preview of the portrait layout, without physically rotating a monitor |
| `--port N` | WebSocket port (default `8765`) |

**Laptop 2 (Player B):**

```bash
cd client
python main.py <HOST_IP> [--port 8765] [--camera N]
```

`<HOST_IP>` is Laptop 1's IP address on the same network.

## Architecture notes

A few decisions worth knowing about if you're picking this codebase back up:

- **Collision detection is deterministic, not physics-based.** Each obstacle tracks its own
  distance to the player; the exact frame it crosses zero is checked once, against the
  lookup table above (lane match + action vs. obstacle type) — not a mesh-overlap test. This
  was a deliberate choice: physics/collider-based detection on procedurally-scaled 3D meshes
  is prone to jitter and near-miss ambiguity, where "the hit rules need to be perfect" matters
  more than realistic physics.
- **Ursina entities aren't automatically lit** just because a light exists in the scene —
  every lit surface in `actors3d.py`/`game.py` explicitly uses `lit_with_shadows_shader`, or
  Panda3D's fixed-function pipeline silently overrides its color with a flat white material
  (this is exactly what caused an all-white scene during development).
- **The WebSocket protocol is one-way and fire-and-forget.** Player B's client only ever
  sends; the host never replies, there's no acknowledgment, and B's client drops queued
  messages if the connection has been down too long. That's an acceptable trade for obstacle
  placements and shield requests — a dropped one is recoverable, a blocked game loop is not.
- **Player A's camera thread writes directly into shared state** (`game_state.py`) rather
  than going over a network, specifically because that input is latency-sensitive in a way
  Player B's placements aren't.

## Known gaps

- Player B's client has no gesture wired up yet to send a shield-activation request — the
  protocol and server-side game logic are ready
  (`{"player": "B", "action": "shield"}` over the existing WebSocket connection), but nothing
  in `client/player_b_tracker.py` sends it yet. Until it is, Player A can still trigger a
  shield with the block pose, or the host can use the `E` key as a manual stand-in during a
  demo.
- The session-best score lives in memory only — it resets when the host process restarts
  (not on an in-session `R` restart), since it isn't written to disk anywhere.

## Team

- **Kevin** — Player A tracking (pose, jump/duck detection, block/shield pose)
- **Eesha** — Player B tracking (hand-gesture grab-and-drop, lane detection)
- **Lohitashwa** — game logic, 3D rendering/scene, networking glue, UI
