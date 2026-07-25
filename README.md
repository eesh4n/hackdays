# RivalRuns

A two-player endless runner where neither player touches a keyboard or controller.

**Player A** dodges an incoming obstacle course with their whole body — lean left/right to change lanes, jump, duck, or raise both arms to block. **Player B**, on a second laptop with its own webcam, actively places those obstacles in real time by reaching into a "grab" zone on screen, closing their fist, carrying the grab down to a LOW/MEDIUM/HIGH "drop" zone, and releasing it. It's adversarial, not cooperative: B is trying to end A's run.

Rendered as a real 3D scene (Ursina/Panda3D) — chase-cam, procedural skyline, coin economy, and a shield mechanic — not a flat sprite game.

## Requirements

- Two laptops, each with a webcam, on the same network (a phone hotspot is more reliable than venue WiFi — see `hackathon-prep/PLAN.md` if present)
- Python 3.10–3.12
- `pip install -r requirements.txt`
- The `models/` folder (already committed in this repo — `pose_landmarker_lite.task` and `hand_landmarker.task`) needs to sit alongside `client/` and `server/`, i.e. at the repo root

## Running it

**Laptop 1 — the game host (Player A):**

```bash
cd server
python main.py
```

Starts the 3D game window, Player A's camera tracker, and the WebSocket server (listens on port 8765). It prints its own IP — that's what Player B connects to. Useful flags: `--camera 1` if the wrong webcam opens, `--no-camera` for keyboard-only testing (arrows/A-D to change lane, SPACE to jump, DOWN/S to duck), `--windowed` to skip fullscreen.

**Laptop 2 — the obstacle placer (Player B):**

```bash
cd client
python main.py <laptop-1-ip>
```

Opens Player B's webcam and connects out to the host. `--camera 1` if needed.

**Controls, Player A (body):** lean to change lanes, jump/duck to clear obstacles (each type needs a specific action — see `server/game.py`'s `avoided()`), raise both arms above shoulder height to activate a shield (costs 100 collected coins).

**Controls, Player B (hands):** reach into the top strip and close a fist to grab, carry it down into the LOW/MEDIUM/HIGH zone you want, open your hand to release. Both hands work independently and simultaneously — alternate hands to place faster. Keyboard fallback exists on both sides if a camera/tracking issue comes up mid-demo (see each `main.py`'s docstring for the exact keys).

## Architecture

```
server/                         (Laptop 1 — game host)
├── main.py                     entrypoint: wires together the pieces below
├── game.py                     the 3D scene, HUD, collision rules, coin/shield economy
├── actors3d.py                 procedural player rig + obstacle/coin meshes
├── game_state.py               thread-safe shared state between the camera thread,
│                                the websocket thread, and the render loop
├── player_a_tracking.py        Player A's pose tracker (lane + jump/duck/block)
├── jump_duck_detector.py       self-calibrating baseline + threshold logic for jump/duck
└── websocket_server.py         accepts Player B's connection, feeds spawn events in

client/                         (Laptop 2 — obstacle placer)
├── main.py                     entrypoint: camera loop, keyboard fallback, sound
├── player_b_tracker.py         two-handed grab/drop tracker (see below)
└── websocket_client.py         connects out to the host, auto-reconnects on drop
```

Player A's tracker runs co-located with the game (direct function calls, no network hop — latency-sensitive continuous input). Player B runs on a separate machine and only ever sends one-way "place this obstacle" messages over WebSocket; the game has no way to reply back to B today.

## Technical notes worth knowing before judging the code

- **Player B's fist detection measures finger *joint angle*, not distance from the wrist.** Distance-from-wrist looked reasonable at first but is rotation-sensitive — turning the hand sideways to the camera changed the numbers without any actual finger movement. Joint bend angle (using x/y/z, not just x/y) only depends on a finger's own two segments relative to each other, so it reads the same regardless of hand orientation.
- **Both of Player B's hands track independently and simultaneously**, each with its own lock, debounce, and cooldown — a proximity-based matching step re-associates each frame's detected hand(s) to the right tracking slot, so the two hands don't swap identities mid-gesture.
- **Distance thresholds are scaled by the player's own shoulder width**, not fixed pixel/coordinate values — a fixed threshold was too tight up close and too loose far from the camera; scaling by a stable per-person body reference fixes that regardless of camera distance.
- **Running pose and hand detection together, every frame, cost a real ~10fps** (mostly resource contention, not a hand-visibility problem). Sampling pose only every 8th frame (lane doesn't need per-frame precision) recovered it to ~27fps.
- Player B's punch-based placement mechanic (an earlier design) was fully replaced with the current grab/drop mechanic after testing showed it didn't read as an intentional-enough gesture — see git history for that iteration if useful context.
