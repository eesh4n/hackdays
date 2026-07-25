"""
Shared mutable state between the three concurrent loops running on Laptop 1:

  - the pygame render loop (game.py), which reads this every frame
  - the Player A camera thread (player_a_tracker.py via main.py), which
    writes lane/action every frame -- no network hop, so this is a plain
    threading.Lock rather than anything websocket-shaped
  - the websocket server thread (websocket_server.py), which writes a
    spawn event every time Player B's punch gesture fires

Everything here is read/written from multiple threads, so every access
goes through the same lock. Kept intentionally dumb (a struct + a queue)
so game.py doesn't need to know anything about threads, mediapipe, or
websockets -- it just calls get_player_a() and drain_spawn_events() once
per frame.
"""
import threading
import time

VALID_LANES = (0, 1, 2)
VALID_ACTIONS = ("run", "jump", "duck")
VALID_OBSTACLE_TYPES = ("high", "medium", "low")


class GameState:
    def __init__(self):
        self._lock = threading.Lock()

        # Player A's live state, written by the camera thread every frame.
        self._lane_a = 1
        self._action_a = "run"
        self._pose_visible_a = False

        # Pending obstacle placements from Player B, written by the
        # websocket server thread. game.py drains this every frame --
        # it's a queue, not a single value, since B can place faster than
        # the render loop ticks and no placement should be silently lost.
        self._spawn_queue = []

    # --- Player A -----------------------------------------------------

    def set_player_a(self, lane, action, pose_visible=True):
        if lane not in VALID_LANES or action not in VALID_ACTIONS:
            return
        with self._lock:
            self._lane_a = lane
            self._action_a = action
            self._pose_visible_a = pose_visible

    def get_player_a(self):
        with self._lock:
            return self._lane_a, self._action_a, self._pose_visible_a

    # --- Player B spawn events ------------------------------------------

    def push_spawn_event(self, message: dict):
        """Called from the websocket server with a decoded JSON message.
        Validates and normalizes untrusted network input before it ever
        touches game.py -- a malformed message just gets dropped, never
        crashes the render loop."""
        lane = message.get("lane")
        obstacle = message.get("obstacle")
        if lane not in VALID_LANES or obstacle not in VALID_OBSTACLE_TYPES:
            return False
        with self._lock:
            self._spawn_queue.append({
                "lane": lane,
                "obstacle": obstacle,
                "seq": message.get("seq"),
                "sent_at": message.get("sent_at"),
                "received_at": time.time(),
            })
        return True

    def drain_spawn_events(self):
        """Returns and clears all pending spawn events. Call once per
        render frame; each event should be turned into exactly one
        obstacle by game.py."""
        with self._lock:
            events, self._spawn_queue = self._spawn_queue, []
        return events

    def reset(self):
        with self._lock:
            self._lane_a = 1
            self._action_a = "run"
            self._pose_visible_a = False
            self._spawn_queue = []
