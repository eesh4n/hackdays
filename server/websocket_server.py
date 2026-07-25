"""
WebSocket server for the game host (Laptop 1): accepts the inbound
connection from Player B's laptop (client/websocket_client.py dials out
to us) and writes every validated spawn event straight into GameState.

Mirrors the client's threaded-asyncio pattern for the same reason: the
pygame loop in game.py is a plain synchronous while-loop, so networking
needs to live on its own thread with its own event loop rather than
forcing the whole game onto asyncio.

Protocol (see client/player_b_tracker.py and client/main.py) -- two
message shapes, routed by which keys are present:
    Obstacle placement:
        {"player": "B", "lane": 0|1|2, "obstacle": "high"|"medium"|"low",
         "seq": int, "sent_at": float}
    Shield activation:
        {"player": "B", "action": "shield", "seq": int, "sent_at": float}
Fire-and-forget from B's side -- no ack is sent back, and delivery isn't
guaranteed (B's client drops messages if its send queue overflows while
disconnected). The game doesn't need to reply for any of this to work.
"""
import asyncio
import json
import threading

import websockets

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765


class WebSocketServer:
    def __init__(self, game_state, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.game_state = game_state
        self.host = host
        self.port = port

        self._loop = None
        self._thread = None
        self._stop_event = threading.Event()
        self.client_connected = False

    def start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        # _serve()'s own while-loop polls _stop_event every 0.1s and exits
        # its `async with websockets.serve(...)` block on its own -- that
        # lets the server close its sockets properly. Forcing loop.stop()
        # here instead would yank the loop out from under that cleanup
        # mid-await and throw "Event loop stopped before Future completed".
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _serve(self):
        async with websockets.serve(self._handler, self.host, self.port):
            print(f"[websocket_server] listening on {self.host}:{self.port}")
            while not self._stop_event.is_set():
                await asyncio.sleep(0.1)

    async def _handler(self, websocket):
        peer = websocket.remote_address
        print(f"[websocket_server] player B connected from {peer}")
        self.client_connected = True
        try:
            async for raw in websocket:
                try:
                    message = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(message, dict) or message.get("player") != "B":
                    continue
                if message.get("action") == "shield":
                    self.game_state.push_shield_request(message)
                else:
                    self.game_state.push_spawn_event(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.client_connected = False
            print(f"[websocket_server] player B disconnected ({peer})")
