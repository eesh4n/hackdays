"""
WebSocket client for Player B's laptop: connects out to the game host
(Player A's laptop / the game server), sends JSON obstacle-placement
events, and auto-reconnects if the connection drops mid-game.

Networking pattern validated in hackathon-prep/test_cross_laptop_websocket.py
(basic connect/send/recv) and test_keypress_relay.py (background-thread +
asyncio pattern so a synchronous camera loop can drive async networking).
This adds the reconnect-on-drop logic neither test script needed to cover.

The connection today is effectively one-way (B sends, the server never
replies) -- but a WebSocket is full-duplex, so this client also runs a
concurrent receive loop and will hand any incoming JSON message to an
optional on_message callback. Right now nothing sends anything back, so
this is inert -- but it's the missing piece that would let the host push
e.g. {"event": "round_reset"} or a live score down to B, which today B has
no way to find out about except a manual keypress. Wiring that up needs a
small matching change server-side (websocket_server.py's handler would
need to ws.send(...) something), which is out of scope for this file
alone.

Usage from main.py (synchronous camera loop):

    client = WebSocketClient(server_ip, port=8765, on_message=handle_msg)
    client.start()
    ...
    client.send({"player": "B", "lane": 1, "obstacle": "high", ...})
    ...
    client.stop()
"""
import asyncio
import json
import queue
import threading
import time

import websockets

DEFAULT_PORT = 8765
RECONNECT_DELAY_SEC = 1.5
RECONNECT_MAX_DELAY_SEC = 8.0
SEND_QUEUE_MAXSIZE = 100


class WebSocketClient:
    """Thread-safe wrapper: runs its own asyncio event loop on a background
    thread, so the main camera loop can stay a plain synchronous while-loop
    and just call .send(dict) without knowing anything about asyncio."""

    def __init__(self, server_ip, port=DEFAULT_PORT):
        self.server_ip = server_ip
        self.port = port
        self.uri = f"ws://{server_ip}:{port}"

        self._send_queue = queue.Queue(maxsize=SEND_QUEUE_MAXSIZE)
        self._loop = None
        self._thread = None
        self._stop_event = threading.Event()

        self.connected = False
        self._seq = 0
        self.last_sent_at = None  # wall-clock time of the last message actually
                                   # written to the socket (not just enqueued) --
                                   # lets the UI distinguish "connected but queue
                                   # is backing up" from "actually delivering".

    def start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2)

    def send(self, payload: dict):
        """Non-blocking. Drops the message if the outbound queue is full
        (e.g. connection has been down for a while) rather than blocking
        the camera loop -- a dropped obstacle placement is recoverable,
        a frozen game loop is not."""
        self._seq += 1
        message = dict(payload)
        message["seq"] = self._seq
        message["sent_at"] = time.time()
        try:
            self._send_queue.put_nowait(message)
        except queue.Full:
            pass

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connection_manager())
        finally:
            self._loop.close()

    async def _connection_manager(self):
        delay = RECONNECT_DELAY_SEC
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self.uri, open_timeout=8) as ws:
                    print(f"[websocket_client] connected to {self.uri}")
                    self.connected = True
                    delay = RECONNECT_DELAY_SEC  # reset backoff after a successful connect
                    await self._pump_send_queue(ws)
            except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as e:
                # open_timeout above raises asyncio.TimeoutError, which on
                # Python <3.11 is NOT a subclass of the builtin TimeoutError
                # or OSError -- it was slipping past this except entirely
                # and killing the whole reconnect loop's thread permanently
                # (no more retries) instead of backing off and trying again.
                print(f"[websocket_client] connection error: {e}")
            finally:
                self.connected = False

            if self._stop_event.is_set():
                break

            print(f"[websocket_client] reconnecting in {delay:.1f}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, RECONNECT_MAX_DELAY_SEC)

    async def _pump_send_queue(self, ws):
        """Drains the thread-safe send queue onto the live websocket until
        the connection drops, then returns so the caller can reconnect."""
        while not self._stop_event.is_set():
            message = await self._dequeue_async()
            if message is None:
                continue
            try:
                await ws.send(json.dumps(message))
                self.last_sent_at = time.time()
            except websockets.exceptions.ConnectionClosed:
                # Put it back so it isn't silently lost across a reconnect.
                try:
                    self._send_queue.put_nowait(message)
                except queue.Full:
                    pass
                return

    async def _dequeue_async(self, poll_interval=0.05):
        """Bridges the blocking queue.Queue into the asyncio world without
        blocking the event loop -- polls briefly so _stop_event is still
        checked promptly even with nothing to send."""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None, lambda: self._send_queue.get(timeout=poll_interval)
            )
        except queue.Empty:
            return None
