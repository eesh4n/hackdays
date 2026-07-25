"""
Entrypoint for Laptop 1 (the game host / "Player A's laptop"): wires
together the three concurrent pieces of the game --

  - the websocket server (accepts Player B's connection, feeds spawns
    into GameState)
  - the Player A camera tracker (feeds lane/action into GameState,
    direct function calls since it's co-located with the game -- no
    network hop for the latency-sensitive continuous input)
  - the pygame game loop (game.py), which only ever reads GameState

The camera tracker and websocket server each run on their own background
thread; pygame needs the main thread for its event loop, so the render
loop is what actually blocks here.

Usage:
    python main.py [--port 8765] [--camera 0] [--no-camera]

--no-camera skips opening Player A's webcam entirely and falls back to
keyboard control (LEFT/RIGHT or A/D = change lane, SPACE = jump,
DOWN or S = duck) -- useful for testing game.py and the websocket server
without a working camera or pose model on hand.
"""
import argparse
import threading

import pygame

from game_state import GameState
from websocket_server import WebSocketServer, DEFAULT_PORT
from game import Game


def parse_args():
    parser = argparse.ArgumentParser(description="Game host: server + Player A tracker + game loop")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--camera", type=int, default=0, help="camera index, try 0/1/2 if wrong device opens")
    parser.add_argument("--no-camera", action="store_true",
                         help="skip Player A's camera/pose tracking, use keyboard control instead")
    return parser.parse_args()


def run_camera_tracker(game_state, camera_index, stop_event):
    """Runs on its own thread: opens the webcam, feeds frames to
    PlayerATracker, and writes the result straight into GameState every
    frame. Imports cv2/player_a_tracking lazily so --no-camera can run on a
    machine without opencv/mediapipe/the pose model installed at all."""
    import cv2
    from player_a_tracking import PlayerATracker

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[main] FAIL: could not open camera index {camera_index}")
        print("[main]   -> try --camera 1/2, or run with --no-camera to use keyboard control")
        stop_event.set()
        return

    tracker = PlayerATracker()
    print("[main] Player A camera tracker running")

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                print("[main] FAIL: camera frame read failed")
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = tracker.process_frame(rgb)
            game_state.set_player_a(result["lane"], result["action"], result["pose_visible"])

            tracker.debug_overlay(frame)
            cv2.imshow("Player A (runner) -- debug", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


def run_with_keyboard_fallback(game, game_state):
    """Same shape as Game.run(), but also reads pygame keyboard state
    each frame and writes it into GameState as a stand-in for the camera
    tracker -- lets game.py and websocket_server.py be tested end-to-end
    with zero camera/mediapipe dependency."""
    lane = 1
    running = True
    while running:
        dt = game.clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r and game.game_over:
                    game.reset()
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    lane = max(0, lane - 1)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    lane = min(2, lane + 1)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action = "jump"
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action = "duck"
        else:
            action = "run"
        game_state.set_player_a(lane, action, pose_visible=True)

        if not game.game_over:
            game.update(dt)
        game.draw()
        pygame.display.flip()

    pygame.quit()


def main():
    args = parse_args()
    game_state = GameState()

    server = WebSocketServer(game_state, port=args.port)
    server.start()
    print(f"[main] websocket server listening on 0.0.0.0:{args.port} "
          "-- have Player B connect to this machine's IP")

    stop_event = threading.Event()
    camera_thread = None

    if not args.no_camera:
        camera_thread = threading.Thread(
            target=run_camera_tracker, args=(game_state, args.camera, stop_event), daemon=True
        )
        camera_thread.start()
    else:
        print("[main] --no-camera: Player A control is keyboard-only "
              "(LEFT/RIGHT = lane, SPACE = jump, DOWN = duck)")

    game = Game(game_state)
    try:
        if args.no_camera:
            run_with_keyboard_fallback(game, game_state)
        else:
            game.run()
    finally:
        stop_event.set()
        server.stop()
        if camera_thread is not None:
            camera_thread.join(timeout=2)


if __name__ == "__main__":
    main()
