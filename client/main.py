"""
Player B entrypoint -- runs on Laptop 2 ("the obstacle placer").

Opens the webcam, runs the lane + obstacle-height tracker, and sends a
placement event over WebSocket to the game host every time Player B
punches -- a short beep and a fading trail of the punching wrist confirm
each placement locally, independent of whether the network delivery
actually succeeds. Also draws a debug preview window (lane dividers,
current lane/obstacle state, punch-armed indicator, last-successful-send
age) so you can see what's being tracked live -- worth leaving on even
during the real demo, just moved off to the side, since it's the fastest
way to tell if tracking is misbehaving before a judge notices.

Usage:
    python main.py <server_ip> [--port 8765] [--camera 0]

Keyboard fallback (per hackathon-prep/PLAN.md resilience guidance -- keep
manual input alive as a permanent fallback in case the camera or pose
tracking dies mid-demo, not just for hour-0 testing):
    1 / 2 / 3   -- force current lane to left / center / right
    h / m / l   -- force current obstacle type to high / medium / low
    SPACE       -- manually fire a placement with whatever lane/obstacle
                   is currently selected (bypasses the punch gesture)
    f           -- toggle fullscreen
    r           -- reset punch count (cooldown scaling starts over --
                   use this at the start of each new round, since B has
                   no way to know the host restarted automatically)
    q           -- quit
"""
import argparse
import sys
import threading
import time
import winsound

import cv2

from player_b_tracker import PlayerBTracker, LANE_NAMES, OBSTACLE_HIGH, OBSTACLE_MEDIUM, OBSTACLE_LOW
from websocket_client import WebSocketClient, DEFAULT_PORT

WINDOW_NAME = "Player B (obstacle placer)"

PUNCH_BEEP_FREQ_HZ = 880
PUNCH_BEEP_DURATION_MS = 80


def _play_punch_beep():
    """winsound.Beep blocks the calling thread for its whole duration, so
    it runs on a short-lived daemon thread -- the camera loop must never
    stall waiting on a beep, even a short one, at 30fps."""
    threading.Thread(
        target=winsound.Beep, args=(PUNCH_BEEP_FREQ_HZ, PUNCH_BEEP_DURATION_MS), daemon=True
    ).start()


def parse_args():
    parser = argparse.ArgumentParser(description="Player B camera + input client")
    parser.add_argument("server_ip", help="IP address of the game host (Player A's laptop)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--camera", type=int, default=0, help="camera index, try 0/1/2 if wrong device opens")
    return parser.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"FAIL: could not open camera index {args.camera}")
        print("  -> try a different --camera index, or check Windows camera privacy settings")
        sys.exit(1)

    tracker = PlayerBTracker()
    client = WebSocketClient(args.server_ip, port=args.port)
    client.start()

    print(f"[main] connecting out to {args.server_ip}:{args.port} ...")
    print("[main] keyboard fallback: 1/2/3 = lane, h/m/l = obstacle type, SPACE = force-place, "
          "f = fullscreen, r = reset punch count, q = quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen = True

    # Manual override state for the keyboard fallback path. None means
    # "use whatever the camera tracker is currently reading".
    manual_lane = None
    manual_obstacle = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[main] FAIL: camera frame read failed")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = tracker.process_frame(rgb)

            effective_lane = manual_lane if manual_lane is not None else result["lane"]
            effective_obstacle = manual_obstacle if manual_obstacle is not None else result["obstacle_type"]

            if result["event"] is not None:
                client.send({
                    "player": "B",
                    "lane": effective_lane,
                    "obstacle": effective_obstacle,
                })
                _play_punch_beep()
                print(f"[main] >> placed {effective_obstacle} obstacle in lane {LANE_NAMES[effective_lane]}")

            tracker.debug_overlay(frame)
            conn_color = (0, 255, 0) if client.connected else (0, 0, 255)
            conn_text = "connected" if client.connected else "reconnecting..."
            cv2.putText(frame, conn_text, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, conn_color, 2)

            if client.last_sent_at is None:
                sent_text = "last sent: never"
                sent_color = (0, 165, 255)
            else:
                age = time.time() - client.last_sent_at
                sent_text = f"last sent: {age:.1f}s ago"
                # Stale send (queued but not actually delivered in a while)
                # is a subtler failure than a dropped connection -- flag it
                # even if `connected` still reads True.
                sent_color = (0, 255, 0) if age < 3.0 else (0, 165, 255)
            cv2.putText(frame, sent_text, (10, frame.shape[0] - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, sent_color, 1)
            if manual_lane is not None or manual_obstacle is not None:
                cv2.putText(frame, "MANUAL OVERRIDE ACTIVE", (10, frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
            elif key == ord('1'):
                manual_lane = 0
            elif key == ord('2'):
                manual_lane = 1
            elif key == ord('3'):
                manual_lane = 2
            elif key == ord('h'):
                manual_obstacle = OBSTACLE_HIGH
            elif key == ord('m'):
                manual_obstacle = OBSTACLE_MEDIUM
            elif key == ord('l'):
                manual_obstacle = OBSTACLE_LOW
            elif key == ord('r'):
                tracker.reset_punch_count()
                print("[main] punch count reset -- cooldown scaling starts over")
            elif key == ord(' '):
                client.send({
                    "player": "B",
                    "lane": effective_lane,
                    "obstacle": effective_obstacle,
                })
                _play_punch_beep()
                print(f"[main] >> (manual) placed {effective_obstacle} obstacle in lane {LANE_NAMES[effective_lane]}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        client.stop()


if __name__ == "__main__":
    main()
