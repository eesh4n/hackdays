"""
PLAYER A TRACKER (lane + jump/duck)
=====================================

PlayerATracker: feed frames in with process_frame(), poll .lane /
.action / .pose_visible for live state each frame. This is the class
server/main.py imports for the real game -- keep its process_frame()
return shape ({"pose_visible", "lane", "action"}) in sync with
player_a_tracker.py's if that ever changes.

Lane: camera split into thirds by hip-center x-position.
Action ("run"/"jump"/"duck"): JumpDuckDetector (see jump_duck_detector.py)
tracks hip_y relative to a self-calibrating baseline, normalized by the
person's own torso length -- thresholds were derived from recorded
motion data (see record_motion_data.py, motion_log*.csv). detector.state
("neutral"/"jump"/"duck") maps directly to the action returned here, so
jump/duck persists for the whole motion rather than firing once.

Running this file directly (`python player_a_tracking.py`) opens a
standalone fullscreen debug window -- useful for testing the tracker in
isolation without running the whole game.
"""

import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from jump_duck_detector import JumpDuckDetector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "pose_landmarker_lite.task")

LANE_COUNT = 3
LANE_NAMES = ["LEFT", "CENTER", "RIGHT"]

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24

DETECTOR_STATE_TO_ACTION = {"neutral": "run", "jump": "jump", "duck": "duck"}

# Block/shield: both wrists raised above shoulder height (a guard pose),
# checked fresh every frame rather than tracked as a motion like jump/duck
# -- it's a held pose, not a displacement to measure against a baseline.
# Hysteresis (enter margin > exit margin) keeps it from flickering at the
# boundary the same way lane switching does elsewhere in this file.
BLOCK_ENTER_MARGIN = 0.05  # normalized y-units wrists must clear above shoulder_y to start blocking
BLOCK_EXIT_MARGIN = 0.02   # margin required to drop back out of block once already blocking

# A subset of BlazePose's 33 connections, enough to see the skeleton clearly.
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),   # torso
    (11, 13), (13, 15),                        # left arm
    (12, 14), (14, 16),                        # right arm
    (23, 25), (25, 27), (27, 29), (27, 31),    # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),    # right leg
]


def draw_lanes(frame, lane_width, lane_count, current_lane):
    h, w = frame.shape[:2]

    # Highlight the active lane.
    if current_lane is not None:
        overlay = frame.copy()
        x0 = current_lane * lane_width
        x1 = w if current_lane == lane_count - 1 else x0 + lane_width
        cv2.rectangle(overlay, (x0, 0), (x1, h), (0, 255, 136), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Vertical divider lines between lanes.
    for i in range(1, lane_count):
        x = i * lane_width
        cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 2)

    # Lane labels.
    for i in range(lane_count):
        x0 = i * lane_width
        label = LANE_NAMES[i] if i < len(LANE_NAMES) else f"LANE {i + 1}"
        color = (0, 255, 136) if i == current_lane else (200, 200, 200)
        cv2.putText(
            frame, label, (x0 + 20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA
        )


class PlayerATracker:
    """Feed frames in with process_frame(); poll .lane / .action /
    .pose_visible for live state each frame."""

    def __init__(self):
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at: {MODEL_PATH}\n"
                "Download it from:\n"
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task\n"
                f"and save it in: {SCRIPT_DIR}"
            )

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._start_time_ms = int(time.time() * 1000)
        self._last_timestamp_ms = -1

        self._detector = JumpDuckDetector()
        self._last_landmarks = None
        self._is_blocking = False

        self.lane = 1
        self.action = "run"
        self.pose_visible = False

    def close(self):
        self._landmarker.close()

    def process_frame(self, rgb_frame):
        """rgb_frame: HxWx3 RGB numpy array (already flipped/converted by caller).

        Returns: {"pose_visible": bool, "lane": 0|1|2, "action": "run"|"jump"|"duck"|"block",
                  "delta": float, "jump_threshold": float, "duck_threshold": float}
        `delta` is the live jump/duck signal (see jump_duck_detector.py) --
        negative crossing -jump_threshold fires a jump, positive crossing
        duck_threshold fires a duck. Exposed so callers (main.py's debug
        window, game.py's HUD) can show players the actual numbers driving
        detection, not just the resulting action.
        """
        _, w = rgb_frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000) - self._start_time_ms
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            if not self.pose_visible:
                # Someone just arrived at the screen (first detection ever,
                # or stepped back into frame after leaving) -- re-arm the
                # detector so delta starts at 0 for them instead of being
                # measured against a stale baseline from before they left.
                self._detector.reset()
                self._is_blocking = False
            self.pose_visible = True
            landmarks = result.pose_landmarks[0]  # first detected person
            self._last_landmarks = landmarks

            lane_width = w / LANE_COUNT
            hip_x = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 2 * w
            self.lane = min(int(hip_x // lane_width), LANE_COUNT - 1)

            hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
            shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
            torso_len = hip_y - shoulder_y

            self._detector.update(hip_y, torso_len)
            detector_action = DETECTOR_STATE_TO_ACTION[self._detector.state]

            # Block takes priority: both wrists raised above shoulder height
            # (a guard/shield pose) overrides whatever jump/duck landed on.
            wrist_y = (landmarks[LEFT_WRIST].y + landmarks[RIGHT_WRIST].y) / 2
            margin = BLOCK_EXIT_MARGIN if self._is_blocking else BLOCK_ENTER_MARGIN
            self._is_blocking = (shoulder_y - wrist_y) > margin

            self.action = "block" if self._is_blocking else detector_action
        else:
            self.pose_visible = False
            self._last_landmarks = None

        return {
            "pose_visible": self.pose_visible,
            "lane": self.lane,
            "action": self.action,
            "delta": self._detector.last_normalized_delta,
            "jump_threshold": self._detector.jump_threshold,
            "duck_threshold": self._detector.duck_threshold,
        }

    def debug_overlay(self, frame):
        """Draws skeleton, lane dividers, and current lane/action onto a
        BGR frame in place. For the manual test window."""
        h, w = frame.shape[:2]
        lane_width = w // LANE_COUNT

        if self._last_landmarks is not None:
            points = [(int(p.x * w), int(p.y * h)) for p in self._last_landmarks]
            for a, b in POSE_CONNECTIONS:
                cv2.line(frame, points[a], points[b], (0, 255, 136), 2)
            for x, y in points:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        draw_lanes(frame, lane_width, LANE_COUNT, self.lane if self.pose_visible else None)

        if not self.pose_visible:
            cv2.putText(
                frame, "No person detected", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
            )

        action_color = {
            "jump": (0, 200, 255), "duck": (255, 100, 0),
            "block": (255, 0, 200), "run": (0, 255, 136),
        }[self.action]
        cv2.putText(
            frame, f"action: {self.action}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, action_color, 2, cv2.LINE_AA
        )
        delta = self._detector.last_normalized_delta
        cv2.putText(
            frame,
            f"delta: {delta:+.2f}  (jump < -{self._detector.jump_threshold:.2f}, "
            f"duck > {self._detector.duck_threshold:.2f})",
            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA
        )
        return frame


def main():
    """Standalone fullscreen debug window -- run this file directly to
    test the tracker in isolation (no game/websocket involved)."""
    tracker = PlayerATracker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    window_name = "Player A Tracker (standalone debug)"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracker.process_frame(rgb)
            tracker.debug_overlay(frame)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
