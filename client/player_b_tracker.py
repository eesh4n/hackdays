"""
Player B tracker: obstacle placer.

Camera is split into thirds (left/center/right = lane 0/1/2) based on the
player's hip-center x-position -- same lean/lateral-position signal
validated in hackathon-prep/test_lean_detection.py, just used for
zone-select instead of raw lean angle since a lane needs a discrete pick,
not a continuous value.

Within whichever lane the player is standing in, hand height selects the
obstacle type:
  - hands above head level          -> HIGH obstacle
  - hands at shoulder/torso level (between head and waist) -> MEDIUM obstacle
  - hands below waist level         -> LOW obstacle

Lane and obstacle type are both just continuously-tracked *state* -- they
don't fire anything by themselves. Placement fires only on a PUNCH gesture
(a fast forward/outward extension of a wrist away from the shoulder), so
standing there with your hands up doesn't spam obstacles every frame.

Uses the MediaPipe Tasks PoseLandmarker API (mp.solutions.pose is gone as
of mediapipe 0.10.35 -- see hackathon-prep/PLAN.md finding #1). Needs
models/pose_landmarker_lite.task alongside the project (see MODEL_PATH).
"""
import time
from collections import deque
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"

# Pose landmark indices (MediaPipe Pose topology)
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26

LANE_NAMES = ("left", "center", "right")
OBSTACLE_HIGH = "high"
OBSTACLE_MEDIUM = "medium"
OBSTACLE_LOW = "low"

# Lane boundaries as fraction of frame width, with a hysteresis band so
# standing near a boundary doesn't flicker between two lanes every frame.
LANE_BOUNDARY_1 = 1 / 3
LANE_BOUNDARY_2 = 2 / 3
LANE_HYSTERESIS = 0.035  # widen/narrow the dead zone around each boundary

# Obstacle-height boundaries: HIGH is above head (nose) level, LOW is below
# waist (hip) level, and everything in between -- including shoulder height
# -- is MEDIUM. A small margin keeps a wrist sitting exactly at head/waist
# level from flip-flopping between two obstacle types every frame.
HEIGHT_MARGIN = 0.02  # normalized y-units

# Punch-gesture detection: a wrist's distance from its shoulder (normalized
# 0-1 coords) has to grow faster than PUNCH_VELOCITY_THRESHOLD (units/sec)
# while already past PUNCH_MIN_EXTENSION, to count as a punch-out. The
# tracker re-arms only once the wrist retracts back below the same
# extension threshold, so one punch can't double-fire while the arm is
# still out.
PUNCH_MIN_EXTENSION = 0.18       # normalized distance, roughly "arm mostly extended"
PUNCH_VELOCITY_THRESHOLD = 1.4   # normalized-distance units per second
PUNCH_COOLDOWN_SEC = 0.5         # minimum time between two accepted placements
VELOCITY_SMOOTHING_WINDOW = 3    # frames averaged for velocity, to fight jitter


class PlayerBTracker:
    """Feed frames in with process_frame(); poll .lane / .obstacle_type for
    live state each frame, and check the returned event for a placement."""

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Pose model not found at {MODEL_PATH} -- copy "
                "pose_landmarker_lite.task into the project's models/ folder."
            )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._start_time = time.time()

        # Live state, updated every frame a pose is detected.
        self.lane = 1              # default to center lane until first detection
        self.obstacle_type = OBSTACLE_MEDIUM
        self.pose_visible = False

        # Punch state machine: "armed" (ready to detect a punch) or
        # "extended" (arm is out, waiting for retraction before re-arming).
        self._punch_state = "armed"
        self._last_extension = 0.0
        self._extension_history = deque(maxlen=VELOCITY_SMOOTHING_WINDOW)
        self._last_frame_time = None
        self._last_punch_time = 0.0

    def close(self):
        self._landmarker.close()

    def process_frame(self, rgb_frame):
        """rgb_frame: HxWx3 RGB numpy array (already flipped/converted by caller).

        Returns a dict describing this frame's result:
            {
              "pose_visible": bool,
              "lane": 0|1|2,
              "obstacle_type": "high"|"medium"|"low",
              "event": {"lane": int, "obstacle_type": str} or None,
            }
        `event` is only non-None on the exact frame a punch placement fires.
        """
        now = time.time()
        timestamp_ms = int((now - self._start_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        event = None

        if result.pose_landmarks:
            self.pose_visible = True
            lm = result.pose_landmarks[0]

            self.lane = self._update_lane(lm)
            self.obstacle_type = self._compute_obstacle_type(lm)
            event = self._update_punch_state(lm, now)
        else:
            self.pose_visible = False
            # Lost tracking mid-extension -- don't leave the state machine
            # stuck waiting for a retraction we'll never see.
            self._punch_state = "armed"
            self._extension_history.clear()

        self._last_frame_time = now

        return {
            "pose_visible": self.pose_visible,
            "lane": self.lane,
            "obstacle_type": self.obstacle_type,
            "event": event,
        }

    def _update_lane(self, lm):
        hip_x = (lm[LEFT_HIP].x + lm[RIGHT_HIP].x) / 2

        # Hysteresis: only cross a boundary if we're clearly past it relative
        # to the lane we're currently in, not just barely over the raw line.
        b1_lo, b1_hi = LANE_BOUNDARY_1 - LANE_HYSTERESIS, LANE_BOUNDARY_1 + LANE_HYSTERESIS
        b2_lo, b2_hi = LANE_BOUNDARY_2 - LANE_HYSTERESIS, LANE_BOUNDARY_2 + LANE_HYSTERESIS

        if self.lane == 0:
            return 1 if hip_x > b1_hi else 0
        elif self.lane == 1:
            if hip_x < b1_lo:
                return 0
            if hip_x > b2_hi:
                return 2
            return 1
        else:  # self.lane == 2
            return 1 if hip_x < b2_lo else 2

    def _compute_obstacle_type(self, lm):
        head_y = lm[NOSE].y
        waist_y = (lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2
        wrist_y = min(lm[LEFT_WRIST].y, lm[RIGHT_WRIST].y)  # higher wrist wins (smaller y = higher up)

        if wrist_y <= head_y + HEIGHT_MARGIN:
            return OBSTACLE_HIGH
        elif wrist_y >= waist_y - HEIGHT_MARGIN:
            return OBSTACLE_LOW
        else:
            return OBSTACLE_MEDIUM

    def _update_punch_state(self, lm, now):
        # Track whichever arm is currently more extended from its shoulder.
        left_ext = _dist(lm[LEFT_WRIST], lm[LEFT_SHOULDER])
        right_ext = _dist(lm[RIGHT_WRIST], lm[RIGHT_SHOULDER])
        extension = max(left_ext, right_ext)

        self._extension_history.append((now, extension))

        velocity = 0.0
        if len(self._extension_history) >= 2:
            (t0, e0) = self._extension_history[0]
            (t1, e1) = self._extension_history[-1]
            dt = t1 - t0
            if dt > 0:
                velocity = (e1 - e0) / dt

        self._last_extension = extension

        if self._punch_state == "armed":
            past_cooldown = (now - self._last_punch_time) >= PUNCH_COOLDOWN_SEC
            if (
                past_cooldown
                and extension >= PUNCH_MIN_EXTENSION
                and velocity >= PUNCH_VELOCITY_THRESHOLD
            ):
                self._punch_state = "extended"
                self._last_punch_time = now
                return {"lane": self.lane, "obstacle_type": self.obstacle_type}
        else:  # "extended" -- wait for retraction before re-arming
            if extension < PUNCH_MIN_EXTENSION * 0.7:
                self._punch_state = "armed"

        return None

    def debug_overlay(self, frame):
        """Draws lane dividers, current lane/obstacle state, and punch
        readiness onto a BGR frame in place. For the manual test window."""
        import cv2

        h, w = frame.shape[:2]
        x1 = int(w * LANE_BOUNDARY_1)
        x2 = int(w * LANE_BOUNDARY_2)
        cv2.line(frame, (x1, 0), (x1, h), (80, 80, 80), 1)
        cv2.line(frame, (x2, 0), (x2, h), (80, 80, 80), 1)

        lane_color = (0, 255, 0) if self.pose_visible else (0, 0, 255)
        cv2.putText(frame, f"lane: {LANE_NAMES[self.lane]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, lane_color, 2)
        cv2.putText(frame, f"obstacle: {self.obstacle_type}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, lane_color, 2)

        state_color = (0, 255, 255) if self._punch_state == "extended" else (200, 200, 200)
        cv2.putText(frame, f"punch state: {self._punch_state}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        return frame


def _dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
