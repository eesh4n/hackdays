"""
Player A tracker: the runner.

Camera is split into thirds (left/center/right = lane 0/1/2) using the
midpoint between the two shoulders on the x-axis (as opposed to Player
B's hip-center, used for B's stance-based zone select) -- shoulder
midpoint since running/jumping motion swings the hips more than the
shoulders, so it's the steadier signal for a runner. Same hysteresis-band
idea as player_b_tracker.py's lane logic, to avoid flicker at a boundary.

Vertical action ("run" / "jump" / "duck") is read off shoulder_y relative
to a rolling baseline captured while the player is presumed upright,
since there's no reliable absolute "on the ground" signal from a single
2D camera:
  - shoulder_y well ABOVE baseline (smaller y => higher in frame) -> JUMP
  - shoulder_y well BELOW baseline (larger y => lower in frame)   -> DUCK
  - otherwise                                                     -> RUN
The baseline recalibrates continuously via a slow moving average so it
adapts if standing height in frame drifts (distance from camera,
posture, etc.) -- it only drifts while the player is in "run", so a held
jump/duck doesn't slowly get absorbed into the baseline and stop firing.

Same MediaPipe Tasks PoseLandmarker API as client/player_b_tracker.py
(mp.solutions.pose is gone as of mediapipe 0.10.35); needs
models/pose_landmarker_lite.task alongside the project.
"""
import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

LANE_NAMES = ("left", "center", "right")
ACTION_RUN = "run"
ACTION_JUMP = "jump"
ACTION_DUCK = "duck"

LANE_BOUNDARY_1 = 1 / 3
LANE_BOUNDARY_2 = 2 / 3
LANE_HYSTERESIS = 0.035

BASELINE_ALPHA = 0.02       # how fast the "standing" baseline drifts (slow, on purpose)
JUMP_MARGIN = 0.045         # normalized y-units above baseline to count as a jump
DUCK_MARGIN = 0.05          # normalized y-units below baseline to count as a duck
ACTION_HYSTERESIS = 0.012   # shrinks the margin needed to *leave* an action,
                            # so a held jump/duck doesn't flicker back to "run"


class PlayerATracker:
    """Feed frames in with process_frame(); poll .lane / .action /
    .pose_visible for live state each frame."""

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

        self.lane = 1
        self.action = ACTION_RUN
        self.pose_visible = False

        self._baseline_shoulder_y = None

    def close(self):
        self._landmarker.close()

    def process_frame(self, rgb_frame):
        """rgb_frame: HxWx3 RGB numpy array (already flipped/converted by caller).

        Returns: {"pose_visible": bool, "lane": 0|1|2, "action": "run"|"jump"|"duck"}
        """
        now = time.time()
        timestamp_ms = int((now - self._start_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            self.pose_visible = True
            lm = result.pose_landmarks[0]
            self.lane = self._update_lane(lm)
            self.action = self._update_action(lm)
        else:
            self.pose_visible = False

        return {
            "pose_visible": self.pose_visible,
            "lane": self.lane,
            "action": self.action,
        }

    def _update_lane(self, lm):
        shoulder_x = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2

        b1_lo, b1_hi = LANE_BOUNDARY_1 - LANE_HYSTERESIS, LANE_BOUNDARY_1 + LANE_HYSTERESIS
        b2_lo, b2_hi = LANE_BOUNDARY_2 - LANE_HYSTERESIS, LANE_BOUNDARY_2 + LANE_HYSTERESIS

        if self.lane == 0:
            return 1 if shoulder_x > b1_hi else 0
        elif self.lane == 1:
            if shoulder_x < b1_lo:
                return 0
            if shoulder_x > b2_hi:
                return 2
            return 1
        else:
            return 1 if shoulder_x < b2_lo else 2

    def _update_action(self, lm):
        shoulder_y = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2

        if self._baseline_shoulder_y is None:
            self._baseline_shoulder_y = shoulder_y

        delta = shoulder_y - self._baseline_shoulder_y  # positive = lower in frame (crouching)

        jump_exit = JUMP_MARGIN - ACTION_HYSTERESIS
        duck_exit = DUCK_MARGIN - ACTION_HYSTERESIS

        if self.action == ACTION_JUMP:
            action = ACTION_JUMP if delta <= -jump_exit else ACTION_RUN
        elif self.action == ACTION_DUCK:
            action = ACTION_DUCK if delta >= duck_exit else ACTION_RUN
        else:
            if delta <= -JUMP_MARGIN:
                action = ACTION_JUMP
            elif delta >= DUCK_MARGIN:
                action = ACTION_DUCK
            else:
                action = ACTION_RUN

        if action == ACTION_RUN:
            self._baseline_shoulder_y += BASELINE_ALPHA * (shoulder_y - self._baseline_shoulder_y)

        return action

    def debug_overlay(self, frame):
        """Draws lane dividers and current lane/action onto a BGR frame
        in place. For the manual test window."""
        import cv2

        h, w = frame.shape[:2]
        x1 = int(w * LANE_BOUNDARY_1)
        x2 = int(w * LANE_BOUNDARY_2)
        cv2.line(frame, (x1, 0), (x1, h), (80, 80, 80), 1)
        cv2.line(frame, (x2, 0), (x2, h), (80, 80, 80), 1)

        color = (0, 255, 0) if self.pose_visible else (0, 0, 255)
        cv2.putText(frame, f"lane: {LANE_NAMES[self.lane]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"action: {self.action}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame
