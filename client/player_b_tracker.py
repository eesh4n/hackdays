"""
Player B tracker: obstacle placer.

Camera is split into thirds (left/center/right = lane 0/1/2) based on the
player's hip-center x-position -- same lean/lateral-position signal
validated in hackathon-prep/test_lean_detection.py, just used for
zone-select instead of raw lean angle since a lane needs a discrete pick,
not a continuous value.

The two hands have separate jobs, so the program never has to guess whether
a hand is selecting a height or starting a punch:

  LEFT hand  -- selects the obstacle type by its height:
      above head level                          -> HIGH obstacle
      shoulder/torso level (head to waist)      -> MEDIUM obstacle
      below waist level                         -> LOW obstacle
  RIGHT hand -- throws the punch that actually places the obstacle.

Lane and obstacle type are both just continuously-tracked *state* -- they
don't fire anything by themselves. Placement fires only on a PUNCH gesture,
detected from the punching wrist's own recent path (not its distance from
the shoulder -- that metric mostly picked up shoulder flexion/raising the
arm, not an actual punch): a short window of positions is checked for
enough net displacement, covered fast enough, and in a straight-enough
line to look like a real thrown punch rather than jitter or a reposition.

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

# Hand roles. main.py mirrors the frame (cv2.flip) so the preview reads like
# a mirror to the player -- and MediaPipe labels landmarks from what it sees,
# so on a mirrored frame its LEFT_* landmarks are the player's physical RIGHT
# side. These two constants are named for the PLAYER's physical hands; if the
# roles come out reversed (e.g. on a setup that doesn't mirror the frame),
# swap the two values.
PUNCH_WRIST = LEFT_WRIST     # player's physical RIGHT hand -- throws the punch
HEIGHT_WRIST = RIGHT_WRIST   # player's physical LEFT hand -- picks obstacle height

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

# Punch-gesture detection: tracks the right wrist's own recent path (a
# rolling window of its last PUNCH_PATH_WINDOW_SEC seconds of positions),
# not its distance from the shoulder -- distance-from-shoulder mostly
# reacted to shoulder flexion/extension (raising the arm), not the wrist
# actually being thrown forward.
#
# A punch is recognized from three things measured over that window:
#   - net displacement (start-to-end distance) is large enough
#   - it happened fast enough (highest single-frame speed within the window)
#   - the path is reasonably straight (net displacement / total path length
#     close to 1.0), so a fast but wandering/circular hand motion doesn't
#     count -- a thrown punch is a fairly direct line.
#
# Debounced by a plain time cooldown (not by requiring any specific
# resting position), so it can't get stuck waiting for a pose that never
# happens. If punches still aren't registering, check the live
# "disp/speed/straight" numbers in the debug overlay against these
# constants to see which condition is failing.
# Thresholds below are calibrated from a recorded session of real punches
# (see calibrate_punch.py). Measured punches landed at disp 0.116-0.198,
# speed 0.65-3.41, straightness 0.81-1.00, while idle hand jitter stayed
# under disp 0.076 -- so displacement is the cleanest separator and these
# sit in the gap between the two clusters. The straightness floor also
# rejects MediaPipe landmark glitches, which show huge speed (10+) but
# near-zero straightness since the wrist "jumps" rather than travels.
PUNCH_PATH_WINDOW_SEC = 0.35       # how far back to look for a punch trajectory
PUNCH_MIN_NET_DISPLACEMENT = 0.10  # normalized distance, start-to-end of the window
PUNCH_MIN_PEAK_SPEED = 0.60        # normalized units/sec, fastest single-frame segment in the window
PUNCH_MIN_STRAIGHTNESS = 0.70      # net displacement / total path length (1.0 = perfectly straight)
PUNCH_COOLDOWN_SEC = 0.5           # minimum time between two accepted placements


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

        # Rolling window of the right wrist's recent (t, x, y) positions,
        # used to detect a punch from its actual path. Debounced by a
        # plain time cooldown, see constants above.
        self._wrist_path = deque()
        self._last_net_displacement = 0.0
        self._last_peak_speed = 0.0
        self._last_straightness = 0.0
        self._last_punch_time = 0.0
        self._punch_wrist_xy = None
        self._height_wrist_xy = None

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
            # Kept for the debug overlay so you can see which physical hand
            # the code has bound to which role.
            self._punch_wrist_xy = (lm[PUNCH_WRIST].x, lm[PUNCH_WRIST].y)
            self._height_wrist_xy = (lm[HEIGHT_WRIST].x, lm[HEIGHT_WRIST].y)
        else:
            self.pose_visible = False
            # Lost tracking -- clear the path so the next detected frame
            # doesn't treat the gap as part of one continuous motion.
            self._wrist_path.clear()

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
        wrist_y = lm[HEIGHT_WRIST].y  # left hand only -- punching hand is ignored here

        if wrist_y <= head_y + HEIGHT_MARGIN:
            return OBSTACLE_HIGH
        elif wrist_y >= waist_y - HEIGHT_MARGIN:
            return OBSTACLE_LOW
        else:
            return OBSTACLE_MEDIUM

    def _update_punch_state(self, lm, now):
        # Right hand only -- the left hand's height selection can't trigger this.
        wrist = lm[PUNCH_WRIST]
        self._wrist_path.append((now, wrist.x, wrist.y))
        while self._wrist_path and now - self._wrist_path[0][0] > PUNCH_PATH_WINDOW_SEC:
            self._wrist_path.popleft()

        pts = list(self._wrist_path)
        path_length = 0.0
        peak_speed = 0.0
        for (t0, x0, y0), (t1, x1, y1) in zip(pts, pts[1:]):
            seg = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            path_length += seg
            dt = t1 - t0
            if dt > 0:
                peak_speed = max(peak_speed, seg / dt)

        if len(pts) >= 2:
            (t0, x0, y0) = pts[0]
            (t1, x1, y1) = pts[-1]
            net_displacement = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        else:
            net_displacement = 0.0

        straightness = (net_displacement / path_length) if path_length > 0 else 0.0

        self._last_net_displacement = net_displacement
        self._last_peak_speed = peak_speed
        self._last_straightness = straightness

        past_cooldown = (now - self._last_punch_time) >= PUNCH_COOLDOWN_SEC
        if (
            past_cooldown
            and net_displacement >= PUNCH_MIN_NET_DISPLACEMENT
            and peak_speed >= PUNCH_MIN_PEAK_SPEED
            and straightness >= PUNCH_MIN_STRAIGHTNESS
        ):
            self._last_punch_time = now
            self._wrist_path.clear()  # don't let the same motion double-fire
            return {"lane": self.lane, "obstacle_type": self.obstacle_type}

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
        cv2.putText(frame, f"obstacle: {self.obstacle_type} (L hand)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, lane_color, 2)

        cooldown_remaining = max(0.0, PUNCH_COOLDOWN_SEC - (time.time() - self._last_punch_time))
        ready = cooldown_remaining <= 0
        state_color = (0, 255, 0) if ready else (0, 165, 255)
        state_text = "ready" if ready else f"cooldown {cooldown_remaining:.1f}s"
        cv2.putText(frame, f"punch (R hand): {state_text}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        cv2.putText(
            frame,
            f"disp: {self._last_net_displacement:.3f} (need {PUNCH_MIN_NET_DISPLACEMENT})  "
            f"speed: {self._last_peak_speed:.2f} (need {PUNCH_MIN_PEAK_SPEED})",
            (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1,
        )
        cv2.putText(
            frame,
            f"straight: {self._last_straightness:.2f} (need {PUNCH_MIN_STRAIGHTNESS})",
            (10, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1,
        )

        # Mark both tracked wrists so it's obvious at a glance which physical
        # hand is bound to which role -- swap PUNCH_WRIST/HEIGHT_WRIST if
        # these labels land on the wrong hands.
        if self._height_wrist_xy is not None:
            hx, hy = self._height_wrist_xy
            cv2.circle(frame, (int(hx * w), int(hy * h)), 10, (255, 180, 0), -1)
            cv2.putText(frame, "HEIGHT", (int(hx * w) + 14, int(hy * h) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)
        if self._punch_wrist_xy is not None:
            px, py = self._punch_wrist_xy
            cv2.circle(frame, (int(px * w), int(py * h)), 10, (0, 100, 255), -1)
            cv2.putText(frame, "PUNCH", (int(px * w) + 14, int(py * h) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
        return frame
