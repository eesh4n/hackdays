"""
Player B tracker: obstacle placer.

Two models, two different jobs, sampled at very different rates:

  POSE (sampled every POSE_SAMPLE_EVERY_N_FRAMES frames, not every frame)
    -- answers one coarse question: which third of the screen is the
    player standing in (left/center/right lane), from hip-center x
    position. Lane doesn't need per-frame precision -- a player walking
    between lanes is a slow motion compared to a hand gesture -- so
    running pose at full rate was wasted compute measured to cost real
    fps for no benefit here.

  HAND (every frame -- this is the main interaction now)
    -- the entire top strip of the frame is one generic GRAB zone (no
    type is decided yet). Close your hand into a fist anywhere in that
    strip to grab a generic object; carry it down (still closed) into
    the large lower area, which is split into three DROP zones
    (left/center/right = LOW/MEDIUM/HIGH) covering most of the remaining
    screen -- opening your hand inside one of those decides the obstacle
    type from *where you released it*, not where you grabbed it. That
    fires a placement event using the drop zone's type and whichever lane
    pose currently reads.

Measured trade-off before building this (see conversation, not re-run
here): running pose AND hand on every single frame cost real performance
(~10fps, and only 20% of frames found a hand at all -- likely resource
contention between the two models rather than a hand-visibility problem,
since sampling pose fixed both numbers at once). Hand-every-frame +
pose-sampled measured ~27fps with an 88% hand-detection rate -- close to
running just one model.

Fist detection is a landmark-geometry heuristic (no separate classifier
needed): for each of the four non-thumb fingers, the fingertip is closer
to the wrist than that finger's PIP joint when curled into a fist, and
farther away when extended. Majority vote (3 of 4) tolerates one
misdetected finger.

Uses the MediaPipe Tasks API for both PoseLandmarker and HandLandmarker
(mp.solutions.* is gone as of mediapipe 0.10.35 -- see
hackathon-prep/PLAN.md finding #1). Needs pose_landmarker_lite.task and
hand_landmarker.task in the project's models/ folder (see MODEL_PATHS).
"""
import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

# Pose landmark indices (MediaPipe Pose topology) -- only hips are used,
# to compute lane from hip-center x position.
LEFT_HIP, RIGHT_HIP = 23, 24

# Hand landmark indices (MediaPipe Hand topology, 21 points per hand).
WRIST = 0
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
# (TIP, PIP) pairs for the four fingers used in the fist heuristic -- thumb
# excluded, it doesn't fold toward the wrist the same way on a fist.
FINGER_TIP_PIP_PAIRS = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)
FIST_MIN_CURLED_FINGERS = 3  # of 4, majority vote

POSE_SAMPLE_EVERY_N_FRAMES = 5  # lane only needs occasional updates, not every frame

LANE_NAMES = ("left", "center", "right")
OBSTACLE_HIGH = "high"
OBSTACLE_MEDIUM = "medium"
OBSTACLE_LOW = "low"
OBSTACLE_TYPES = (OBSTACLE_HIGH, OBSTACLE_MEDIUM, OBSTACLE_LOW)

# Lane boundaries as fraction of frame width, with a hysteresis band so
# standing near a boundary doesn't flicker between two lanes every frame.
LANE_BOUNDARY_1 = 1 / 3
LANE_BOUNDARY_2 = 2 / 3
LANE_HYSTERESIS = 0.035  # widen/narrow the dead zone around each boundary

# GRAB zone: the entire top strip of the frame, full width, one zone --
# grabbing here doesn't decide a type, it just picks up a generic object.
GRAB_ZONE_HEIGHT_FRAC = 0.20

# DROP zones: the rest of the frame (below the grab strip), split into
# three equal-width zones left-to-right = LOW/MEDIUM/HIGH. Deliberately
# large -- these need to be easy to release into, not a thin strip like
# the grab zone, since this is where the type actually gets decided.
DROP_TYPES_LEFT_TO_RIGHT = (OBSTACLE_LOW, OBSTACLE_MEDIUM, OBSTACLE_HIGH)
DROP_ZONE_COLORS = {
    OBSTACLE_LOW: (80, 220, 80),
    OBSTACLE_MEDIUM: (60, 200, 220),
    OBSTACLE_HIGH: (60, 60, 220),
}

# Placement pacing: starts long, shortens as B places more obstacles this
# session -- a local count, not the live game score, so it needs no
# networking beyond what already exists. Decays linearly from MAX down to
# MIN, floored so B can never spam faster than MIN regardless of count.
PLACEMENT_COOLDOWN_MAX_SEC = 0.9
PLACEMENT_COOLDOWN_MIN_SEC = 0.5
PLACEMENT_COOLDOWN_DECAY_PER_PLACEMENT = 0.006


class PlayerBTracker:
    """Feed frames in with process_frame(); poll .lane / .carried_obstacle /
    .fist_closed for live state each frame, and check the returned event
    for a placement."""

    def __init__(self):
        for p in (POSE_MODEL_PATH, HAND_MODEL_PATH):
            if not p.exists():
                raise FileNotFoundError(
                    f"Model not found at {p} -- copy the matching .task file "
                    "into the project's models/ folder."
                )
        self._pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
            )
        )
        # num_hands=2 even though we only ever act on ONE -- if a second
        # hand (or anyone else's) enters frame, we need candidates to
        # choose from so we can lock onto whichever one is actually being
        # tracked, rather than only ever seeing whatever the model ranks
        # highest that frame (which can silently swap hands frame to frame).
        self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )
        self._locked_palm_xy = None  # last known position of the hand we're tracking
        self._start_time = time.time()
        self._frame_count = 0

        # Live state.
        self.lane = 1              # default to center lane until first detection
        self.pose_visible = False
        self.hand_visible = False
        self.fist_closed = False
        self.hand_xy = None        # (x, y) normalized, palm-center position of the tracked hand
        self.is_carrying = False   # holding a (still generic, type undecided) grabbed object

        self.placement_count = 0
        self._last_placement_time = 0.0

        # Brief visual "pop" at the place location, purely cosmetic.
        self._place_flash_xy = None
        self._place_flash_time = 0.0
        self._place_flash_color = (0, 255, 255)

    def close(self):
        self._pose_landmarker.close()
        self._hand_landmarker.close()

    def reset_placement_count(self):
        """Call this when a new round starts -- B has no way to know the
        host reset the game (the protocol is one-way, B sends only), so
        this has to be triggered locally instead of staying in sync
        automatically."""
        self.placement_count = 0

    @property
    def current_cooldown_sec(self):
        """Cooldown for the *next* placement, given how many have happened so far."""
        decayed = PLACEMENT_COOLDOWN_MAX_SEC - PLACEMENT_COOLDOWN_DECAY_PER_PLACEMENT * self.placement_count
        return max(PLACEMENT_COOLDOWN_MIN_SEC, decayed)

    def process_frame(self, rgb_frame):
        """rgb_frame: HxWx3 RGB numpy array (already flipped/converted by caller).

        Returns a dict describing this frame's result:
            {
              "pose_visible": bool,
              "hand_visible": bool,
              "lane": 0|1|2,
              "is_carrying": bool,
              "event": {"lane": int, "obstacle_type": str} or None,
            }
        `event` is only non-None on the exact frame a placement fires.
        """
        now = time.time()
        timestamp_ms = int((now - self._start_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._frame_count += 1

        if self._frame_count % POSE_SAMPLE_EVERY_N_FRAMES == 0:
            pose_result = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            if pose_result.pose_landmarks:
                self.pose_visible = True
                self.lane = self._update_lane(pose_result.pose_landmarks[0])
            else:
                self.pose_visible = False

        hand_result = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        event = None
        if hand_result.hand_landmarks:
            self.hand_visible = True
            hand_lm = self._select_locked_hand(hand_result.hand_landmarks)
            self.hand_xy = _palm_center(hand_lm)
            self._locked_palm_xy = self.hand_xy
            self.fist_closed = _is_fist(hand_lm)
            event = self._update_grab_place_state(now)
        else:
            self.hand_visible = False
            self.hand_xy = None
            self._locked_palm_xy = None  # hand fully lost -- next detection re-acquires fresh
            self.fist_closed = False

        return {
            "pose_visible": self.pose_visible,
            "hand_visible": self.hand_visible,
            "lane": self.lane,
            "is_carrying": self.is_carrying,
            "event": event,
        }

    def _select_locked_hand(self, all_hand_landmarks):
        """Given 1-2 detected hands this frame, picks whichever one we're
        actually tracking. If we have no prior lock (first detection, or
        just re-acquired after losing the hand entirely), take the first
        candidate. Otherwise, lock onto whichever candidate's palm is
        closest to where our tracked hand was last seen -- this is what
        stops a second hand (or someone walking through frame) from
        hijacking tracking just because the model happened to rank it
        first that frame."""
        if len(all_hand_landmarks) == 1 or self._locked_palm_xy is None:
            return all_hand_landmarks[0]

        lx, ly = self._locked_palm_xy
        best = min(
            all_hand_landmarks,
            key=lambda lm: (_palm_center(lm)[0] - lx) ** 2 + (_palm_center(lm)[1] - ly) ** 2,
        )
        return best

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

    def _in_grab_zone(self, y):
        return y <= GRAB_ZONE_HEIGHT_FRAC

    def _drop_zone_at(self, x, y):
        """Returns the obstacle type for the drop zone at frame position
        (x, y), or None if that position is still inside the grab strip
        (too high up to count as a real drop)."""
        if y <= GRAB_ZONE_HEIGHT_FRAC:
            return None
        third = int(x * 3)
        third = max(0, min(2, third))
        return DROP_TYPES_LEFT_TO_RIGHT[third]

    def _update_grab_place_state(self, now):
        """Level-triggered, not edge-triggered: the fist just needs to BE
        closed while in the grab zone -- it doesn't matter whether it
        closed before or after entering the zone, both grab. Same for
        drop. is_carrying flipping True/False is itself the one-shot
        guard against re-firing every frame the condition holds, so this
        doesn't need to track a previous-frame fist state to detect a
        precise transition anymore -- that precise-timing requirement was
        most of what made grabbing/dropping feel clunky."""
        x, y = self.hand_xy
        event = None

        if not self.is_carrying:
            if self.fist_closed and self._in_grab_zone(y):
                self.is_carrying = True
        else:
            if not self.fist_closed:
                drop_type = self._drop_zone_at(x, y)
                past_cooldown = (now - self._last_placement_time) >= self.current_cooldown_sec
                if drop_type is not None and past_cooldown:
                    event = {"lane": self.lane, "obstacle_type": drop_type}
                    self._last_placement_time = now
                    self.placement_count += 1
                    self._place_flash_xy = (x, y)
                    self._place_flash_time = now
                    self._place_flash_color = DROP_ZONE_COLORS[drop_type]
                # Whether it landed in a valid drop zone, was still on
                # cooldown, or opened too early back in the grab strip --
                # the hand is empty again either way. Don't leave it
                # "stuck" carrying; the player can just grab again.
                self.is_carrying = False

        return event

    def debug_overlay(self, frame):
        """Draws the grab strip, the three drop zones, hand/fist state,
        and a placement flash onto a BGR frame in place. Lane (from body
        position, not hand position) is shown as text only -- it's a
        different left/center/right axis than the drop zones and drawing
        both as vertical dividers in the same space would be genuinely
        confusing to look at. For the manual test window."""
        import cv2

        h, w = frame.shape[:2]
        grab_h = int(h * GRAB_ZONE_HEIGHT_FRAC)

        # Grab strip: one zone, full width, no type.
        grab_color = (200, 200, 0) if self.is_carrying else (120, 120, 120)
        cv2.rectangle(frame, (4, 4), (w - 4, grab_h - 4), grab_color, 3)
        cv2.putText(frame, "GRAB HERE", (w // 2 - 70, grab_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, grab_color, 2)

        # Drop zones: large, cover the rest of the frame.
        for i, obstacle_type in enumerate(DROP_TYPES_LEFT_TO_RIGHT):
            x0, x1 = int(w * i / 3), int(w * (i + 1) / 3)
            color = DROP_ZONE_COLORS[obstacle_type]
            cv2.rectangle(frame, (x0 + 4, grab_h + 4), (x1 - 4, h - 4), color, 3)
            cv2.putText(frame, obstacle_type.upper(), (x0 + 20, grab_h + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if i > 0:
                cv2.line(frame, (x0, grab_h), (x0, h), (80, 80, 80), 1)

        lane_color = (0, 255, 0) if self.pose_visible else (0, 0, 255)
        cv2.putText(frame, f"lane (body position): {LANE_NAMES[self.lane]}", (10, h - 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, lane_color, 2)

        carried_text = "YES (find a drop zone)" if self.is_carrying else "no (grab in the top strip)"
        cv2.putText(frame, f"carrying: {carried_text}", (10, h - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        cooldown = self.current_cooldown_sec
        cooldown_remaining = max(0.0, cooldown - (time.time() - self._last_placement_time))
        ready = cooldown_remaining <= 0
        state_color = (0, 255, 0) if ready else (0, 165, 255)
        state_text = "ready" if ready else f"cooldown {cooldown_remaining:.1f}s"
        cv2.putText(frame, f"placement: {state_text}  [#{self.placement_count}, cd={cooldown:.2f}s]",
                    (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

        hand_color = (0, 0, 255) if not self.hand_visible else ((0, 165, 255) if self.fist_closed else (0, 255, 0))
        hand_state = "no hand" if not self.hand_visible else ("FIST" if self.fist_closed else "open")
        cv2.putText(frame, f"hand: {hand_state}", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)

        if self.hand_xy is not None:
            hx, hy = self.hand_xy
            marker_color = (0, 165, 255) if self.fist_closed else (0, 255, 0)
            cv2.circle(frame, (int(hx * w), int(hy * h)), 14, marker_color, 3 if self.fist_closed else 2)

        # Brief flash where a placement just fired, colored to match the
        # drop zone's type -- confirms visually which type actually fired,
        # not just that something did.
        flash_age = time.time() - self._place_flash_time
        if self._place_flash_xy is not None and flash_age < 0.3:
            fx, fy = self._place_flash_xy
            radius = int(20 + flash_age * 120)
            cv2.circle(frame, (int(fx * w), int(fy * h)), radius, self._place_flash_color, 2)

        return frame


def _dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


# Average of the wrist and the four finger base-knuckles -- a much more
# central, stable point than the wrist alone, which sits at the very edge
# of the hand near the arm. Grabbing/dropping felt clunky partly because
# zone checks were being done against the wrist's position rather than
# where the hand (and the fingers actually doing the grabbing) really are.
_PALM_LANDMARKS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


def _palm_center(hand_lm):
    xs = [hand_lm[i].x for i in _PALM_LANDMARKS]
    ys = [hand_lm[i].y for i in _PALM_LANDMARKS]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _is_fist(hand_lm):
    """A finger is curled if its tip sits closer to the wrist than its own
    PIP joint does -- true when folded into a fist, false when extended.
    Majority vote across the four non-thumb fingers tolerates one
    misdetected landmark without flipping the whole classification."""
    wrist = hand_lm[WRIST]
    curled = 0
    for tip_idx, pip_idx in FINGER_TIP_PIP_PAIRS:
        tip_dist = _dist(hand_lm[tip_idx], wrist)
        pip_dist = _dist(hand_lm[pip_idx], wrist)
        if tip_dist < pip_dist:
            curled += 1
    return curled >= FIST_MIN_CURLED_FINGERS
