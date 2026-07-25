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
needed): for each of the four non-thumb fingers, curl is the bend ANGLE
at that finger's own PIP joint (using x/y/z, so the hand's rough depth
counts too) -- not distance from the wrist. Distance-from-wrist is
rotation-sensitive: it depends on the hand's overall position/orientation
in frame, so turning the hand sideways to the camera changes those
numbers even if the fingers haven't moved relative to each other at all.
Joint angle only depends on a finger's own two segments relative to one
another, so "is this finger bent" reads the same regardless of how the
whole hand is rotated in view. Majority vote (3 of 4 fingers) tolerates
one misdetected finger.

Uses the MediaPipe Tasks API for both PoseLandmarker and HandLandmarker
(mp.solutions.* is gone as of mediapipe 0.10.35 -- see
hackathon-prep/PLAN.md finding #1). Needs pose_landmarker_lite.task and
hand_landmarker.task in the project's models/ folder (see MODEL_PATHS).
"""
import math
import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

# Pose landmark indices (MediaPipe Pose topology). Hips give lane from
# hip-center x position; shoulders give a stable body-scale reference (see
# MAX_LOCK_JUMP_FRAC below) so distance thresholds scale with how close
# the player is standing to the camera instead of being a fixed number of
# raw image-coordinate units.
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

# Hand landmark indices (MediaPipe Hand topology, 21 points per hand).
WRIST = 0
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
# (MCP, PIP, TIP) trios for the four fingers used in the fist heuristic --
# thumb excluded, it doesn't fold the same way on a fist. The bend angle
# is measured AT the PIP joint (the middle element of each trio).
FINGER_MCP_PIP_TIP_TRIOS = (
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
)
FIST_MIN_CURLED_FINGERS = 3  # of 4, majority vote
# A fully straight finger reads ~180 degrees at the PIP joint; this is how
# far below straight it has to bend to count as "curled". Tune down (e.g.
# 140) to require a tighter fist, up (e.g. 165) to accept a looser one.
FIST_CURL_ANGLE_MAX_DEG = 150.0

POSE_SAMPLE_EVERY_N_FRAMES = 5  # lane only needs occasional updates, not every frame

# Hand-lock tuning. A locked candidate is only accepted if its palm sits
# within MAX_LOCK_JUMP_FRAC shoulder-widths of the last known position --
# without a ceiling, "pick whichever candidate is closest" will still
# confidently adopt a completely different hand (or person) if that's all
# that's on screen, since "closest" has no concept of "too far to
# plausibly be the same hand between two frames". Scaled by shoulder width
# (a stable per-person reference from pose, not the moving hand itself) so
# the same real-world hand speed reads the same regardless of how far the
# player is standing from the camera -- a fixed raw-coordinate threshold
# would be too tight up close and too loose far away, the same distance-
# invariance problem that bit the old punch thresholds.
MAX_LOCK_JUMP_FRAC = 1.0
DEFAULT_SHOULDER_WIDTH = 0.15  # fallback before pose has sampled even once

# If the tracked hand goes fully undetected for this long, drop whatever
# was being carried rather than risk a "phantom drop" firing wherever the
# hand happens to reappear (e.g. back at the player's side, not a real
# gesture) once tracking resumes.
HAND_LOST_CANCEL_CARRY_SEC = 0.6
# If undetected for even longer than that, give up the lock entirely --
# the next detection is treated as a fresh first-ever acquisition rather
# than something that must match a stale remembered position.
HAND_LOST_FORGET_LOCK_SEC = 1.0

# The fist-closed reading is a coarse per-frame heuristic and can flicker
# for a single frame even mid-gesture. Requiring it to read "open"/"closed"
# for this many consecutive frames before treating either as real stops
# one noisy frame from silently cancelling an entire carry (OPEN) or
# triggering an accidental grab just from passing a hand through the
# strip (CLOSE) -- same root cause (a single-frame misread), so both
# directions get the same debounce, not just the one that was reported.
OPEN_DEBOUNCE_FRAMES = 2
CLOSE_DEBOUNCE_FRAMES = 2

# Detections below this confidence are discarded before they ever reach the
# lock/fist logic -- a low-confidence detection is often a half-occluded
# or motion-blurred hand, exactly the kind of noisy read that would
# otherwise feed garbage into everything downstream. 0.6 was picked
# without measuring real handedness scores from this pipeline and turned
# out to reject far too many genuinely-fine detections in practice --
# lowered until proven otherwise by the on-screen confidence readout
# (see debug_overlay) rather than guessed again blind.
HAND_MIN_CONFIDENCE = 0.2

# Exponential smoothing on the tracked palm position (0 < alpha <= 1;
# higher = less smoothing, more responsive). Raw per-frame landmark
# position has real pixel-level jitter even when the hand isn't actually
# moving, which was making zone-boundary checks (grab/drop) flicker right
# at the edges. This is applied to the position used for BOTH the
# on-screen marker and the zone/lock math, not just cosmetically.
PALM_SMOOTHING_ALPHA = 0.5

LANE_NAMES = ("left", "center", "right")
OBSTACLE_HIGH = "high"
OBSTACLE_MEDIUM = "medium"
OBSTACLE_LOW = "low"

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
    """Feed frames in with process_frame(); poll .lane / .is_carrying /
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
        self._locked_palm_xy = None  # last known SMOOTHED position of the hand we're tracking
        self._hand_lost_since = None  # timestamp the hand FIRST went undetected, or None if currently seen
        self._fist_open_streak = 0    # consecutive frames read "open" while carrying, for debounce
        self._fist_closed_streak = 0  # consecutive frames read "closed" while not carrying, for debounce
        self._shoulder_width = DEFAULT_SHOULDER_WIDTH  # updated whenever pose is sampled
        self._smoothed_palm_xy = None  # EMA state, reset to None whenever the hand is lost
        self._start_time = time.time()
        self._frame_count = 0

        # Live state.
        self.lane = 1              # default to center lane until first detection
        self.pose_visible = False
        self.hand_visible = False
        self.hand_confidence = None  # best handedness score seen this frame, even if it didn't qualify
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

    def try_manual_placement(self, lane, obstacle_type):
        """For the keyboard fallback (SPACE key) -- applies the exact same
        cooldown pacing as a real gesture placement, rather than letting
        the fallback path spam instantly with no pacing at all. Returns
        the event dict if it fired, or None if still on cooldown."""
        now = time.time()
        if (now - self._last_placement_time) < self.current_cooldown_sec:
            return None
        self._last_placement_time = now
        self.placement_count += 1
        return {"lane": lane, "obstacle_type": obstacle_type}

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
                pose_lm = pose_result.pose_landmarks[0]
                self.lane = self._update_lane(pose_lm)
                width = _dist(pose_lm[LEFT_SHOULDER], pose_lm[RIGHT_SHOULDER])
                if width > 0:
                    self._shoulder_width = width
            else:
                self.pose_visible = False

        hand_result = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        scored = _pair_with_confidence(hand_result.hand_landmarks, hand_result.handedness)
        # Tracked regardless of whether anything qualifies below, so the
        # debug overlay can show "a hand WAS seen, just below threshold"
        # instead of an undifferentiated "no hand" -- this is exactly the
        # number that should drive HAND_MIN_CONFIDENCE, not a guess.
        self.hand_confidence = max((s for _, s in scored), default=None)
        candidates = [lm for lm, s in scored if s >= HAND_MIN_CONFIDENCE]

        selected = None
        if candidates:
            # None means the model found a confident hand (or two), but
            # neither is close enough to our last known position to
            # plausibly be the SAME hand -- treated the same as no
            # detection at all, rather than blindly adopting whatever
            # showed up.
            selected = self._select_locked_hand(candidates)

        event = None
        if selected is not None:
            self.hand_visible = True
            self._hand_lost_since = None
            raw_xy = _palm_center(selected)
            if self._smoothed_palm_xy is None:
                self._smoothed_palm_xy = raw_xy
            else:
                sx, sy = self._smoothed_palm_xy
                rx, ry = raw_xy
                self._smoothed_palm_xy = (
                    PALM_SMOOTHING_ALPHA * rx + (1 - PALM_SMOOTHING_ALPHA) * sx,
                    PALM_SMOOTHING_ALPHA * ry + (1 - PALM_SMOOTHING_ALPHA) * sy,
                )
            self.hand_xy = self._smoothed_palm_xy
            self._locked_palm_xy = self.hand_xy
            self.fist_closed = _is_fist(selected)
            event = self._update_grab_place_state(now)
        else:
            self._smoothed_palm_xy = None  # don't let stale smoothing bleed into the next acquisition
            self._handle_hand_not_visible(now)

        return {
            "pose_visible": self.pose_visible,
            "hand_visible": self.hand_visible,
            "lane": self.lane,
            "is_carrying": self.is_carrying,
            "event": event,
        }

    def _select_locked_hand(self, all_hand_landmarks):
        """Given 1-2 detected hands this frame, picks whichever one we're
        actually tracking, or None if nothing qualifies.

        With no prior lock (first-ever detection, or the lock has been
        given up after a long enough gap -- see HAND_LOST_FORGET_LOCK_SEC),
        the first candidate is trusted outright; there's nothing to compare
        it against yet.

        With a prior lock, this does NOT just trust whichever candidate is
        closest -- distance alone doesn't distinguish "this is the same
        hand, slightly moved" from "this is a completely different hand
        that happens to be nearest of the options available". A candidate
        is only accepted if it's within MAX_LOCK_JUMP_FRAC shoulder-widths
        of the last known position -- scaled by the player's own shoulder
        width (from pose) rather than a fixed raw-coordinate distance, so
        the same real hand speed reads the same whether the player is
        standing close to or far from the camera. A single detected hand
        still has to clear this bar, not just multiple candidates (a
        normal single-player frame usually has exactly one hand detected,
        so gating only the multi-candidate case would leave the common
        failure mode -- the tracked hand leaving frame and a different
        hand becoming the sole detection -- completely unguarded)."""
        if self._locked_palm_xy is None:
            return all_hand_landmarks[0]

        lx, ly = self._locked_palm_xy
        max_dist = MAX_LOCK_JUMP_FRAC * self._shoulder_width

        def _dist_sq(lm):
            px, py = _palm_center(lm)
            return (px - lx) ** 2 + (py - ly) ** 2

        best = min(all_hand_landmarks, key=_dist_sq)
        if _dist_sq(best) > max_dist ** 2:
            return None
        return best

    def _handle_hand_not_visible(self, now):
        """Called when no hand landmarks were detected this frame, OR a
        hand was detected but didn't pass the lock's distance gate (see
        _select_locked_hand) -- both cases mean "we don't currently have a
        trustworthy read on our tracked hand", handled identically."""
        self.hand_visible = False
        self.hand_xy = None
        self.fist_closed = False
        self._fist_open_streak = 0

        if self._hand_lost_since is None:
            self._hand_lost_since = now
        lost_duration = now - self._hand_lost_since

        if self.is_carrying and lost_duration >= HAND_LOST_CANCEL_CARRY_SEC:
            # Don't leave a carry to be silently "placed" wherever the hand
            # happens to reappear once tracking resumes -- that's a
            # tracking hiccup, not a real drop gesture.
            self.is_carrying = False

        if lost_duration >= HAND_LOST_FORGET_LOCK_SEC:
            self._locked_palm_xy = None

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
        closed before or after entering the zone, both grab. is_carrying
        flipping True is itself the one-shot guard against re-firing every
        frame the condition holds, so this doesn't need to track a
        previous-frame fist state to detect a precise transition -- that
        precise-timing requirement was most of what made grabbing feel
        clunky.

        Both grab and drop are debounced, not just drop: the fist-closed
        heuristic is a coarse per-frame geometric read and can flicker for
        a single frame in either direction, so both a false "open"
        mid-carry (would silently cancel the whole carry) and a false
        "closed" while merely passing a hand through the strip (would
        grab when nothing was intended) get the same treatment."""
        x, y = self.hand_xy
        event = None

        if not self.is_carrying:
            self._fist_open_streak = 0  # not relevant yet -- don't let a stale streak leak into the next carry

            if self.fist_closed:
                self._fist_closed_streak += 1
            else:
                self._fist_closed_streak = 0

            if self._fist_closed_streak >= CLOSE_DEBOUNCE_FRAMES and self._in_grab_zone(y):
                self.is_carrying = True
                self._fist_closed_streak = 0
        else:
            self._fist_closed_streak = 0  # symmetric reset, mirrors the open-streak reset above
            if self.fist_closed:
                self._fist_open_streak = 0
            else:
                self._fist_open_streak += 1

            if self._fist_open_streak >= OPEN_DEBOUNCE_FRAMES:
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
                self._fist_open_streak = 0

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
        conf_text = f"{self.hand_confidence:.2f}" if self.hand_confidence is not None else "--"
        cv2.putText(frame, f"hand: {hand_state}  (confidence: {conf_text}, need >= {HAND_MIN_CONFIDENCE})",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hand_color, 2)

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


def _pair_with_confidence(hand_landmarks_list, handedness_list):
    """Pairs each detected hand's landmarks with its handedness confidence
    score (0.0 if the score is missing for some reason). Does NOT filter --
    the caller decides what to do with low scores, and keeping every score
    around (not just the ones that pass) is what makes the on-screen
    confidence readout actually useful for tuning HAND_MIN_CONFIDENCE from
    real numbers instead of guessing at it again."""
    pairs = []
    for lm, handedness in zip(hand_landmarks_list, handedness_list):
        score = handedness[0].score if handedness else 0.0
        pairs.append((lm, score))
    return pairs


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


def _joint_angle_deg(a, b, c):
    """Angle at vertex b, between vectors b->a and b->c, in degrees. Uses
    x/y/z (not just x/y) so a finger curling mostly in depth -- which is
    exactly what happens when the hand is turned sideways to the camera --
    still shows up. ~180 degrees means straight; smaller means bent."""
    v1 = (a.x - b.x, a.y - b.y, a.z - b.z)
    v2 = (c.x - b.x, c.y - b.y, c.z - b.z)
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    if n1 == 0 or n2 == 0:
        return 180.0  # degenerate (overlapping landmarks) -- treat as straight, not curled
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # guard float drift outside acos's domain
    return math.degrees(math.acos(cos_angle))


def _is_fist(hand_lm):
    """A finger is curled if its PIP joint is bent past FIST_CURL_ANGLE_MAX_DEG
    from straight -- true when folded into a fist, false when extended.
    Deliberately NOT based on distance from the wrist: that depends on the
    hand's overall position/orientation in frame (turning the hand sideways
    changes those distances even with no actual finger movement), while a
    joint's own bend angle only depends on its two segments relative to
    each other. Majority vote across the four non-thumb fingers tolerates
    one misdetected landmark without flipping the whole classification."""
    curled = 0
    for mcp_idx, pip_idx, tip_idx in FINGER_MCP_PIP_TIP_TRIOS:
        angle = _joint_angle_deg(hand_lm[mcp_idx], hand_lm[pip_idx], hand_lm[tip_idx])
        if angle <= FIST_CURL_ANGLE_MAX_DEG:
            curled += 1
    return curled >= FIST_MIN_CURLED_FINGERS
