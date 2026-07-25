"""
Player B tracker: obstacle placer.

Two models, two different jobs, sampled at very different rates:

  POSE (sampled every POSE_SAMPLE_EVERY_N_FRAMES frames, not every frame)
    -- answers one coarse question: which third of the screen is the
    player standing in (left/center/right lane), from hip-center x
    position. Lane doesn't need per-frame precision -- a player walking
    between lanes is a slow motion compared to a hand gesture -- so
    running pose at full rate was wasted compute measured to cost real
    fps for no benefit here. Shoulder width is also sampled here, as a
    stable per-person distance reference (see MAX_LOCK_JUMP_FRAC).

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

Both hands work independently and simultaneously (see _HandSlot): each
has its own lock, debounce, carry state, and cooldown timer, so you can
alternate hands to place obstacles back-to-back without waiting on a
single shared cooldown. Every frame, up to two detected hands are matched
to the two slots by proximity to each slot's last known position (see
_assign_candidates_to_slots) -- this is what lets each slot keep tracking
the SAME physical hand across frames instead of the two hands potentially
swapping which slot represents which.

Measured trade-off before building the hand-tracking approach (see
conversation, not re-run here): running pose AND hand on every single
frame cost real performance (~10fps, and only 20% of frames found a hand
at all -- likely resource contention between the two models rather than
a hand-visibility problem, since sampling pose fixed both numbers at
once). Hand-every-frame + pose-sampled measured ~27fps with an 88%
hand-detection rate -- close to running just one model.

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
FIST_MIN_CURLED_FINGERS = 2  # of 4 -- lowered from 3 so a loose/partial fist still registers
# A fully straight finger reads ~180 degrees at the PIP joint; this is how
# far below straight it has to bend to count as "curled". Tune down (e.g.
# 140) to require a tighter fist, up (e.g. 165) to accept a looser one.
# Raised from 150 -- a half-curled hand was too often reading as "open".
FIST_CURL_ANGLE_MAX_DEG = 165.0

POSE_SAMPLE_EVERY_N_FRAMES = 8  # lane only needs occasional updates, not every frame

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
MAX_LOCK_JUMP_FRAC = 2.2  # raised again from 1.6 -- being rejected as "a different hand"
                          # mid-gesture is worse than occasionally being too lenient about
                          # what counts as the same hand
DEFAULT_SHOULDER_WIDTH = 0.15  # fallback before pose has sampled even once

# If a tracked hand goes fully undetected for this long, drop whatever it
# was carrying rather than risk a "phantom drop" firing wherever the hand
# happens to reappear (e.g. back at the player's side, not a real gesture)
# once tracking resumes.
HAND_LOST_CANCEL_CARRY_SEC = 0.6
# If undetected for even longer than that, give up that slot's lock
# entirely -- the next detection is treated as a fresh first-ever
# acquisition rather than something that must match a stale position.
HAND_LOST_FORGET_LOCK_SEC = 1.0

# The fist-closed reading is a coarse per-frame heuristic and can flicker
# for a single frame even mid-gesture -- these debounce counts were 2/2 to
# guard against that, at the cost of a small delay before a grab/release
# is accepted. Lowered to 1 (effectively no debounce, especially for
# OPEN/release) to prioritize immediate responsiveness -- if releases
# start firing on a single noisy frame mid-carry, raise OPEN_DEBOUNCE_FRAMES
# back toward 2 rather than tolerating dropped carries.
OPEN_DEBOUNCE_FRAMES = 1
CLOSE_DEBOUNCE_FRAMES = 1

# Detections below this confidence are discarded before they ever reach the
# lock/fist logic -- a low-confidence detection is often a half-occluded
# or motion-blurred hand, exactly the kind of noisy read that would
# otherwise feed garbage into everything downstream. 0.6 was picked
# without measuring real handedness scores from this pipeline and turned
# out to reject far too many genuinely-fine detections in practice --
# lowered until proven otherwise by the on-screen confidence readout
# (see debug_overlay) rather than guessed again blind.
HAND_MIN_CONFIDENCE = 0.12

# Exponential smoothing on each tracked palm position (0 < alpha <= 1;
# higher = less smoothing, more responsive). Raw per-frame landmark
# position has real pixel-level jitter even when the hand isn't actually
# moving, which was making zone-boundary checks (grab/drop) flicker right
# at the edges. This is applied to the position used for BOTH the
# on-screen marker and the zone/lock math, not just cosmetically.
#
# Alpha is VELOCITY-ADAPTIVE, not fixed: smoothing always trades
# responsiveness for stability, and a fixed alpha tuned to kill jitter at
# rest makes the tracked position lag behind a genuinely fast-moving hand,
# which is exactly the opposite of what's needed during a fast gesture.
# When the raw frame-to-frame movement is large, that's real motion, not
# noise -- so alpha ramps up toward 1.0 (trust the raw position almost
# completely) and only drops toward PALM_SMOOTHING_ALPHA_MIN when the hand
# is nearly still, where jitter actually dominates the signal.
PALM_SMOOTHING_ALPHA_MIN = 0.35   # used when the hand is roughly stationary
PALM_SMOOTHING_ALPHA_MAX = 0.95   # used at or above FAST_MOVEMENT_THRESHOLD
FAST_MOVEMENT_THRESHOLD = 0.06    # normalized-distance frame-to-frame jump considered "fast"

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
# Widened from 0.20 -- a bigger target is easier to reach into and grab
# from without needing your hand right up near the top edge of frame.
GRAB_ZONE_HEIGHT_FRAC = 0.32

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
# This cooldown is GLOBAL (shared placement_count across both hands, so it
# reflects overall session pacing), but each hand slot tracks its OWN
# last-placement-time against it -- that's what lets both hands be
# independently ready/cooling at different moments, doubling throughput
# versus a single shared timer.
PLACEMENT_COOLDOWN_MAX_SEC = 0.9
PLACEMENT_COOLDOWN_MIN_SEC = 0.5
PLACEMENT_COOLDOWN_DECAY_PER_PLACEMENT = 0.006

# Visual style: matches server/player_a_tracking.py's debug overlay exactly
# (same colors, same font, same layout conventions) -- these two debug
# windows are meant to be watched side by side, and having Player A's
# skeleton view and Player B's hand view look like two unrelated tools
# would be a needless inconsistency. All colors are BGR (cv2's order).
BONE_COLOR = (0, 255, 136)      # green -- landmark connections
LANDMARK_COLOR = (0, 0, 255)    # red -- individual landmark points
LANE_DIVIDER_COLOR = (255, 255, 255)  # white
ZONE_ACTIVE_LABEL_COLOR = (0, 255, 136)     # green, matches the active-zone tint
ZONE_INACTIVE_LABEL_COLOR = (200, 200, 200)  # light grey
ZONE_ACTIVE_TINT_ALPHA = 0.15

# One fixed color per distinct hand state, reused everywhere that state is
# shown -- mirrors Player A's jump/duck/block/run mapping. OPEN is the
# neutral/idle state (like "run"), FIST is an active input just registered
# (like "jump"), CARRYING is a held/special state (like "block"). No B
# state maps to A's "duck" color -- there's no fourth distinct state to
# assign it to, and inventing one just to use every color would be
# arbitrary rather than meaningful.
STATE_COLOR_OPEN = (0, 255, 136)      # green
STATE_COLOR_FIST = (0, 200, 255)      # orange -- matches A's "jump"
STATE_COLOR_CARRYING = (255, 0, 200)  # magenta -- matches A's "block"
NO_HAND_COLOR = (0, 0, 255)           # red -- matches A's "No person detected"

# MediaPipe Hand topology connections (21 landmarks) -- the hand-tracking
# equivalent of Player A's POSE_CONNECTIONS subset, drawn the same way.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                # palm base
)


class _HandSlot:
    """Independent per-hand tracking state -- lock, debounce, carry, and a
    per-hand cooldown timer. Two of these let both hands grab/drop
    independently without one hand's gesture interfering with the
    other's, and without either accidentally locking onto the other's
    position."""

    def __init__(self, label):
        self.label = label  # purely for the overlay ("1" / "2")

        self.locked_palm_xy = None    # last known SMOOTHED position of this slot's hand
        self.smoothed_palm_xy = None  # EMA state, reset to None whenever this hand is lost
        self.hand_lost_since = None   # timestamp this hand FIRST went undetected, or None if currently seen
        self.fist_open_streak = 0     # consecutive frames read "open" while carrying, for debounce
        self.fist_closed_streak = 0   # consecutive frames read "closed" while not carrying, for debounce
        self.last_placement_time = 0.0

        # Live per-frame state, read by the overlay.
        self.hand_visible = False
        self.hand_confidence = None  # best handedness score matched to this slot this frame
        self.fist_closed = False
        self.hand_xy = None
        self.is_carrying = False
        self.last_landmarks = None  # raw (unsmoothed) 21-point landmarks, for skeleton drawing

        # Brief visual "pop" at this slot's place location, purely cosmetic.
        self.place_flash_xy = None
        self.place_flash_time = 0.0
        self.place_flash_color = (0, 255, 255)


class PlayerBTracker:
    """Feed frames in with process_frame(); poll .lane / ._slots (two
    _HandSlot instances) for live state each frame, and check the
    returned "events" list for placements -- both hands can each fire one
    on the same frame."""

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
        self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )
        self._shoulder_width = DEFAULT_SHOULDER_WIDTH  # updated whenever pose is sampled
        self._start_time = time.time()
        self._frame_count = 0

        self._slots = [_HandSlot("1"), _HandSlot("2")]

        # Live state.
        self.lane = 1              # default to center lane until first detection
        self.pose_visible = False

        self.placement_count = 0   # global, shared across both hands -- see cooldown comment above
        self._manual_last_placement_time = 0.0  # separate cooldown gate for the keyboard fallback

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
        if (now - self._manual_last_placement_time) < self.current_cooldown_sec:
            return None
        self._manual_last_placement_time = now
        self.placement_count += 1
        return {"lane": lane, "obstacle_type": obstacle_type}

    def process_frame(self, rgb_frame):
        """rgb_frame: HxWx3 RGB numpy array (already flipped/converted by caller).

        Returns a dict describing this frame's result:
            {
              "pose_visible": bool,
              "lane": 0|1|2,
              "events": [{"lane": int, "obstacle_type": str}, ...],
            }
        `events` is a list because both hands can each fire a placement on
        the same frame -- usually empty, occasionally one entry, rarely two.
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
        # Kept as (landmarks, score) pairs, not filtered yet -- the debug
        # overlay wants to show a slot's best-seen confidence even on
        # frames where nothing qualified, so tuning HAND_MIN_CONFIDENCE
        # can be read off real numbers instead of guessed at again.
        candidates = [lm for lm, s in scored if s >= HAND_MIN_CONFIDENCE]
        confidences = [s for lm, s in scored if s >= HAND_MIN_CONFIDENCE]

        assignment = self._assign_candidates_to_slots(candidates)

        events = []
        for slot, candidate_idx in zip(self._slots, assignment):
            if candidate_idx is not None:
                event = self._update_slot_from_candidate(
                    slot, candidates[candidate_idx], confidences[candidate_idx], now
                )
            else:
                self._handle_slot_not_visible(slot, now)
                event = None
            if event is not None:
                events.append(event)

        return {
            "pose_visible": self.pose_visible,
            "lane": self.lane,
            "events": events,
        }

    def _assign_candidates_to_slots(self, candidates):
        """Matches up to len(candidates) detected hands to self._slots, so
        each slot keeps tracking the same physical hand frame to frame
        instead of the two hands potentially swapping which slot
        represents which. Returns a list, one entry per slot, of the
        assigned candidate's index into `candidates`, or None if nothing
        was matched to that slot this frame.

        Locked slots get first claim on their nearest candidate (within
        MAX_LOCK_JUMP_FRAC shoulder-widths), matched smallest-distance-
        first so the better-fitting pair wins when two locked slots'
        nearest candidates would otherwise collide on the same one --
        correct for the 1-2 candidates / 1-2 slots case this always
        operates in, without needing a general assignment-problem solver.
        Any slot without a lock yet (first-ever acquisition) claims
        whatever candidate is left over, in slot order."""
        n_slots = len(self._slots)
        assigned = [None] * n_slots
        if not candidates:
            return assigned

        centers = [_palm_center(c) for c in candidates]
        max_dist_sq = (MAX_LOCK_JUMP_FRAC * self._shoulder_width) ** 2

        locked_slot_idxs = [i for i, s in enumerate(self._slots) if s.locked_palm_xy is not None]
        unlocked_slot_idxs = [i for i, s in enumerate(self._slots) if s.locked_palm_xy is None]

        pairs = []
        for si in locked_slot_idxs:
            lx, ly = self._slots[si].locked_palm_xy
            for ci, (cx, cy) in enumerate(centers):
                d2 = (cx - lx) ** 2 + (cy - ly) ** 2
                pairs.append((d2, si, ci))
        pairs.sort(key=lambda p: p[0])

        taken_slots, taken_candidates = set(), set()
        for d2, si, ci in pairs:
            if si in taken_slots or ci in taken_candidates or d2 > max_dist_sq:
                continue
            assigned[si] = ci
            taken_slots.add(si)
            taken_candidates.add(ci)

        leftover = [ci for ci in range(len(candidates)) if ci not in taken_candidates]
        for si in unlocked_slot_idxs:
            if si in taken_slots:
                continue
            if leftover:
                ci = leftover.pop(0)
                assigned[si] = ci
                taken_candidates.add(ci)

        return assigned

    def _update_slot_from_candidate(self, slot, lm, confidence, now):
        slot.hand_visible = True
        slot.hand_confidence = confidence
        slot.hand_lost_since = None
        slot.last_landmarks = lm

        raw_xy = _palm_center(lm)
        if slot.smoothed_palm_xy is None:
            slot.smoothed_palm_xy = raw_xy
        else:
            sx, sy = slot.smoothed_palm_xy
            rx, ry = raw_xy
            jump = ((rx - sx) ** 2 + (ry - sy) ** 2) ** 0.5
            # Ramp alpha up toward MAX as the jump approaches/exceeds
            # FAST_MOVEMENT_THRESHOLD -- a big frame-to-frame move is real
            # motion, not jitter, so trust the raw position almost
            # completely rather than lagging behind it.
            blend = min(1.0, jump / FAST_MOVEMENT_THRESHOLD)
            alpha = PALM_SMOOTHING_ALPHA_MIN + blend * (PALM_SMOOTHING_ALPHA_MAX - PALM_SMOOTHING_ALPHA_MIN)
            slot.smoothed_palm_xy = (
                alpha * rx + (1 - alpha) * sx,
                alpha * ry + (1 - alpha) * sy,
            )
        slot.hand_xy = slot.smoothed_palm_xy
        slot.locked_palm_xy = slot.hand_xy
        slot.fist_closed = _is_fist(lm)
        return self._update_grab_place_state(slot, now)

    def _handle_slot_not_visible(self, slot, now):
        """Called when no candidate was matched to this slot this frame --
        either nothing was detected at all, or a hand was detected but
        didn't pass the lock's distance gate (see
        _assign_candidates_to_slots) -- both cases mean "we don't
        currently have a trustworthy read on this slot's hand", handled
        identically."""
        slot.hand_visible = False
        slot.hand_xy = None
        slot.hand_confidence = None
        slot.fist_closed = False
        slot.fist_open_streak = 0
        slot.smoothed_palm_xy = None
        slot.last_landmarks = None

        if slot.hand_lost_since is None:
            slot.hand_lost_since = now
        lost_duration = now - slot.hand_lost_since

        if slot.is_carrying and lost_duration >= HAND_LOST_CANCEL_CARRY_SEC:
            # Don't leave a carry to be silently "placed" wherever the hand
            # happens to reappear once tracking resumes -- that's a
            # tracking hiccup, not a real drop gesture.
            slot.is_carrying = False

        if lost_duration >= HAND_LOST_FORGET_LOCK_SEC:
            slot.locked_palm_xy = None

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

    def _update_grab_place_state(self, slot, now):
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
        grab when nothing was intended) get the same treatment.

        Operates on one slot at a time -- lane and the global cooldown
        curve are shared, but is_carrying/streaks/last_placement_time are
        all per-slot, so this hand's gesture never affects the other's."""
        x, y = slot.hand_xy
        event = None

        if not slot.is_carrying:
            slot.fist_open_streak = 0  # not relevant yet -- don't let a stale streak leak into the next carry

            if slot.fist_closed:
                slot.fist_closed_streak += 1
            else:
                slot.fist_closed_streak = 0

            if slot.fist_closed_streak >= CLOSE_DEBOUNCE_FRAMES and self._in_grab_zone(y):
                slot.is_carrying = True
                slot.fist_closed_streak = 0
        else:
            slot.fist_closed_streak = 0  # symmetric reset, mirrors the open-streak reset above
            if slot.fist_closed:
                slot.fist_open_streak = 0
            else:
                slot.fist_open_streak += 1

            if slot.fist_open_streak >= OPEN_DEBOUNCE_FRAMES:
                drop_type = self._drop_zone_at(x, y)
                past_cooldown = (now - slot.last_placement_time) >= self.current_cooldown_sec
                if drop_type is not None and past_cooldown:
                    event = {"lane": self.lane, "obstacle_type": drop_type}
                    slot.last_placement_time = now
                    self.placement_count += 1
                    slot.place_flash_xy = (x, y)
                    slot.place_flash_time = now
                    slot.place_flash_color = DROP_ZONE_COLORS[drop_type]
                # Whether it landed in a valid drop zone, was still on
                # cooldown, or opened too early back in the grab strip --
                # the hand is empty again either way. Don't leave it
                # "stuck" carrying; the player can just grab again.
                slot.is_carrying = False
                slot.fist_open_streak = 0

        return event

    def debug_overlay(self, frame):
        """Draws both hands' skeletons, the grab/drop zones, and status
        text onto a BGR frame in place -- same color/layout convention as
        server/player_a_tracking.py's debug_overlay (bones, landmarks,
        zone highlighting, status text all follow the identical scheme),
        since the two debug windows are meant to be watched side by side.
        For the manual test window."""
        import cv2

        h, w = frame.shape[:2]
        grab_h = int(h * GRAB_ZONE_HEIGHT_FRAC)

        # Hand skeletons: same treatment as Player A's pose skeleton --
        # green connection lines, red filled landmark points.
        for slot in self._slots:
            if slot.last_landmarks is None:
                continue
            points = [(int(p.x * w), int(p.y * h)) for p in slot.last_landmarks]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, points[a], points[b], BONE_COLOR, 2, cv2.LINE_AA)
            for x, y in points:
                cv2.circle(frame, (x, y), 4, LANDMARK_COLOR, -1, cv2.LINE_AA)
            # Small identifying label at the wrist -- the skeleton itself
            # carries no per-hand color (matching the "one fixed color per
            # state, not per instance" principle), so this is the only cue
            # for which physical hand is which when both are on screen.
            wx, wy = points[WRIST]
            cv2.putText(frame, slot.label, (wx + 14, wy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Zone active-state: a zone is "active" if any tracked hand is
        # currently positioned in it -- shown live, regardless of whether
        # that hand ends up actually grabbing/dropping there.
        grab_active = any(
            s.hand_visible and s.hand_xy is not None and self._in_grab_zone(s.hand_xy[1])
            for s in self._slots
        )
        active_drop_types = {
            self._drop_zone_at(*s.hand_xy)
            for s in self._slots
            if s.hand_visible and s.hand_xy is not None and self._drop_zone_at(*s.hand_xy) is not None
        }

        # Grab zone tint + divider + label -- same scheme as Player A's
        # draw_lanes: soft green blend if active, white divider, label
        # colored green (active) or grey (inactive) at the zone's bottom.
        if grab_active:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, grab_h), ZONE_ACTIVE_LABEL_COLOR, -1)
            cv2.addWeighted(overlay, ZONE_ACTIVE_TINT_ALPHA, frame, 1 - ZONE_ACTIVE_TINT_ALPHA, 0, dst=frame)
        cv2.line(frame, (0, grab_h), (w, grab_h), LANE_DIVIDER_COLOR, 2, cv2.LINE_AA)
        grab_label_color = ZONE_ACTIVE_LABEL_COLOR if grab_active else ZONE_INACTIVE_LABEL_COLOR
        cv2.putText(frame, "GRAB", (w // 2 - 50, grab_h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, grab_label_color, 2, cv2.LINE_AA)

        # Drop zones -- same scheme, one third-width zone per obstacle type.
        for i, obstacle_type in enumerate(DROP_TYPES_LEFT_TO_RIGHT):
            x0 = int(w * i / 3)
            x1 = w if i == 2 else int(w * (i + 1) / 3)
            if obstacle_type in active_drop_types:
                overlay = frame.copy()
                cv2.rectangle(overlay, (x0, grab_h), (x1, h), ZONE_ACTIVE_LABEL_COLOR, -1)
                cv2.addWeighted(overlay, ZONE_ACTIVE_TINT_ALPHA, frame, 1 - ZONE_ACTIVE_TINT_ALPHA, 0, dst=frame)
            if i > 0:
                cv2.line(frame, (x0, grab_h), (x0, h), LANE_DIVIDER_COLOR, 2, cv2.LINE_AA)
            label_color = ZONE_ACTIVE_LABEL_COLOR if obstacle_type in active_drop_types else ZONE_INACTIVE_LABEL_COLOR
            cv2.putText(frame, obstacle_type.upper(), (x0 + 20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, label_color, 2, cv2.LINE_AA)

        # Status text, top-left, stacked -- one 3-line block per hand
        # (primary state / secondary detail / no-hand warning), same
        # positions and scale as Player A's single block, just repeated.
        for row, slot in enumerate(self._slots):
            self._draw_slot_status(frame, slot, row)

        # Lane (body position, a different axis than the hand-driven drop
        # zones above) doesn't correspond to any zone drawn on this screen,
        # so it's a plain secondary-style readout rather than forced into
        # the zone-label convention.
        lane_color = (120, 255, 120) if self.pose_visible else (120, 120, 255)
        cv2.putText(
            frame, f"lane (body position): {LANE_NAMES[self.lane].upper()}",
            (20, 40 + len(self._slots) * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lane_color, 1, cv2.LINE_AA,
        )

        # Placement flashes -- colored to match the drop zone's type,
        # confirming visually which type actually fired.
        for slot in self._slots:
            flash_age = time.time() - slot.place_flash_time
            if slot.place_flash_xy is not None and flash_age < 0.3:
                fx, fy = slot.place_flash_xy
                radius = int(20 + flash_age * 120)
                thickness = max(1, int(3 * (1 - flash_age / 0.3)))
                cv2.circle(frame, (int(fx * w), int(fy * h)), radius, slot.place_flash_color, thickness, cv2.LINE_AA)

        return frame

    def _draw_slot_status(self, frame, slot, row):
        """Primary action line (color-coded, (20,40)-style position) and a
        red "no hand" warning below when applicable -- Player A's layout,
        minus the numeric detail line, offset down per hand slot."""
        import cv2

        base_y = 40 + row * 50

        if slot.is_carrying:
            state, state_color = "CARRYING", STATE_COLOR_CARRYING
        elif slot.fist_closed:
            state, state_color = "FIST", STATE_COLOR_FIST
        else:
            state, state_color = "OPEN", STATE_COLOR_OPEN

        cv2.putText(
            frame, f"hand {slot.label}: {state}", (20, base_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 2, cv2.LINE_AA,
        )

        if not slot.hand_visible:
            cv2.putText(
                frame, f"No hand {slot.label} detected", (20, base_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, NO_HAND_COLOR, 2, cv2.LINE_AA,
            )


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
