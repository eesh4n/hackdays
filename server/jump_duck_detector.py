"""
JUMP / DUCK DETECTOR
=====================

Detects jumps and ducks from pose landmarks (subway-surfer style).

Signal: hip_y (midpoint of left/right hip, normalized 0-1, y grows
downward) moves roughly one torso-length up for a jump, or one
torso-length down for a duck, while torso_len (hip_y - shoulder_y)
stays roughly constant. Normalizing the hip_y delta by the person's
own torso length makes the thresholds self-calibrate to how close
they're standing to the camera.

Derived from recorded motion data. A dedicated small/quick-movement
calibration (server/motion_log_small_jumps.csv, motion_log_small_ducks.csv --
isolated jump-only and duck-only sessions, deliberately minimal-effort
movements) found: quick jumps land at -0.42x torso_len or beyond, quick
ducks at +0.12x to +0.54x (most clustering above +0.24x), and true
standing noise stays within +-0.09x. Thresholds sit at roughly 1.5x the
noise floor -- tighter than the original 2x margin, trading a little
false-positive risk for a lot more reflex sensitivity, since in-game
testing found 2x still too hard to trigger reliably for a quick reaction.
The live delta and these thresholds are surfaced on-screen (both the
debug camera window and the game HUD) specifically so this can keep
being tuned by eye without needing another full recording session.

The very first frame is NOT used as the baseline outright -- if it
catches the player mid-motion or not yet settled, every future delta
would be measured against a bad reference and jump/duck could stop
firing correctly for the rest of the session. Instead the first
WARMUP_FRAMES frames fast-converge the baseline (higher alpha) before
detection turns on.
"""

JUMP_THRESHOLD = 0.14
DUCK_THRESHOLD = 0.13
RETURN_THRESHOLD = 0.09
BASELINE_ALPHA = 0.05

WARMUP_FRAMES = 15
WARMUP_ALPHA = 0.3

MIN_TORSO_LEN = 0.05  # guards against a noisy frame's near-zero/negative torso_len blowing up the ratio


class JumpDuckDetector:
    def __init__(self, jump_threshold=JUMP_THRESHOLD, duck_threshold=DUCK_THRESHOLD,
                 return_threshold=RETURN_THRESHOLD, baseline_alpha=BASELINE_ALPHA):
        self.jump_threshold = jump_threshold
        self.duck_threshold = duck_threshold
        self.return_threshold = return_threshold
        self.baseline_alpha = baseline_alpha

        self.baseline_hip_y = None
        self.baseline_torso_len = None
        self.state = "neutral"  # "neutral", "jump", "duck"
        self.last_normalized_delta = 0.0  # exposed for debug overlays

        self._warmup_frames_left = WARMUP_FRAMES

    def reset(self):
        """Clears the baseline so the next update() call re-arms from
        scratch (fresh warmup included) -- call this when a person first
        appears in frame, so delta starts at 0 for them instead of being
        measured against whoever (or wherever) was last tracked before
        they left."""
        self.baseline_hip_y = None
        self.baseline_torso_len = None
        self.state = "neutral"
        self.last_normalized_delta = 0.0
        self._warmup_frames_left = WARMUP_FRAMES

    def update(self, hip_y, torso_len):
        """Feed one frame's hip_y / torso_len. Returns 'JUMP', 'DUCK', or None."""
        torso_len = max(torso_len, MIN_TORSO_LEN)

        if self.baseline_hip_y is None:
            self.baseline_hip_y = hip_y
            self.baseline_torso_len = torso_len
            return None

        if self._warmup_frames_left > 0:
            self._warmup_frames_left -= 1
            self.baseline_hip_y = (1 - WARMUP_ALPHA) * self.baseline_hip_y + WARMUP_ALPHA * hip_y
            self.baseline_torso_len = (1 - WARMUP_ALPHA) * self.baseline_torso_len + WARMUP_ALPHA * torso_len
            return None

        normalized_delta = (hip_y - self.baseline_hip_y) / self.baseline_torso_len
        self.last_normalized_delta = normalized_delta

        event = None
        if self.state == "neutral":
            if normalized_delta < -self.jump_threshold:
                self.state = "jump"
                event = "JUMP"
            elif normalized_delta > self.duck_threshold:
                self.state = "duck"
                event = "DUCK"
            else:
                a = self.baseline_alpha
                self.baseline_hip_y = (1 - a) * self.baseline_hip_y + a * hip_y
                self.baseline_torso_len = (1 - a) * self.baseline_torso_len + a * torso_len
        else:
            if abs(normalized_delta) < self.return_threshold:
                self.state = "neutral"

        return event
