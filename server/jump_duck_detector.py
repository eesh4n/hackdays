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

Derived from recorded motion data (server/motion_log.csv,
motion_log_jumps.csv): jumps land around -1.1x torso_len, ducks
around +1.1x torso_len, standing noise stays under ~0.15x.
"""

JUMP_THRESHOLD = 0.6
DUCK_THRESHOLD = 0.6
RETURN_THRESHOLD = 0.3
BASELINE_ALPHA = 0.05


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

    def update(self, hip_y, torso_len):
        """Feed one frame's hip_y / torso_len. Returns 'JUMP', 'DUCK', or None."""
        if self.baseline_hip_y is None:
            self.baseline_hip_y = hip_y
            self.baseline_torso_len = torso_len
            return None

        normalized_delta = (hip_y - self.baseline_hip_y) / self.baseline_torso_len

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
