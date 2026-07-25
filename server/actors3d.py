"""
Procedurally-built 3D actors for the Ursina scene: the player rig and
the three obstacle kinds. Everything here is assembled from Ursina's
built-in primitive meshes (cube/sphere/procedural cylinder) with solid
colors and lighting -- no external model or texture files, so there's
nothing to download, license, or go missing on a teammate's laptop.

Every entity here uses lit_with_shadows_shader explicitly -- plain
Ursina Entities are *not* automatically lit just because a Light exists
in the scene; Panda3D's fixed-function lighting pipeline kicks in
instead and overrides the vertex color with a flat white material,
which is exactly what turns a scene "white" if this shader is missing
(see game.py's lighting setup for the matching note).

Cylinder pivot note: Ursina's procedural Cylinder mesh is built from
(0,0,0) to (0,height,0) by default -- its pivot is at one END, not the
center, and not hanging downward. Two helpers below build it two
different ways depending on what a given part needs:
  - _cylinder(): centered (start=-0.5) -- for parts meant to span
    symmetrically around their placement point (barrel, bands, chains).
  - _limb(): pivoted at the TOP, hanging down (direction=(0,-1,0)) --
    for arms/legs, which need to swing from a joint (shoulder/hip) like
    a real hinge, not from their own far tip.

Obstacle height storytelling (matches game.py's avoided() rule table):
    high   -- a beam hanging near the top of the lane   -> duck under it
    medium -- a barrel at chest height, center of lane   -> jump OR duck
    low    -- a hurdle bar at ground level               -> jump over it
"""
import math

from ursina import Entity, color
from ursina.models.procedural.cylinder import Cylinder
from ursina.shaders import lit_with_shadows_shader

HIGH_Y = 2.05
MEDIUM_Y = 1.45  # raised again (was 1.05, then 1.35) for a bigger safety
                 # margin against the ducked torso's top edge -- see
                 # update_pose()'s "duck" branch for the matching squash
                 # tightening. At 1.05 the two actually overlapped; 1.35 left
                 # only ~0.245 units of clearance, still too tight to call
                 # "strict". Current gap is ~0.38 units, verified in the
                 # geometry test suite.
LOW_Y = 0.35

OBSTACLE_COLORS = {
    "high": color.rgb32(255, 176, 40),
    "medium": color.rgb32(150, 92, 52),
    "low": color.rgb32(224, 60, 60),
}
MEDIUM_WALL_HEIGHT = 0.55  # kept the same as the old barrel's diameter so
                           # the validated clearance math still applies
MEDIUM_WALL_DEPTH = 0.4

COIN_Y = 1.0
COIN_COLOR = color.rgb32(255, 210, 60)
COIN_RIM_COLOR = color.rgb32(205, 150, 20)


def _cube(**kwargs):
    kwargs.setdefault("model", "cube")
    kwargs.setdefault("shader", lit_with_shadows_shader)
    return Entity(**kwargs)


def _cylinder(**kwargs):
    """Centered cylinder: spans -0.5..+0.5 along its local axis, so it
    extends symmetrically around wherever it's placed/rotated."""
    kwargs.setdefault("shader", lit_with_shadows_shader)
    return Entity(model=Cylinder(resolution=10, start=-0.5), **kwargs)


def _limb(**kwargs):
    """A limb segment pivoted at its TOP, hanging straight down --
    rotating it swings from the joint (shoulder/hip), not from its own
    tip, matching how a real arm/leg hinges."""
    kwargs.setdefault("shader", lit_with_shadows_shader)
    return Entity(model=Cylinder(resolution=8, direction=(0, -1, 0)), **kwargs)


def build_obstacle(kind, lane_width):
    """Returns a root Entity for one obstacle instance; game.py positions
    it via .x/.z and destroys it once it's passed the player."""
    root = Entity()
    span = lane_width * 0.82

    if kind == "high":
        root.y = HIGH_Y
        _cube(parent=root, color=OBSTACLE_COLORS["high"], scale=(span, 0.34, 0.55))
        # hazard-stripe teeth along the front face
        n_teeth = max(3, int(span / 0.35))
        for i in range(n_teeth):
            tx = -span / 2 + (i + 0.5) * (span / n_teeth)
            _cube(parent=root, color=color.rgb32(35, 35, 40),
                  scale=(span / n_teeth * 0.5, 0.36, 0.06), position=(tx, 0, -0.29))
        # support chains up to a fixed "ceiling"
        for cx in (-span * 0.32, span * 0.32):
            _cylinder(parent=root, color=color.rgb32(90, 90, 100),
                      scale=(0.05, 1.15, 0.05), position=(cx, 0.9, 0))

    elif kind == "medium":
        # A solid barricade wall (was a rolling barrel) -- same overall
        # bounding box (MEDIUM_WALL_HEIGHT) as the old barrel so the
        # validated duck/jump clearance math still applies unchanged.
        root.y = MEDIUM_Y
        _cube(parent=root, color=OBSTACLE_COLORS["medium"],
              scale=(span, MEDIUM_WALL_HEIGHT, MEDIUM_WALL_DEPTH))
        # hazard-stripe teeth on the front face, same treatment as high/low
        n_teeth = max(3, int(span / 0.35))
        for i in range(n_teeth):
            tx = -span / 2 + (i + 0.5) * (span / n_teeth)
            if i % 2 == 0:
                _cube(parent=root, color=color.rgb32(35, 35, 40),
                      scale=(span / n_teeth * 0.5, MEDIUM_WALL_HEIGHT * 0.85, MEDIUM_WALL_DEPTH + 0.02),
                      position=(tx, 0, 0))
        # end posts give it a barricade silhouette instead of a plain slab
        post_w = 0.14
        for px in (-span / 2 + post_w * 0.6, span / 2 - post_w * 0.6):
            _cube(parent=root, color=color.rgb32(90, 90, 98),
                  scale=(post_w, MEDIUM_WALL_HEIGHT * 1.2, MEDIUM_WALL_DEPTH * 1.3), position=(px, 0, 0))

    else:  # "low"
        root.y = LOW_Y
        _cube(parent=root, color=OBSTACLE_COLORS["low"], scale=(span, 0.24, 0.3))
        n_teeth = max(3, int(span / 0.3))
        for i in range(n_teeth):
            tx = -span / 2 + (i + 0.5) * (span / n_teeth)
            if i % 2 == 0:
                _cube(parent=root, color=color.rgb32(245, 245, 245),
                      scale=(span / n_teeth * 0.55, 0.26, 0.05), position=(tx, 0, -0.16))
        leg_h = LOW_Y  # legs reach from the ground up to the bar
        for lx in (-span * 0.42, span * 0.42):
            _cube(parent=root, color=color.rgb32(75, 75, 82),
                  scale=(0.1, leg_h, 0.1), position=(lx, -LOW_Y + leg_h / 2, 0))

    return root


def build_coin():
    """A flat gold disc standing upright, facing the player -- spins
    around its own vertical axis in game.py for the classic
    face-to-edge-to-face "spinning coin" look."""
    root = Entity()
    # scale=(x, y, z): built as a flat puck (thin along y, radius in x/z),
    # then tipped 90 degrees around x so the flat faces point along z
    # (toward the player) instead of up/down.
    _cylinder(parent=root, color=COIN_COLOR, scale=(0.5, 0.12, 0.5), rotation=(90, 0, 0))
    _cylinder(parent=root, color=COIN_RIM_COLOR, scale=(0.56, 0.05, 0.56), rotation=(90, 0, 0))
    root.y = COIN_Y
    return root


# Joint layout, bottom-up: legs hang from the hip, torso sits on top of
# the legs, arms hang from the shoulder near the top of the torso.
LEG_LENGTH = 0.55
HIP_Y = LEG_LENGTH
TORSO_HEIGHT = 0.7
TORSO_CENTER_Y = HIP_Y + TORSO_HEIGHT / 2
SHOULDER_Y = TORSO_CENTER_Y + TORSO_HEIGHT / 2 - 0.06
ARM_LENGTH = 0.5
HEAD_Y = TORSO_CENTER_Y + TORSO_HEIGHT / 2 + 0.22


class PlayerRig(Entity):
    """A small low-poly runner: head + torso + limbs, with a distinct
    pose per action and a light idle stride animation while running.
    Arms/legs are hung from fixed joint heights (shoulder/hip) using
    _limb()'s top-pivoted cylinders, so rotating them swings from the
    joint like a real hinge instead of flailing from the wrong end."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        body_color = color.rgb32(80, 200, 255)
        shade_color = color.rgb32(40, 130, 175)

        self.torso = _cube(parent=self, color=body_color,
                            scale=(0.62, TORSO_HEIGHT, 0.4), y=TORSO_CENTER_Y)
        self.head = Entity(parent=self, model="sphere", color=body_color, scale=0.42, y=HEAD_Y,
                            shader=lit_with_shadows_shader)

        self.left_arm = _limb(parent=self, color=shade_color, scale=(0.13, ARM_LENGTH, 0.13),
                               position=(-0.38, SHOULDER_Y, 0))
        self.right_arm = _limb(parent=self, color=shade_color, scale=(0.13, ARM_LENGTH, 0.13),
                                position=(0.38, SHOULDER_Y, 0))
        self.left_leg = _limb(parent=self, color=shade_color, scale=(0.16, LEG_LENGTH, 0.16),
                               position=(-0.18, HIP_Y, 0))
        self.right_leg = _limb(parent=self, color=shade_color, scale=(0.16, LEG_LENGTH, 0.16),
                                position=(0.18, HIP_Y, 0))

        self._t = 0.0
        self.current_action = "run"

    def update_pose(self, action, dt):
        self._t += dt
        self.current_action = action

        if action == "block":
            self.torso.scale = (0.62, TORSO_HEIGHT, 0.4)
            self.torso.y = TORSO_CENTER_Y
            self.head.y = HEAD_Y
            self.left_arm.rotation = (-170, 0, 25)
            self.right_arm.rotation = (-170, 0, -25)
            self.left_leg.rotation = (0, 0, 0)
            self.right_leg.rotation = (0, 0, 0)
        elif action == "duck":
            # 0.35 (was 0.55, then 0.4) -- squashed further each time a
            # visible overlap with the medium obstacle's bottom edge turned
            # up. Current gap is ~0.38 units, verified in the geometry test
            # suite rather than eyeballed.
            self.torso.scale = (0.85, TORSO_HEIGHT * 0.35, 0.55)
            self.torso.y = HIP_Y + TORSO_HEIGHT * 0.35 / 2
            self.head.y = self.torso.y + TORSO_HEIGHT * 0.35 / 2 + 0.2
            self.left_arm.rotation = (-100, 0, 0)
            self.right_arm.rotation = (-100, 0, 0)
            self.left_leg.rotation = (20, 0, 0)
            self.right_leg.rotation = (-20, 0, 0)
        elif action == "jump":
            self.torso.scale = (0.62, TORSO_HEIGHT, 0.4)
            self.torso.y = TORSO_CENTER_Y
            self.head.y = HEAD_Y
            self.left_arm.rotation = (-150, 0, 0)
            self.right_arm.rotation = (-150, 0, 0)
            self.left_leg.rotation = (55, 0, 0)
            self.right_leg.rotation = (-55, 0, 0)
        else:  # run -- opposite arm/leg pairs swing together (natural gait)
            self.torso.scale = (0.62, TORSO_HEIGHT, 0.4)
            self.torso.y = TORSO_CENTER_Y
            self.head.y = HEAD_Y
            swing = math.sin(self._t * 11.0) * 45
            self.left_arm.rotation = (-swing, 0, 0)
            self.right_arm.rotation = (swing, 0, 0)
            self.left_leg.rotation = (swing, 0, 0)
            self.right_leg.rotation = (-swing, 0, 0)
