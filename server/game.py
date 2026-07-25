"""
The game itself, rendered as a real 3D scene with Ursina (Panda3D under
the hood) -- a chase-cam endless-runner view down a 3-lane track,
inspired by Subway Surfers. Reads Player A's live state and Player B's
pending spawn events out of GameState every frame -- doesn't know or
care that one comes from a local camera thread and the other from a
network thread.

Collision is still deterministic rule-based (lane + world-depth
proximity + an avoided() lookup table), not Ursina's physics/collider
system -- physics collision on procedurally-scaled meshes is prone to
jitter and near-miss ambiguity; a plain rule table is exact and matches
"the low/medium/high hits need to be perfect."

Obstacle avoidance rules -- each type sits at a different height, so a
different action clears it:
    "high"   a beam hanging near the top of the lane
             -> avoided only by DUCK (jumping would put you into it)
    "medium" a barrel at chest height, centered in the lane
             -> avoided by EITHER jump or duck (low enough to clear by
                jumping, high enough to clear by ducking)
    "low"    a ground-level hurdle
             -> avoided only by JUMP (ducking would put you into it)
"""
import math
import os
import random
import sys

from ursina import (
    Ursina, Entity, Text, Sky, DirectionalLight, AmbientLight,
    color, held_keys, destroy, curve, camera, window, application, invoke, Vec3,
    time as ursina_time,
)
from ursina.shaders import lit_with_shadows_shader

import actors3d

# A bolder, more "HUD"-styled font for score/coins/shield text than
# Ursina's plain default. Referenced by system path rather than bundled
# in the repo (Windows-licensed font, not ours to redistribute) -- falls
# back to Ursina's default automatically if this exact file isn't present
# on a given machine, so it degrades gracefully rather than crashing.
_HUD_FONT_WIN_PATH = "C:/Windows/Fonts/AGENCYB.TTF"
if os.path.exists(_HUD_FONT_WIN_PATH):
    # Panda3D's font loader wants its own drive-letter convention
    # (/c/Windows/...), not a plain Windows path, or it fails to load it.
    HUD_FONT = "/c/Windows/Fonts/AGENCYB.TTF"
else:
    HUD_FONT = None

LANE_COUNT = 3
LANE_WIDTH = 2.6
TRACK_LENGTH = 40  # world units of track surface drawn behind the horizon

Z_NEAR = 0.0
Z_FAR = 34.0
OBSTACLE_WORLD_SPEED_START = 7.0    # world units/sec, how fast z decreases
OBSTACLE_WORLD_SPEED_RAMP = 0.22    # added per second survived

LANE_LERP_DURATION = 0.16
JUMP_HEIGHT = 1.7
JUMP_DURATION = 0.5
DUCK_HOLD = 0.0  # duck pose handled every frame by PlayerRig.update_pose, no tween needed

CAMERA_FOV = 78              # vertical fov, tuned/approved on a landscape window
CAMERA_BACK_OFFSET = 7.5     # landscape baseline/floor -- see camera_back_offset_for()
CAMERA_HEIGHT = 3.4
CAMERA_LANE_LERP = 6.0  # how eagerly the chase camera follows the player's lane
REMOVE_Z_MARGIN = 2.5   # obstacles/coins must clear the camera itself (not just the
                        # player) before being destroyed, or they visibly pop out of
                        # existence while still on screen -- see Game._build_scene(),
                        # which computes self.remove_z from the actual per-session
                        # camera distance (aspect-dependent, see camera_back_offset_for)

TIE_COUNT = 16
TIE_SPACING_Z = 2.2

BUILDING_COUNT = 140
BUILDING_MIN_DIST = 13
BUILDING_MAX_DIST = 95
# Height tiers, weighted so short/normal/landmark all show up mixed
# together rather than one uniform-looking wall of towers.
BUILDING_HEIGHT_TIERS = [
    ("short", 0.30, (3, 8)),
    ("normal", 0.55, (9, 32)),
    ("landmark", 0.15, (45, 72)),
]
WINDOWS_PER_BUILDING_AREA = 0.55  # roughly one window per this many sq. units of facade
STAR_COUNT = 60

COIN_SPAWN_INTERVAL_MIN = 1.8
COIN_SPAWN_INTERVAL_MAX = 3.2
COIN_SCORE_VALUE = 25
COIN_SPIN_SPEED = 220  # degrees/sec
COIN_ROW_SPACING_Z = 1.5

# Coin trail patterns -- weighted so a mix of short taps, long straight
# runs, and side-to-side zigzags show up, Subway-Surfers style, instead
# of always the same shape.
COIN_PATTERN_SHORT_LEN = (2, 3)
COIN_PATTERN_LONG_LEN = (5, 8)
COIN_PATTERN_ZIGZAG_LEN = (6, 10)
COIN_PATTERN_ZIGZAG_HOLD = (2, 3)  # coins per lane before the trail shifts over
COIN_PATTERN_WEIGHTS = {"short": 3, "long": 3, "zigzag": 3}

SHIELD_COIN_COST = 100    # coins required to activate a shield
SHIELD_DURATION_SEC = 2.0  # a shield lasts a fixed 2s once activated,
                           # regardless of how many hits it absorbs during
                           # that window -- not consumed by use
SHIELD_COLOR = color.rgba32(120, 220, 255, 90)
SHIELD_BAR_WIDTH = 0.26

# UI feedback: a floating "+N" on every coin pickup, a brief full-screen
# color flash on shield activation, and a quick scale-pop on each
# countdown digit change -- all driven by Ursina's animate_* scheduling
# so they play out over several frames without needing their own
# per-frame bookkeeping in update()/draw().
PICKUP_POPUP_DURATION = 0.6
SHIELD_FLASH_DURATION = 0.4
COUNTDOWN_PULSE_SCALE = 1.3
COUNTDOWN_PULSE_DURATION = 0.15

HUD_PANEL_COLOR = color.rgba32(20, 18, 30, 195)

# Countdown before each run starts (also on restart after game over). This
# isn't just cosmetic: PlayerATracker's JumpDuckDetector spends its first
# ~15 frames calibrating a baseline before it'll fire jump/duck at all
# (server/jump_duck_detector.py, WARMUP_FRAMES) -- without a countdown, a
# player who reacts to the window opening by immediately jumping has that
# very first input silently eaten. COUNTDOWN_SECONDS comfortably outlasts
# the warmup at any real webcam frame rate.
COUNTDOWN_SECONDS = 3.0
COUNTDOWN_GO_HOLD_SEC = 0.4  # how long "GO!" stays up before hiding

# Collision feedback: brief camera shake + player squash, both driven by
# Ursina's own animate_* scheduling (see _trigger_collision_feedback) so
# they keep playing after game_over stops Game.update() from running.
SHAKE_DURATION = 0.35
SHAKE_MAGNITUDE = 0.35   # world units of camera jitter
SHAKE_STEPS = 6
SQUASH_OUT_DURATION = 0.07
SQUASH_RECOVER_DURATION = 0.18

SKY_COLOR = color.rgb32(20, 15, 40)
GROUND_COLOR = color.rgb32(30, 28, 40)
TRACK_COLOR = color.rgb32(46, 44, 60)
LANE_LINE_COLOR = color.rgb32(150, 140, 190)
TIE_COLOR = color.rgb32(70, 66, 92)
# A varied skyline palette -- glass towers (blue/teal), concrete (warm
# grey/brown), brick, and violet-tinted high-rises, rather than one hue
# repeated across every building.
BUILDING_COLORS = [
    color.rgb32(64, 52, 92),    # violet
    color.rgb32(76, 58, 104),   # lighter violet
    color.rgb32(56, 48, 86),    # deep indigo
    color.rgb32(50, 74, 104),   # blue glass
    color.rgb32(56, 92, 100),   # teal glass
    color.rgb32(92, 76, 64),    # warm concrete
    color.rgb32(70, 76, 88),    # cool grey concrete
    color.rgb32(84, 68, 112),   # bright violet
    color.rgb32(104, 62, 70),   # brick red
    color.rgb32(58, 84, 132),   # deep blue glass
    color.rgb32(98, 90, 60),    # amber/gold-lit concrete
    color.rgb32(72, 100, 96),   # sage/teal
]
LANDMARK_COLORS = [
    color.rgb32(100, 88, 130),
    color.rgb32(70, 96, 140),
    color.rgb32(120, 80, 90),
]
WINDOW_COLOR = color.rgba32(255, 214, 140, 235)
HORIZON_GLOW_COLOR = color.rgba32(255, 150, 120, 60)
STAR_COLOR = color.rgba32(230, 230, 255, 210)
PLAYER_GLOW_COLOR = color.rgba32(80, 200, 255, 40)


def camera_back_offset_for(aspect_ratio):
    """Distance behind the player the chase camera sits. Keeps camera.fov
    fixed at its already-tuned landscape value (CAMERA_FOV) and instead
    pulls the camera back further on a narrower/portrait aspect ratio, so
    the full 3-lane track width stays framed -- rather than cranking fov
    itself to an extreme value, which would look fisheye-distorted on a
    tall monitor (preserving the exact landscape horizontal fov on a 9:16
    portrait aspect works out to a ~137 degree vertical fov, checked by
    hand -- too extreme to be worth it). Landscape aspect ratios get
    exactly the original fixed offset back (max() with the floor), so the
    already-tuned landscape camera is untouched by this change.
    """
    half_v = math.radians(CAMERA_FOV / 2)
    half_h = math.atan(math.tan(half_v) * aspect_ratio)
    half_width_needed = (LANE_WIDTH * LANE_COUNT) / 2 + 1.2  # track half-width + margin
    return max(CAMERA_BACK_OFFSET, half_width_needed / math.tan(half_h))


def avoided(action, obstacle_kind):
    if obstacle_kind == "high":
        return action == "duck"
    if obstacle_kind == "low":
        return action == "jump"
    if obstacle_kind == "medium":
        return action in ("jump", "duck")
    return False


def lane_x(lane):
    return (lane - 1) * LANE_WIDTH


class Obstacle:
    __slots__ = ("lane", "kind", "z", "entity", "resolved")

    def __init__(self, lane, kind):
        self.lane = lane
        self.kind = kind
        self.z = Z_FAR
        self.entity = actors3d.build_obstacle(kind, LANE_WIDTH)
        self.entity.x = lane_x(lane)
        self.entity.z = self.z
        self.resolved = False  # has this obstacle already had its hit/avoid check?

    def sync_transform(self):
        self.entity.z = self.z

    def destroy(self):
        destroy(self.entity)


class Coin:
    """A collectible spawned by the game itself (not placed by Player B)
    -- picked up just by being in the right lane when it arrives,
    regardless of jump/duck, same as a real endless runner. Normally
    spawned in a row (see Game._spawn_coin_row) with a staggered
    starting z, Subway-Surfers style, rather than one at a time."""
    __slots__ = ("lane", "z", "entity", "resolved")

    def __init__(self, lane, z=Z_FAR):
        self.lane = lane
        self.z = z
        self.entity = actors3d.build_coin()
        self.entity.x = lane_x(lane)
        self.entity.z = self.z
        self.resolved = False

    def sync_transform(self, dt):
        self.entity.z = self.z
        self.entity.rotation_y += COIN_SPIN_SPEED * dt

    def destroy(self):
        destroy(self.entity)


class Game:
    def __init__(self, game_state, fullscreen=True, window_size=None, keyboard_fallback=False):
        self.game_state = game_state
        self.keyboard_fallback = keyboard_fallback
        self._kb_lane = 1

        self.app = Ursina(
            title="Hackdays -- Player A vs Player B",
            borderless=False,
            fullscreen=fullscreen and window_size is None,
            size=window_size,
        )
        window.color = SKY_COLOR
        # Editor UI (exit button / fps counter / entity+collider debug
        # counters) only exists for a real onscreen window -- absent under
        # window_type='none' (headless testing), so don't assume it's there.
        if hasattr(window, "exit_button"):
            window.exit_button.visible = False
        if hasattr(window, "fps_counter"):
            window.fps_counter.enabled = False
        if hasattr(window, "entity_counter"):
            window.entity_counter.enabled = False
        if hasattr(window, "collider_counter"):
            window.collider_counter.enabled = False

        self._build_scene()
        self._build_hud()

        self._lane_a = 1
        self._action_a = "run"
        self._pose_visible_a = False
        self._last_lane_a = 1
        self._last_action_a = None
        self.best_score = 0.0  # persists across reset() -- a run's own high-water mark

        self.reset()

        application.trace = False

    # --- scene construction --------------------------------------------

    def _build_background(self):
        """Static atmosphere behind the track -- a warm horizon glow, a
        silhouette skyline, and a scatter of stars. None of this scrolls
        or reacts to gameplay; it's purely there so the world doesn't
        feel like an empty void around the track."""
        rng = random.Random(1234)  # fixed seed -- same skyline every run, not randomized noise

        # The true horizon -- where the flat ground plane visually recedes
        # to on screen -- is wherever the camera's own eye-height direction
        # points, i.e. y == CAMERA_HEIGHT, extended out to any distance
        # (the ray from the camera to any point at the same height it's
        # sitting at is always exactly horizontal). Placing the glow at
        # ground height (tried previously) was backwards; CAMERA_HEIGHT is
        # what actually lines the road up with it, no gap.
        for i, r in enumerate((26, 20, 14, 8)):
            Entity(model="circle", color=HORIZON_GLOW_COLOR, unlit=True,
                   billboard=True, scale=r, position=(0, CAMERA_HEIGHT - i * 0.15, 160))

        tier_names = [t[0] for t in BUILDING_HEIGHT_TIERS]
        tier_weights = [t[1] for t in BUILDING_HEIGHT_TIERS]
        tier_ranges = {t[0]: t[2] for t in BUILDING_HEIGHT_TIERS}

        # Stratified placement (jittered grid) instead of pure uniform
        # random -- guarantees even coverage along the track's length with
        # no large empty gaps, rather than leaving it to chance.
        z_min, z_max = -10, TRACK_LENGTH + 40
        per_side = BUILDING_COUNT // 2
        slot_size = (z_max - z_min) / per_side
        for side in (-1, 1):
            for slot in range(per_side):
                dist = rng.uniform(BUILDING_MIN_DIST, BUILDING_MAX_DIST)
                depth_along_track = z_min + slot * slot_size + rng.uniform(0, slot_size)

                tier = rng.choices(tier_names, weights=tier_weights)[0]
                height = rng.uniform(*tier_ranges[tier])

                if tier == "landmark":
                    # Taller, narrower "signature" towers standing out
                    # above the rest of the skyline.
                    width = rng.uniform(3, 5)
                    building_color = rng.choice(LANDMARK_COLORS)
                else:
                    width = rng.uniform(3, 9) if tier == "short" else rng.uniform(4, 9)
                    building_color = rng.choice(BUILDING_COLORS)
                bx, by, bz = side * dist, height / 2 - 0.5, depth_along_track
                Entity(model="cube", color=building_color,
                       scale=(width, height, width), unlit=True, position=(bx, by, bz))

                # Lit windows -- small billboarded quads scattered near the
                # building's volume rather than placed on an exact face
                # (the camera's viewing angle down the track varies enough
                # that billboarding reads convincingly as "windows" from
                # any angle, without needing per-face geometry).
                n_windows = max(3, int(height * width * WINDOWS_PER_BUILDING_AREA / 8))
                for _ in range(n_windows):
                    Entity(model="quad", color=WINDOW_COLOR, unlit=True, billboard=True, scale=0.3,
                           position=(bx + rng.uniform(-width * 0.4, width * 0.4),
                                     by - height / 2 + rng.uniform(1, height - 1),
                                     bz + rng.uniform(-width * 0.4, width * 0.4)))

        for _ in range(STAR_COUNT):
            Entity(model="circle", color=STAR_COLOR, unlit=True, billboard=True,
                   scale=rng.uniform(0.3, 0.8),
                   position=(rng.uniform(-90, 90), rng.uniform(20, 55), rng.uniform(-10, 160)))

    def _build_scene(self):
        Sky(color=SKY_COLOR)
        Entity(model="plane", scale=(200, 1, 200), y=-0.01, color=GROUND_COLOR, double_sided=True,
               shader=lit_with_shadows_shader)
        self._build_background()

        self.track = Entity(
            model="cube", color=TRACK_COLOR,
            scale=(LANE_WIDTH * LANE_COUNT, 0.05, TRACK_LENGTH),
            position=(0, 0, TRACK_LENGTH / 2 - 2),
            shader=lit_with_shadows_shader,
        )

        self.sun = DirectionalLight(shadows=True)
        self.sun.look_at((1, -1.4, 1.2))
        AmbientLight(color=color.rgba32(130, 120, 150, 140))

        for i in (1, 2):
            Entity(model="cube", color=LANE_LINE_COLOR,
                   scale=(0.06, 0.06, TRACK_LENGTH),
                   position=((i - 1.5) * LANE_WIDTH, 0.03, TRACK_LENGTH / 2 - 2),
                   shader=lit_with_shadows_shader)

        self._ties = []
        for _ in range(TIE_COUNT):
            tie = Entity(model="cube", color=TIE_COLOR, scale=(LANE_WIDTH * LANE_COUNT * 0.98, 0.03, 0.08),
                         shader=lit_with_shadows_shader)
            self._ties.append(tie)

        self.player = actors3d.PlayerRig(position=(lane_x(1), 0, 0))
        self._glow = Entity(model="circle", color=PLAYER_GLOW_COLOR, rotation_x=90,
                             scale=LANE_WIDTH * 0.95, y=0.02)
        self._shield_bubble = Entity(parent=self.player, model="sphere", color=SHIELD_COLOR,
                                      scale=1.7, y=0.9, unlit=True, enabled=False)

        # window.aspect_ratio reflects the real fullscreen resolution at this
        # point (native, whatever orientation the monitor/OS is set to) --
        # not a fixed assumption of landscape. fov itself stays fixed; only
        # the camera distance adapts (see camera_back_offset_for's docstring
        # for why deriving fov directly was rejected -- fisheye risk).
        camera.fov = CAMERA_FOV
        self.camera_back_offset = camera_back_offset_for(window.aspect_ratio)
        self.remove_z = -(self.camera_back_offset + REMOVE_Z_MARGIN)
        self._camera_x = lane_x(1)

    def _build_hud(self):
        # One shared panel behind score/best/coins/shield instead of each
        # having its own separate auto-sized chip -- reads as a single HUD
        # block instead of three disconnected floating boxes.
        self._hud_panel = Entity(parent=camera.ui, model="quad", color=HUD_PANEL_COLOR,
                                  scale=(0.62, 0.32), position=(-0.55, 0.325), origin=(0, 0))

        self.score_text = Text(
            parent=camera.ui, text="SCORE: 0", position=(-0.85, 0.45), scale=1.8,
            color=color.rgb32(235, 235, 240), font=HUD_FONT,
        )
        self.best_text = Text(
            parent=camera.ui, text="BEST: 0", position=(-0.85, 0.405), scale=1.0,
            color=color.rgb32(180, 178, 190), font=HUD_FONT,
        )
        self.coins_text = Text(
            parent=camera.ui, text="COINS: 0", position=(-0.85, 0.335), scale=1.35,
            color=color.rgb32(255, 210, 60), font=HUD_FONT,
        )
        self.shield_text = Text(
            parent=camera.ui, text="SHIELD: 0%", position=(-0.85, 0.275), scale=1.35,
            color=color.rgb32(120, 220, 255), font=HUD_FONT,
        )
        self.shield_bar_bg = Entity(parent=camera.ui, model="quad", color=color.rgba32(255, 255, 255, 35),
                                     scale=(SHIELD_BAR_WIDTH, 0.022), position=(-0.71, 0.232), origin=(-0.5, 0))
        self.shield_bar_fill = Entity(parent=camera.ui, model="quad", color=color.rgb32(120, 220, 255),
                                       scale=(0.001, 0.022), position=(-0.71, 0.232), origin=(-0.5, 0))

        self.warn_text = Text(
            parent=camera.ui, text="TRACKING LOST", position=(-0.85, -0.46), scale=0.9,
            color=color.rgb32(255, 120, 120), background=True, enabled=False, font=HUD_FONT,
        )
        # Positioned below the shared HUD panel (which ends around y=0.16)
        # rather than at the original y=0.27, which used to land right on
        # top of the shield text/bar added alongside it.
        self.jump_duck_debug_text = Text(
            parent=camera.ui, text="", position=(-0.85, 0.15), scale=0.9,
            color=color.rgb32(180, 180, 190), background=True,
        )

        self.gameover_text = Text(
            parent=camera.ui, text="GAME OVER", origin=(0, 0), position=(0, 0.2),
            scale=3.6, color=color.rgb32(255, 80, 80), enabled=False, font=HUD_FONT,
        )
        self.new_best_text = Text(
            parent=camera.ui, text="NEW BEST!", origin=(0, 0), position=(0, 0.1),
            scale=1.9, color=color.rgb32(255, 210, 60), enabled=False, font=HUD_FONT,
        )
        self.final_stats_text = Text(
            parent=camera.ui, text="", origin=(0, 0), position=(0, 0.0),
            scale=1.3, color=color.rgb32(220, 220, 228), enabled=False, font=HUD_FONT,
        )
        self.hint_text = Text(
            parent=camera.ui, text="PRESS R TO RESTART, Q TO QUIT", origin=(0, 0),
            position=(0, -0.12), scale=1.4, color=color.rgb32(235, 235, 240), enabled=False,
            font=HUD_FONT,
        )
        self.countdown_text = Text(
            parent=camera.ui, text="", origin=(0, 0), position=(0, 0.1),
            scale=4.5, color=color.rgb32(255, 255, 255), background=True, enabled=False,
        )

    def _spawn_pickup_popup(self, text, popup_color):
        """A floating "+N" that pops up above the coins counter and fades
        out -- otherwise a coin pickup is silent except for the counter
        ticking, which is easy to miss mid-sprint."""
        popup = Text(parent=camera.ui, text=text, position=(-0.55, 0.335), scale=1.5,
                     origin=(0, 0), color=popup_color, font=HUD_FONT)
        popup.animate_position((-0.55, 0.5), duration=PICKUP_POPUP_DURATION, curve=curve.out_quad)
        popup.animate_color(color.rgba(popup_color[0], popup_color[1], popup_color[2], 0),
                             duration=PICKUP_POPUP_DURATION, curve=curve.linear)
        destroy(popup, delay=PICKUP_POPUP_DURATION + 0.05)

    def _flash_screen(self, flash_color):
        """A brief full-screen color wash on shield activation -- the
        bubble around the player is easy to miss if the camera's mid-shake
        or the player's off to one side of frame."""
        flash = Entity(parent=camera.ui, model="quad",
                        color=color.rgba(flash_color[0], flash_color[1], flash_color[2], 0.35),
                        scale=(8, 8), z=-1)
        flash.animate_color(color.rgba(flash_color[0], flash_color[1], flash_color[2], 0),
                             duration=SHIELD_FLASH_DURATION, curve=curve.out_quad)
        destroy(flash, delay=SHIELD_FLASH_DURATION + 0.05)

    # --- lifecycle -------------------------------------------------------

    def reset(self):
        for obstacle in getattr(self, "obstacles", []):
            obstacle.destroy()
        self.obstacles = []
        for coin in getattr(self, "coins", []):
            coin.destroy()
        self.coins = []
        self.coins_collected = 0       # spendable balance -- drops by SHIELD_COIN_COST on activation
        self.total_coins_collected = 0  # lifetime-this-run count, for the game-over stats -- never decreases
        self.shield_active = False
        self.shield_timer = 0.0
        self._next_coin_spawn = random.uniform(COIN_SPAWN_INTERVAL_MIN, COIN_SPAWN_INTERVAL_MAX)
        self.score = 0.0
        self.survived_sec = 0.0
        self.game_over = False
        self.game_state.reset()
        self.gameover_text.enabled = False
        self.hint_text.enabled = False
        self.new_best_text.enabled = False
        self.final_stats_text.enabled = False

        self.countdown_active = True
        self.countdown_remaining = COUNTDOWN_SECONDS
        self.countdown_text.text = str(math.ceil(COUNTDOWN_SECONDS))
        self.countdown_text.enabled = True

    def run(self):
        game = self

        class _Driver(Entity):
            def update(self):
                if game.keyboard_fallback:
                    game._apply_keyboard_fallback()
                dt = ursina_time.dt
                if game.countdown_active:
                    game.update_countdown(dt)
                elif not game.game_over:
                    game.update(dt)
                game.draw()

            def input(self, key):
                if key in ("escape", "q"):
                    application.quit()
                    sys.exit(0)
                elif key == "r" and game.game_over:
                    game.reset()
                elif key == "e":
                    # Stands in for Player B's real shield gesture -- goes
                    # through the exact same GameState entrypoint a real
                    # websocket message would (still gated by game.py's own
                    # >=100 coins / no-stacking rules), so it's available in
                    # both --no-camera testing AND with the real camera
                    # running, as a manual fallback if Player B's gesture
                    # isn't wired up yet for a demo.
                    game.game_state.push_shield_request({"player": "B", "action": "shield"})
                elif game.keyboard_fallback:
                    if key in ("left arrow", "a"):
                        game._kb_lane = max(0, game._kb_lane - 1)
                    elif key in ("right arrow", "d"):
                        game._kb_lane = min(2, game._kb_lane + 1)

        self._driver = _Driver()
        self.app.run()

    def _apply_keyboard_fallback(self):
        """Stand-in for the camera thread when --no-camera is used --
        writes keyboard state into GameState exactly like the real
        tracker would, so game.py and websocket_server.py can be tested
        end-to-end with zero camera/mediapipe dependency."""
        if held_keys["space"]:
            action = "jump"
        elif held_keys["down arrow"] or held_keys["s"]:
            action = "duck"
        else:
            action = "run"
        self.game_state.set_player_a(self._kb_lane, action, pose_visible=True)

    # --- per-frame logic ---------------------------------------------------

    def update_countdown(self, dt):
        """Runs instead of update() while the "Get Ready" countdown is up.
        Player A's live pose still drives the visible rig (so it's obvious
        tracking is already working) and the camera/scenery keep animating,
        but no obstacles/coins spawn or resolve. Player B's spawn events are
        drained and discarded here rather than left to pile up, so punches
        thrown during the countdown don't all land at once the instant it
        ends."""
        self.game_state.drain_spawn_events()
        self._lane_a, self._action_a, self._pose_visible_a = self.game_state.get_player_a()

        self._update_player(dt)
        self._update_camera(dt)
        self._update_scenery()

        self.countdown_remaining -= dt
        if self.countdown_remaining > 0:
            new_text = str(math.ceil(self.countdown_remaining))
            if new_text != self.countdown_text.text:
                self.countdown_text.text = new_text
                self._pulse_countdown()
        else:
            self.countdown_active = False
            self.countdown_text.text = "GO!"
            self._pulse_countdown()
            invoke(setattr, self.countdown_text, "enabled", False, delay=COUNTDOWN_GO_HOLD_SEC)

    def _pulse_countdown(self):
        base_scale = 4.5
        self.countdown_text.scale = base_scale * COUNTDOWN_PULSE_SCALE
        self.countdown_text.animate_scale(base_scale, duration=COUNTDOWN_PULSE_DURATION, curve=curve.out_quad)

    def update(self, dt):
        self.survived_sec += dt
        self.score += dt * 10
        speed = OBSTACLE_WORLD_SPEED_START + OBSTACLE_WORLD_SPEED_RAMP * self.survived_sec

        for event in self.game_state.drain_spawn_events():
            self.obstacles.append(Obstacle(event["lane"], event["obstacle"]))

        for _ in self.game_state.drain_shield_requests():
            self._try_activate_shield()

        self._lane_a, self._action_a, self._pose_visible_a = self.game_state.get_player_a()

        # Player A's own shield trigger: PlayerATracker already detects a
        # "block" pose (both wrists raised above shoulder height, see
        # server/player_a_tracking.py) and reports it through the same
        # direct GameState channel as lane/jump/duck -- no network hop
        # needed since A is co-located with the game, unlike B's
        # websocket-based shield request. Calling this every frame the
        # pose holds is safe/idempotent: _try_activate_shield() itself
        # already guards against re-activating while shield_active or
        # spending coins twice.
        if self._action_a == "block":
            self._try_activate_shield()

        self._update_shield_timer(dt)

        still_alive = []
        for obstacle in self.obstacles:
            prev_z = obstacle.z
            obstacle.z -= speed * dt

            # Exact zero-crossing, not a wide "am I near" band -- fires exactly
            # once, the instant the obstacle reaches the player, regardless of
            # frame rate/speed (prev_z was still in front, new z is at/behind).
            # A band-based check could freeze the game-over frame with the
            # obstacle already deep inside the player model; snapping it to
            # Z_NEAR here means a hit always freezes on a clean contact pose.
            if not obstacle.resolved and prev_z > Z_NEAR >= obstacle.z:
                obstacle.resolved = True
                if obstacle.lane == self._lane_a and not avoided(self._action_a, obstacle.kind):
                    if self.shield_active:
                        # The shield breaks the collision -- the obstacle
                        # shatters instead of ending the run. It stays
                        # active (time-based only, see _update_shield_timer)
                        # so it can absorb further hits until it expires.
                        obstacle.z = self.remove_z - 1  # marks it for cleanup below
                    else:
                        obstacle.z = Z_NEAR
                        self.game_over = True
                        self._trigger_collision_feedback()

            obstacle.sync_transform()

            if obstacle.z > self.remove_z:
                still_alive.append(obstacle)
            else:
                obstacle.destroy()
        self.obstacles = still_alive

        self._update_coins(dt, speed)
        self._update_player(dt)
        self._update_camera(dt)
        self._update_scenery()

        if self.game_over:
            self._show_game_over()

    def _spawn_coin_row(self):
        """A trail of coins, staggered in z so they file in and get
        collected one after another -- the classic Subway Surfers "coin
        trail" look, instead of one lone coin at a time. Picks one of
        three shapes each time: a short 2-3 coin tap, a longer straight
        run, or a zigzag that walks back and forth across the lanes."""
        pattern = random.choices(
            list(COIN_PATTERN_WEIGHTS.keys()), weights=list(COIN_PATTERN_WEIGHTS.values())
        )[0]

        if pattern == "zigzag":
            length = random.randint(*COIN_PATTERN_ZIGZAG_LEN)
            lane = random.randint(0, LANE_COUNT - 1)
            direction = random.choice((-1, 1))
            hold = random.randint(*COIN_PATTERN_ZIGZAG_HOLD)
            since_shift = 0
            for i in range(length):
                self.coins.append(Coin(lane, z=Z_FAR + i * COIN_ROW_SPACING_Z))
                since_shift += 1
                if since_shift >= hold:
                    since_shift = 0
                    hold = random.randint(*COIN_PATTERN_ZIGZAG_HOLD)
                    next_lane = lane + direction
                    if not 0 <= next_lane < LANE_COUNT:
                        direction *= -1
                        next_lane = lane + direction
                    lane = next_lane
        else:
            length_range = COIN_PATTERN_SHORT_LEN if pattern == "short" else COIN_PATTERN_LONG_LEN
            length = random.randint(*length_range)
            lane = random.randint(0, LANE_COUNT - 1)
            for i in range(length):
                self.coins.append(Coin(lane, z=Z_FAR + i * COIN_ROW_SPACING_Z))

    def _update_coins(self, dt, speed):
        self._next_coin_spawn -= dt
        if self._next_coin_spawn <= 0:
            self._next_coin_spawn = random.uniform(COIN_SPAWN_INTERVAL_MIN, COIN_SPAWN_INTERVAL_MAX)
            self._spawn_coin_row()

        still_alive = []
        for coin in self.coins:
            prev_z = coin.z
            coin.z -= speed * dt

            # Same exact-crossing approach as obstacles: collect the instant
            # the coin reaches the player, in whichever lane they're in --
            # unlike obstacles, no action is required, just being in lane.
            if not coin.resolved and prev_z > Z_NEAR >= coin.z:
                coin.resolved = True
                if coin.lane == self._lane_a:
                    # Coins just accumulate here -- spending SHIELD_COIN_COST
                    # of them only happens in _try_activate_shield(), on an
                    # explicit shield-activation message, not automatically
                    # the moment the count crosses the threshold.
                    self.coins_collected += 1
                    self.total_coins_collected += 1
                    self.score += COIN_SCORE_VALUE
                    self._spawn_pickup_popup(f"+{COIN_SCORE_VALUE}", color.rgb32(255, 210, 60))
                    coin.destroy()
                    continue

            coin.sync_transform(dt)

            if coin.z > self.remove_z:
                still_alive.append(coin)
            else:
                coin.destroy()
        self.coins = still_alive

    def _trigger_collision_feedback(self):
        """Camera shake + player squash on the exact frame a collision
        fires. Both are built entirely from Ursina's animate_* scheduling
        (not this file's own update()/draw() loop) since Game.update()
        never runs again once game_over is set -- an effect that needed a
        few more frames of a normal update() call to play out would just
        freeze on frame one. animate_* keeps ticking on Ursina's own task
        manager regardless."""
        base = Vec3(self._camera_x, CAMERA_HEIGHT, -self.camera_back_offset)
        step_duration = SHAKE_DURATION / SHAKE_STEPS
        for i in range(1, SHAKE_STEPS + 1):
            magnitude = SHAKE_MAGNITUDE * (1 - i / SHAKE_STEPS)
            jitter = Vec3(random.uniform(-magnitude, magnitude), random.uniform(-magnitude, magnitude), 0)
            camera.animate_position(base + jitter, duration=step_duration,
                                     delay=step_duration * (i - 1), curve=curve.linear)
        camera.animate_position(base, duration=step_duration,
                                 delay=step_duration * SHAKE_STEPS, curve=curve.linear)

        normal_scale = self.player.torso.scale
        squashed_scale = Vec3(normal_scale.x * 1.35, normal_scale.y * 0.5, normal_scale.z * 1.25)
        self.player.torso.animate_scale(squashed_scale, duration=SQUASH_OUT_DURATION, curve=curve.out_quad)
        self.player.torso.animate_scale(normal_scale, duration=SQUASH_RECOVER_DURATION,
                                         delay=SQUASH_OUT_DURATION, curve=curve.out_quad)

    def _try_activate_shield(self):
        """Called once per valid shield-activation message received over
        the websocket (see websocket_server.py / game_state.py). Does
        nothing if a shield is already active (no stacking) or there
        aren't enough coins banked -- either way the request is just
        dropped, not queued for later."""
        if self.shield_active or self.coins_collected < SHIELD_COIN_COST:
            return
        self.coins_collected -= SHIELD_COIN_COST
        self.shield_active = True
        self.shield_timer = SHIELD_DURATION_SEC
        self._flash_screen(color.rgb32(120, 220, 255))

    def _update_shield_timer(self, dt):
        if not self.shield_active:
            return
        self.shield_timer -= dt
        if self.shield_timer <= 0:
            self.shield_active = False
            self.shield_timer = 0.0

    def _update_player(self, dt):
        target_x = lane_x(self._lane_a)
        if self._lane_a != self._last_lane_a:
            self.player.animate_x(target_x, duration=LANE_LERP_DURATION, curve=curve.out_quad)
            self._glow.animate_x(target_x, duration=LANE_LERP_DURATION, curve=curve.out_quad)
            self._last_lane_a = self._lane_a

        if self._action_a != self._last_action_a:
            if self._action_a == "jump":
                self.player.animate_y(JUMP_HEIGHT, duration=JUMP_DURATION * 0.42,
                                       curve=curve.out_quad)
                self.player.animate_y(0, duration=JUMP_DURATION * 0.58,
                                       delay=JUMP_DURATION * 0.42, curve=curve.in_quad)
            elif self._last_action_a == "jump":
                self.player.y = 0
            self._last_action_a = self._action_a

        self.player.update_pose(self._action_a, dt)

    def _update_camera(self, dt):
        target_x = lane_x(self._lane_a)
        self._camera_x += (target_x - self._camera_x) * min(1, CAMERA_LANE_LERP * dt)
        camera.position = (self._camera_x, CAMERA_HEIGHT, -self.camera_back_offset)
        camera.look_at((target_x, 1.1, 12))

    def _update_scenery(self):
        speed = OBSTACLE_WORLD_SPEED_START + OBSTACLE_WORLD_SPEED_RAMP * self.survived_sec

        tie_phase = (self.survived_sec * speed) % TIE_SPACING_Z
        for i, tie in enumerate(self._ties):
            z = i * TIE_SPACING_Z - tie_phase
            tie.z = z
            tie.enabled = Z_NEAR - 1 <= z <= Z_FAR

    def draw(self):
        self.score_text.text = f"SCORE: {int(self.score)}"
        self.best_text.text = f"BEST: {int(max(self.best_score, self.score))}"
        self.coins_text.text = f"COINS: {self.coins_collected}"
        if self.shield_active:
            self.shield_text.text = f"SHIELD: ACTIVE ({self.shield_timer:.1f}s)"
            self.shield_bar_fill.scale_x = SHIELD_BAR_WIDTH
        else:
            pct = min(1.0, self.coins_collected / SHIELD_COIN_COST)
            self.shield_text.text = f"SHIELD: {int(pct * 100)}%"
            self.shield_bar_fill.scale_x = max(0.001, SHIELD_BAR_WIDTH * pct)
        self._shield_bubble.enabled = self.shield_active
        self.warn_text.enabled = not self._pose_visible_a

        delta, jump_threshold, duck_threshold = self.game_state.get_player_a_debug()
        self.jump_duck_debug_text.text = (
            f"delta: {delta:+.2f}  (jump<-{jump_threshold:.2f}  duck>{duck_threshold:.2f})"
        )

    def _show_game_over(self):
        self.gameover_text.enabled = True
        self.hint_text.enabled = True

        is_new_best = self.score > self.best_score
        if is_new_best:
            self.best_score = self.score
        self.new_best_text.enabled = is_new_best
        self.final_stats_text.enabled = True
        self.final_stats_text.text = f"COINS COLLECTED: {self.total_coins_collected}   SCORE: {int(self.score)}"
