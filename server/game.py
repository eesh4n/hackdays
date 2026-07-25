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
import random
import sys

from ursina import (
    Ursina, Entity, Text, Sky, DirectionalLight, AmbientLight,
    color, held_keys, destroy, curve, camera, window, application, time as ursina_time,
)
from ursina.shaders import lit_with_shadows_shader

import actors3d

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
SCENERY_COUNT_PER_SIDE = 10
SCENERY_SPACING_Z = 4.2

BUILDING_COUNT = 22
BUILDING_MIN_DIST = 55
BUILDING_MAX_DIST = 110
STAR_COUNT = 60

COIN_SPAWN_INTERVAL_MIN = 1.1
COIN_SPAWN_INTERVAL_MAX = 2.2
COIN_SCORE_VALUE = 25
COIN_SPIN_SPEED = 220  # degrees/sec

SKY_COLOR = color.rgb32(20, 15, 40)
GROUND_COLOR = color.rgb32(30, 28, 40)
TRACK_COLOR = color.rgb32(46, 44, 60)
LANE_LINE_COLOR = color.rgb32(150, 140, 190)
TIE_COLOR = color.rgb32(70, 66, 92)
SCENERY_COLOR = color.rgb32(35, 30, 52)
BUILDING_COLORS = [
    color.rgb32(38, 30, 58),
    color.rgb32(48, 34, 64),
    color.rgb32(30, 26, 50),
]
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
    regardless of jump/duck, same as a real endless runner."""
    __slots__ = ("lane", "z", "entity", "resolved")

    def __init__(self, lane):
        self.lane = lane
        self.z = Z_FAR
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
        # Editor UI (exit button / fps counter) only exists for a real
        # onscreen window -- absent under window_type='none' (headless
        # testing), so don't assume it's there.
        if hasattr(window, "exit_button"):
            window.exit_button.visible = False
        if hasattr(window, "fps_counter"):
            window.fps_counter.enabled = False

        self._build_scene()
        self._build_hud()

        self._lane_a = 1
        self._action_a = "run"
        self._pose_visible_a = False
        self._last_lane_a = 1
        self._last_action_a = None

        self.reset()

        application.trace = False

    # --- scene construction --------------------------------------------

    def _build_background(self):
        """Static atmosphere behind the track -- a warm horizon glow, a
        silhouette skyline, and a scatter of stars. None of this scrolls
        or reacts to gameplay; it's purely there so the world doesn't
        feel like an empty void around the track."""
        rng = random.Random(1234)  # fixed seed -- same skyline every run, not randomized noise

        for i, r in enumerate((26, 20, 14, 8)):
            Entity(model="circle", color=HORIZON_GLOW_COLOR, unlit=True,
                   billboard=True, scale=r, position=(0, 5 + i * 0.6, 140))

        for _ in range(BUILDING_COUNT):
            side = rng.choice((-1, 1))
            dist = rng.uniform(BUILDING_MIN_DIST, BUILDING_MAX_DIST)
            depth_along_track = rng.uniform(-10, TRACK_LENGTH + 40)
            height = rng.uniform(6, 30)
            width = rng.uniform(4, 9)
            Entity(model="cube", color=rng.choice(BUILDING_COLORS),
                   scale=(width, height, width), unlit=True,
                   position=(side * dist, height / 2 - 0.5, depth_along_track))

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
        for i in (1, 2):
            Entity(model="cube", color=LANE_LINE_COLOR,
                   scale=(0.06, 0.06, TRACK_LENGTH),
                   position=((i - 1.5) * LANE_WIDTH, 0.03, TRACK_LENGTH / 2 - 2),
                   shader=lit_with_shadows_shader)

        self.sun = DirectionalLight(shadows=True)
        self.sun.look_at((1, -1.4, 1.2))
        AmbientLight(color=color.rgba32(130, 120, 150, 140))

        self._ties = []
        for _ in range(TIE_COUNT):
            tie = Entity(model="cube", color=TIE_COLOR, scale=(LANE_WIDTH * LANE_COUNT * 0.98, 0.03, 0.08),
                         shader=lit_with_shadows_shader)
            self._ties.append(tie)

        self._scenery = []
        for side in (-1, 1):
            for _ in range(SCENERY_COUNT_PER_SIDE):
                pillar = Entity(model="cube", color=SCENERY_COLOR, scale=(0.3, 1, 0.3),
                                shader=lit_with_shadows_shader)
                self._scenery.append((side, pillar))

        self.player = actors3d.PlayerRig(position=(lane_x(1), 0, 0))
        self._glow = Entity(model="circle", color=PLAYER_GLOW_COLOR, rotation_x=90,
                             scale=LANE_WIDTH * 0.95, y=0.02)

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
        self.score_text = Text(
            parent=camera.ui, text="Score: 0", position=(-0.85, 0.45), scale=1.6,
            color=color.rgb32(235, 235, 240), background=True,
        )
        self.coins_text = Text(
            parent=camera.ui, text="Coins: 0", position=(-0.85, 0.39), scale=1.2,
            color=color.rgb32(255, 210, 60), background=True,
        )
        self.warn_text = Text(
            parent=camera.ui, text="Player A pose not detected -- using last known state",
            position=(-0.85, -0.46), scale=1.0, color=color.rgb32(255, 200, 80), enabled=False,
        )
        self.gameover_text = Text(
            parent=camera.ui, text="GAME OVER", origin=(0, 0), position=(0, 0.08),
            scale=3.2, color=color.rgb32(255, 80, 80), enabled=False,
        )
        self.hint_text = Text(
            parent=camera.ui, text="Press R to restart, Q to quit", origin=(0, 0),
            position=(0, -0.08), scale=1.3, color=color.rgb32(235, 235, 240), enabled=False,
        )

    # --- lifecycle -------------------------------------------------------

    def reset(self):
        for obstacle in getattr(self, "obstacles", []):
            obstacle.destroy()
        self.obstacles = []
        for coin in getattr(self, "coins", []):
            coin.destroy()
        self.coins = []
        self.coins_collected = 0
        self._next_coin_spawn = random.uniform(COIN_SPAWN_INTERVAL_MIN, COIN_SPAWN_INTERVAL_MAX)
        self.score = 0.0
        self.survived_sec = 0.0
        self.game_over = False
        self.game_state.reset()
        self.gameover_text.enabled = False
        self.hint_text.enabled = False

    def run(self):
        game = self

        class _Driver(Entity):
            def update(self):
                if game.keyboard_fallback:
                    game._apply_keyboard_fallback()
                dt = ursina_time.dt
                if not game.game_over:
                    game.update(dt)
                game.draw()

            def input(self, key):
                if key in ("escape", "q"):
                    application.quit()
                    sys.exit(0)
                elif key == "r" and game.game_over:
                    game.reset()
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

    def update(self, dt):
        self.survived_sec += dt
        self.score += dt * 10
        speed = OBSTACLE_WORLD_SPEED_START + OBSTACLE_WORLD_SPEED_RAMP * self.survived_sec

        for event in self.game_state.drain_spawn_events():
            self.obstacles.append(Obstacle(event["lane"], event["obstacle"]))

        self._lane_a, self._action_a, self._pose_visible_a = self.game_state.get_player_a()

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
                    obstacle.z = Z_NEAR
                    self.game_over = True

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

    def _update_coins(self, dt, speed):
        self._next_coin_spawn -= dt
        if self._next_coin_spawn <= 0:
            self._next_coin_spawn = random.uniform(COIN_SPAWN_INTERVAL_MIN, COIN_SPAWN_INTERVAL_MAX)
            self.coins.append(Coin(random.randint(0, LANE_COUNT - 1)))

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
                    self.coins_collected += 1
                    self.score += COIN_SCORE_VALUE
                    coin.destroy()
                    continue

            coin.sync_transform(dt)

            if coin.z > self.remove_z:
                still_alive.append(coin)
            else:
                coin.destroy()
        self.coins = still_alive

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

        scenery_phase = (self.survived_sec * speed) % SCENERY_SPACING_Z
        for i, (side, pillar) in enumerate(self._scenery):
            z = i * SCENERY_SPACING_Z - scenery_phase
            pillar.z = z
            pillar.x = side * (LANE_WIDTH * LANE_COUNT / 2 + 1.2)
            pillar.y = 0.5
            pillar.enabled = Z_NEAR - 1 <= z <= Z_FAR

    def draw(self):
        self.score_text.text = f"Score: {int(self.score)}"
        self.coins_text.text = f"Coins: {self.coins_collected}"
        self.warn_text.enabled = not self._pose_visible_a

    def _show_game_over(self):
        self.gameover_text.enabled = True
        self.hint_text.enabled = True
