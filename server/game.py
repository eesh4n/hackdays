"""
The game itself: pygame render loop, pseudo-3D lane/obstacle rendering,
collision rules, scoring. Reads Player A's live state and Player B's
pending spawn events out of GameState every frame -- doesn't know or
care that one comes from a local camera thread and the other from a
network thread.

Pseudo-3D perspective (no 3D engine, just projection math on top of
pygame's 2D drawing): the 3 lanes are modeled as running away from the
camera along a world-space depth axis `z` (0 = right at the player /
the collision plane, Z_FAR = the horizon where obstacles spawn).
Everything -- lane x-position, obstacle size, vertical high/low offset,
the road cross-ties -- is scaled by the same `scale(z)` projection
factor, so the whole scene converges toward one vanishing point instead
of each element being faked independently. Obstacle and player art
itself is procedurally drawn vector art (sprites.py), not flat blocks.

Obstacle avoidance rules (the actual "game logic" B is trying to beat A
with) -- each type sits at a different height, so a different action
clears it:
    "high"   a beam hanging from the roof, near the top of the lane
             -> avoided only by DUCK (jumping would put you into it)
    "medium" a chest-height roller barrel, centered in the lane
             -> avoided by EITHER jump or duck (low enough to clear by
                jumping, high enough to clear by ducking)
    "low"    a ground-level hurdle
             -> avoided only by JUMP (ducking would put you into it)
"""
import pygame

import sprites

SCREEN_W, SCREEN_H = 900, 600
LANE_COUNT = 3
FPS = 60

# --- Perspective / camera model --------------------------------------
#
# World-space depth z: 0 is the collision plane (right in front of the
# player), Z_FAR is where obstacles spawn (the horizon). CAMERA_D is a
# focal-length-like constant controlling how quickly things shrink with
# distance -- smaller = more dramatic (fisheye-ish) convergence.
Z_NEAR = 0.0
Z_FAR = 18.0
CAMERA_D = 5.0

HORIZON_Y = int(SCREEN_H * 0.32)
NEAR_Y = SCREEN_H - 90          # the collision plane on screen
HORIZON_TRACK_WIDTH = 50
NEAR_TRACK_WIDTH = 620

_SCALE_AT_FAR = CAMERA_D / (CAMERA_D + Z_FAR)


def _scale(z):
    """1.0 right at the camera, shrinking toward _SCALE_AT_FAR at Z_FAR."""
    z = max(Z_NEAR, min(z, Z_FAR))
    return CAMERA_D / (CAMERA_D + z)


def _norm(z):
    """scale(z) renormalized to a clean 0..1 range across [Z_FAR, Z_NEAR],
    so every other projected quantity (position, size) can just lerp
    between its "horizon" and "near" value with this one number."""
    s = _scale(z)
    return (s - _SCALE_AT_FAR) / (1.0 - _SCALE_AT_FAR)


def _lerp(far_value, near_value, t):
    return far_value + (near_value - far_value) * t


def screen_y(z):
    return _lerp(HORIZON_Y, NEAR_Y, _norm(z))


def track_half_width(z):
    return _lerp(HORIZON_TRACK_WIDTH / 2, NEAR_TRACK_WIDTH / 2, _norm(z))


def lane_center_x(lane, z=Z_NEAR):
    half_w = track_half_width(z)
    lane_w = (half_w * 2) / LANE_COUNT
    return SCREEN_W / 2 + (lane - 1) * lane_w


def lane_boundary_x(boundary_index, z):
    """boundary_index 0/1/2 = left edge / lane1|2 divider / lane2|3 divider
    / right edge, i.e. call with 0..LANE_COUNT."""
    half_w = track_half_width(z)
    lane_w = (half_w * 2) / LANE_COUNT
    return SCREEN_W / 2 - half_w + boundary_index * lane_w


PLAYER_SIZE = 60
PLAYER_DUCK_SIZE = 30
PLAYER_JUMP_LIFT = 55

OBSTACLE_WORLD_SPEED_START = 3.2   # world units/sec, how fast z decreases
OBSTACLE_WORLD_SPEED_RAMP = 0.09   # added per second survived
COLLISION_Z_BAND = 0.9             # world-units window around Z_NEAR to check a hit
REMOVE_Z = -1.5                    # obstacle fully passed the player, stop drawing/tracking it

# Obstacle vertical placement, at z=0 (right in front of the player):
# height of the sprite, and its offset from the lane centerline.
# high sits near the top of the lane (duck under it), low sits near the
# ground (jump over it), medium is centered at chest height (clearable
# either way).
NEAR_OBSTACLE_HEIGHT_PX = {"high": 46, "medium": 42, "low": 40}
NEAR_OBSTACLE_Y_OFFSET_PX = {"high": -80, "medium": 0, "low": 58}
MIN_OBSTACLE_PX = 6
OBSTACLE_WIDTH_FRACTION = 0.72  # of the lane's width at that depth

SKY_TOP_COLOR = (18, 14, 42)
SKY_BOTTOM_COLOR = (64, 40, 74)
GROUND_COLOR = (26, 24, 34)
TRACK_COLOR = (42, 40, 55)
TRACK_EDGE_COLOR = (120, 110, 150)
LANE_LINE_COLOR = (90, 85, 115)
TIE_COLOR = (60, 56, 78)
SCENERY_COLOR = (30, 26, 44)
HORIZON_GLOW_COLOR = (255, 170, 120)
GAMEOVER_COLOR = (255, 80, 80)
TEXT_COLOR = (235, 235, 240)
PANEL_COLOR = (20, 18, 30, 170)

ROAD_TIE_SPACING_Z = 2.2
SCENERY_SPACING_Z = 4.5


def avoided(action, obstacle_kind):
    if obstacle_kind == "high":
        return action == "duck"
    if obstacle_kind == "low":
        return action == "jump"
    if obstacle_kind == "medium":
        return action in ("jump", "duck")
    return False


class Obstacle:
    __slots__ = ("lane", "kind", "z")

    def __init__(self, lane, kind):
        self.lane = lane
        self.kind = kind
        self.z = Z_FAR

    def rect(self):
        t = _norm(self.z)
        h = max(MIN_OBSTACLE_PX, NEAR_OBSTACLE_HEIGHT_PX[self.kind] * t)
        lane_w = (track_half_width(self.z) * 2) / LANE_COUNT
        w = max(MIN_OBSTACLE_PX, lane_w * OBSTACLE_WIDTH_FRACTION)
        cx = lane_center_x(self.lane, self.z)
        cy = screen_y(self.z) + NEAR_OBSTACLE_Y_OFFSET_PX[self.kind] * t
        return pygame.Rect(cx - w / 2, cy - h / 2, w, h)


class Game:
    def __init__(self, game_state):
        self.game_state = game_state
        pygame.init()
        pygame.display.set_caption("Hackdays -- Player A vs Player B")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 32)
        self.big_font = pygame.font.SysFont(None, 64)

        self._sky = sprites.build_vertical_gradient(SCREEN_W, HORIZON_Y + 2, SKY_TOP_COLOR, SKY_BOTTOM_COLOR)

        self._lane_a = 1
        self._action_a = "run"
        self._pose_visible_a = False

        self.reset()

    def reset(self):
        self.obstacles = []
        self.score = 0.0
        self.survived_sec = 0.0
        self.game_over = False
        self.game_state.reset()

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r and self.game_over:
                        self.reset()

            if not self.game_over:
                self.update(dt)
            self.draw()
            pygame.display.flip()

        pygame.quit()

    def update(self, dt):
        self.survived_sec += dt
        self.score += dt * 10
        speed = OBSTACLE_WORLD_SPEED_START + OBSTACLE_WORLD_SPEED_RAMP * self.survived_sec

        for event in self.game_state.drain_spawn_events():
            self.obstacles.append(Obstacle(event["lane"], event["obstacle"]))

        self._lane_a, self._action_a, self._pose_visible_a = self.game_state.get_player_a()

        still_alive = []
        for obstacle in self.obstacles:
            obstacle.z -= speed * dt

            if abs(obstacle.z - Z_NEAR) <= COLLISION_Z_BAND and obstacle.lane == self._lane_a:
                if not avoided(self._action_a, obstacle.kind):
                    self.game_over = True

            if obstacle.z > REMOVE_Z:
                still_alive.append(obstacle)
        self.obstacles = still_alive

    def draw(self):
        self._draw_scene()

        # Obstacles farthest-first so nearer ones draw on top (painter's algorithm).
        for obstacle in sorted(self.obstacles, key=lambda o: -o.z):
            rect = obstacle.rect()
            sprite = sprites.get_obstacle_sprite(obstacle.kind, rect.width, rect.height)
            self.screen.blit(sprite, rect.topleft)

        self._draw_player()
        self._draw_hud()

        if self.game_over:
            self._draw_game_over()

    def _draw_scene(self):
        self.screen.blit(self._sky, (0, 0))
        pygame.draw.rect(self.screen, GROUND_COLOR, (0, HORIZON_Y, SCREEN_W, SCREEN_H - HORIZON_Y))

        # Soft glow behind the vanishing point for atmosphere.
        glow_r = 90
        glow = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        for r in range(glow_r, 0, -6):
            alpha = int(50 * (1 - r / glow_r))
            pygame.draw.circle(glow, (*HORIZON_GLOW_COLOR, alpha), (glow_r, glow_r), r)
        self.screen.blit(glow, (SCREEN_W // 2 - glow_r, HORIZON_Y - glow_r), special_flags=pygame.BLEND_RGBA_ADD)

        self._draw_scenery()

        far_l, far_r = lane_boundary_x(0, Z_FAR), lane_boundary_x(LANE_COUNT, Z_FAR)
        near_l, near_r = lane_boundary_x(0, Z_NEAR), lane_boundary_x(LANE_COUNT, Z_NEAR)
        pygame.draw.polygon(self.screen, TRACK_COLOR, [
            (far_l, HORIZON_Y), (far_r, HORIZON_Y), (near_r, NEAR_Y), (near_l, NEAR_Y),
        ])
        pygame.draw.line(self.screen, TRACK_EDGE_COLOR, (far_l, HORIZON_Y), (near_l, NEAR_Y), 3)
        pygame.draw.line(self.screen, TRACK_EDGE_COLOR, (far_r, HORIZON_Y), (near_r, NEAR_Y), 3)

        # Soft highlight glow under the player's current lane -- makes it
        # obvious at a glance which lane is "yours".
        cur_l = lane_boundary_x(self._lane_a, Z_NEAR)
        cur_r = lane_boundary_x(self._lane_a + 1, Z_NEAR)
        cur_fl = lane_boundary_x(self._lane_a, Z_FAR * 0.35)
        cur_fr = lane_boundary_x(self._lane_a + 1, Z_FAR * 0.35)
        glow_poly = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        pygame.draw.polygon(glow_poly, (80, 200, 255, 35), [
            (cur_fl, screen_y(Z_FAR * 0.35)), (cur_fr, screen_y(Z_FAR * 0.35)), (cur_r, NEAR_Y), (cur_l, NEAR_Y),
        ])
        self.screen.blit(glow_poly, (0, 0))

        for i in (1, 2):
            pygame.draw.line(
                self.screen, LANE_LINE_COLOR,
                (lane_boundary_x(i, Z_FAR), HORIZON_Y),
                (lane_boundary_x(i, Z_NEAR), NEAR_Y), 2,
            )

        self._draw_road_ties()

    def _draw_road_ties(self):
        """Cross-lines that scroll toward the camera as the game runs --
        the main "we are moving forward" cue, at the same world speed as
        the obstacles so the depth cue stays consistent with gameplay."""
        speed = OBSTACLE_WORLD_SPEED_START + OBSTACLE_WORLD_SPEED_RAMP * self.survived_sec
        phase = (self.survived_sec * speed) % ROAD_TIE_SPACING_Z
        z = -phase
        while z <= Z_FAR:
            if z >= Z_NEAR:
                y = screen_y(z)
                l = lane_boundary_x(0, z)
                r = lane_boundary_x(LANE_COUNT, z)
                pygame.draw.line(self.screen, TIE_COLOR, (l, y), (r, y), 1)
            z += ROAD_TIE_SPACING_Z

    def _draw_scenery(self):
        """Simple converging pillars along both sides of the track so
        the world doesn't feel like an empty void off-track."""
        speed = OBSTACLE_WORLD_SPEED_START + OBSTACLE_WORLD_SPEED_RAMP * self.survived_sec
        phase = (self.survived_sec * speed) % SCENERY_SPACING_Z
        z = -phase
        while z <= Z_FAR:
            if z >= Z_NEAR:
                t = _norm(z)
                y = screen_y(z)
                pillar_h = _lerp(6, 70, t)
                pillar_w = _lerp(3, 18, t)
                margin = _lerp(4, 30, t)
                for side_x in (lane_boundary_x(0, z) - margin, lane_boundary_x(LANE_COUNT, z) + margin):
                    rect = pygame.Rect(0, 0, pillar_w, pillar_h)
                    rect.midbottom = (side_x, y)
                    pygame.draw.rect(self.screen, SCENERY_COLOR, rect, border_radius=2)
            z += SCENERY_SPACING_Z

    def _draw_player(self):
        x = lane_center_x(self._lane_a, Z_NEAR)
        base_y = NEAR_Y

        if self._action_a == "duck":
            size = PLAYER_DUCK_SIZE
            y = base_y + (PLAYER_SIZE - PLAYER_DUCK_SIZE) // 2
        elif self._action_a == "jump":
            size = PLAYER_SIZE
            y = base_y - PLAYER_JUMP_LIFT
        else:
            size = PLAYER_SIZE
            y = base_y

        # Ground shadow stays on the track regardless of jump height --
        # reads as "how far off the ground you currently are".
        shadow_w = PLAYER_SIZE * 1.05
        shadow_h = PLAYER_SIZE * 0.24
        shadow = pygame.Surface((int(shadow_w), int(shadow_h)), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 90), shadow.get_rect())
        self.screen.blit(shadow, (x - shadow_w / 2, base_y + PLAYER_SIZE * 0.32 - shadow_h / 2))

        sprites.draw_player(self.screen, x, y, size, self._action_a)

    def _draw_hud(self):
        panel = pygame.Surface((190, 46), pygame.SRCALPHA)
        pygame.draw.rect(panel, PANEL_COLOR, panel.get_rect(), border_radius=10)
        self.screen.blit(panel, (10, 10))
        score_surf = self.font.render(f"Score: {int(self.score)}", True, TEXT_COLOR)
        self.screen.blit(score_surf, (24, 22))

        if not self._pose_visible_a:
            warn = self.font.render("Player A pose not detected -- using last known state", True, (255, 200, 80))
            self.screen.blit(warn, (10, SCREEN_H - 30))

    def _draw_game_over(self):
        dim = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 120))
        self.screen.blit(dim, (0, 0))

        panel = pygame.Surface((360, 140), pygame.SRCALPHA)
        pygame.draw.rect(panel, (20, 18, 30, 210), panel.get_rect(), border_radius=16)
        pygame.draw.rect(panel, GAMEOVER_COLOR, panel.get_rect(), width=2, border_radius=16)
        self.screen.blit(panel, (SCREEN_W // 2 - 180, SCREEN_H // 2 - 70))

        over_surf = self.big_font.render("GAME OVER", True, GAMEOVER_COLOR)
        hint_surf = self.font.render("Press R to restart, Q to quit", True, TEXT_COLOR)
        self.screen.blit(over_surf, over_surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 20)))
        self.screen.blit(hint_surf, hint_surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 30)))
