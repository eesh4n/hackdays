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
of each element being faked independently.

Obstacle avoidance rules (the actual "game logic" B is trying to beat A
with):
    "high"   obstacle sits up high (like a bar across the lane)
             -> avoided only by DUCK
    "low"    obstacle sits on the ground (like a hurdle)
             -> avoided only by JUMP
    "medium" obstacle fills the whole lane height
             -> can't be avoided by jump or duck at all -- the only
                escape is not being in that lane when it arrives
This gives B two ways to threaten A: force a reflex check (high/low) or
force a lane change (medium).
"""
import pygame

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

NEAR_OBSTACLE_HEIGHT_PX = {"high": 40, "medium": 90, "low": 34}
NEAR_OBSTACLE_Y_OFFSET_PX = {"high": -85, "medium": 0, "low": 55}  # from lane centerline, at z=0
MIN_OBSTACLE_PX = 6
OBSTACLE_WIDTH_FRACTION = 0.72  # of the lane's width at that depth

SKY_COLOR = (14, 14, 22)
GROUND_COLOR = (26, 24, 34)
TRACK_COLOR = (40, 38, 52)
TRACK_EDGE_COLOR = (90, 88, 110)
LANE_LINE_COLOR = (70, 68, 88)
TIE_COLOR = (55, 53, 70)
PLAYER_COLOR = (80, 200, 255)
PLAYER_SHADE_COLOR = (45, 130, 175)
GAMEOVER_COLOR = (255, 80, 80)
OBSTACLE_COLORS = {"high": (255, 180, 60), "medium": (200, 80, 220), "low": (255, 90, 90)}
TEXT_COLOR = (230, 230, 230)

ROAD_TIE_SPACING_Z = 2.2
ROAD_TIE_COUNT = 10


def avoided(action, obstacle_kind):
    if obstacle_kind == "high":
        return action == "duck"
    if obstacle_kind == "low":
        return action == "jump"
    return False  # "medium" is never avoided by an action, only by lane


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
            pygame.draw.rect(self.screen, OBSTACLE_COLORS[obstacle.kind], obstacle.rect(), border_radius=4)

        self._draw_player()

        score_surf = self.font.render(f"Score: {int(self.score)}", True, TEXT_COLOR)
        self.screen.blit(score_surf, (10, 10))

        if not self._pose_visible_a:
            warn = self.font.render("Player A pose not detected -- using last known state", True, (255, 200, 80))
            self.screen.blit(warn, (10, SCREEN_H - 30))

        if self.game_over:
            over_surf = self.big_font.render("GAME OVER", True, GAMEOVER_COLOR)
            hint_surf = self.font.render("Press R to restart, Q to quit", True, TEXT_COLOR)
            self.screen.blit(over_surf, over_surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 20)))
            self.screen.blit(hint_surf, hint_surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 30)))

    def _draw_scene(self):
        self.screen.fill(SKY_COLOR)
        pygame.draw.rect(self.screen, GROUND_COLOR, (0, HORIZON_Y, SCREEN_W, SCREEN_H - HORIZON_Y))

        # Track surface: a trapezoid from the horizon down to the near edge.
        far_l, far_r = lane_boundary_x(0, Z_FAR), lane_boundary_x(LANE_COUNT, Z_FAR)
        near_l, near_r = lane_boundary_x(0, Z_NEAR), lane_boundary_x(LANE_COUNT, Z_NEAR)
        pygame.draw.polygon(self.screen, TRACK_COLOR, [
            (far_l, HORIZON_Y), (far_r, HORIZON_Y), (near_r, NEAR_Y), (near_l, NEAR_Y),
        ])
        pygame.draw.line(self.screen, TRACK_EDGE_COLOR, (far_l, HORIZON_Y), (near_l, NEAR_Y), 2)
        pygame.draw.line(self.screen, TRACK_EDGE_COLOR, (far_r, HORIZON_Y), (near_r, NEAR_Y), 2)

        # Lane divider lines, converging toward the horizon.
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

        # A short shaded "shadow" block behind the main body reads as a
        # cheap sense of depth/thickness without needing real 3D geometry.
        shade = pygame.Rect(0, 0, size, size)
        shade.center = (x + 6, y + 6)
        pygame.draw.rect(self.screen, PLAYER_SHADE_COLOR, shade, border_radius=8)

        rect = pygame.Rect(0, 0, size, size)
        rect.center = (x, y)
        pygame.draw.rect(self.screen, PLAYER_COLOR, rect, border_radius=8)
