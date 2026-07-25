"""
Procedurally-drawn art for obstacles and reusable gradients -- built
entirely from pygame primitives (no external image files), so there's
nothing to download, license, or go missing on a teammate's laptop.
Everything scales cleanly to any pixel size via pygame.transform since
the perspective projection in game.py needs obstacles to grow smoothly
as they approach the camera.

Each obstacle kind is drawn once at a reference resolution (MASTER_SIZE)
and cached; get_obstacle_sprite() then scales + caches per requested
size (rounded to a small bucket) so the per-frame perspective resize in
game.py doesn't redraw vector art every frame for every obstacle.
"""
import pygame

MASTER_SIZE = (240, 140)  # (w, h) reference canvas all master art is drawn on
SIZE_BUCKET_PX = 4        # round requested sizes to this many pixels for cache reuse

_master_cache = {}
_scaled_cache = {}


def _striped_polygon_band(surf, rect, color_a, color_b, stripe_w=20):
    """Diagonal hazard-stripe fill, clipped to `rect`."""
    surf.set_clip(rect)
    x = rect.left - rect.height
    i = 0
    while x < rect.right + rect.height:
        color = color_a if i % 2 == 0 else color_b
        pygame.draw.polygon(surf, color, [
            (x, rect.bottom), (x + rect.height, rect.top),
            (x + rect.height + stripe_w, rect.top), (x + stripe_w, rect.bottom),
        ])
        x += stripe_w
        i += 1
    surf.set_clip(None)


def _build_high():
    """Overhead hazard beam, hanging from the "roof" on two chains --
    reads instantly as "duck under this"."""
    w, h = MASTER_SIZE
    surf = pygame.Surface((w, h), pygame.SRCALPHA)

    beam_h = int(h * 0.5)
    beam_rect = pygame.Rect(0, (h - beam_h) // 2, w, beam_h)

    chain_color = (95, 95, 105)
    for cx in (w * 0.22, w * 0.78):
        pygame.draw.line(surf, chain_color, (cx, 0), (cx, beam_rect.top + 4), 5)
        pygame.draw.circle(surf, chain_color, (int(cx), 4), 5)

    pygame.draw.rect(surf, (35, 35, 40), beam_rect, border_radius=10)
    _striped_polygon_band(surf, beam_rect, (255, 196, 30), (35, 35, 40))
    pygame.draw.rect(surf, (255, 255, 255, 55), beam_rect.inflate(-10, -int(beam_h * 0.6)), border_radius=6)
    pygame.draw.rect(surf, (0, 0, 0, 110), beam_rect, width=3, border_radius=10)
    return surf


def _build_medium():
    """Chest-height wooden roller barrel -- low enough to jump over,
    high enough to duck under."""
    w, h = MASTER_SIZE
    surf = pygame.Surface((w, h), pygame.SRCALPHA)

    body_h = int(h * 0.6)
    body_rect = pygame.Rect(0, (h - body_h) // 2, w, body_h)

    pygame.draw.rect(surf, (150, 92, 52), body_rect, border_radius=body_h // 2)
    for i in (1, 2, 3):
        by = body_rect.top + body_h * i // 4
        pygame.draw.line(surf, (108, 64, 34), (12, by), (w - 12, by), 3)

    cap_w = int(w * 0.09)
    for cx in (0, w - cap_w):
        pygame.draw.rect(surf, (100, 100, 108), (cx, body_rect.top, cap_w, body_h), border_radius=body_h // 2)
        pygame.draw.rect(surf, (100, 100, 108), (cx, body_rect.top, cap_w, body_h), width=2, border_radius=body_h // 2)

    pygame.draw.rect(surf, (255, 255, 255, 45), body_rect.inflate(0, -int(body_h * 0.65)), border_radius=body_h // 4)
    pygame.draw.rect(surf, (0, 0, 0, 90), body_rect, width=3, border_radius=body_h // 2)
    return surf


def _build_low():
    """Ground-level hurdle bar on two legs -- jump over it."""
    w, h = MASTER_SIZE
    surf = pygame.Surface((w, h), pygame.SRCALPHA)

    bar_h = int(h * 0.26)
    bar_rect = pygame.Rect(0, h - bar_h - int(h * 0.18), w, bar_h)

    _striped_polygon_band(surf, bar_rect, (230, 60, 60), (245, 245, 245))
    pygame.draw.rect(surf, (0, 0, 0, 120), bar_rect, width=3, border_radius=6)

    leg_w = int(w * 0.06)
    for lx in (int(w * 0.12), int(w * 0.88) - leg_w):
        pygame.draw.polygon(surf, (75, 75, 82), [
            (lx, bar_rect.bottom), (lx + leg_w, bar_rect.bottom),
            (lx + leg_w + 8, h), (lx - 8, h),
        ])
    return surf


_BUILDERS = {"high": _build_high, "medium": _build_medium, "low": _build_low}


def _master(kind):
    if kind not in _master_cache:
        _master_cache[kind] = _BUILDERS[kind]()
    return _master_cache[kind]


def get_obstacle_sprite(kind, width_px, height_px):
    w = max(4, (int(width_px) // SIZE_BUCKET_PX) * SIZE_BUCKET_PX)
    h = max(4, (int(height_px) // SIZE_BUCKET_PX) * SIZE_BUCKET_PX)
    key = (kind, w, h)
    scaled = _scaled_cache.get(key)
    if scaled is None:
        scaled = pygame.transform.smoothscale(_master(kind), (w, h))
        _scaled_cache[key] = scaled
    return scaled


def build_vertical_gradient(width, height, top_color, bottom_color):
    """A simple top-to-bottom color lerp, pre-rendered once (e.g. for a
    sky) rather than per-pixel every frame."""
    surf = pygame.Surface((width, height))
    for y in range(height):
        t = y / max(1, height - 1)
        color = tuple(int(top_color[i] + (bottom_color[i] - top_color[i]) * t) for i in range(3))
        pygame.draw.line(surf, color, (0, y), (width, y))
    return surf


def draw_player(surface, x, y, size, action):
    """A small humanoid silhouette instead of a flat block -- head,
    torso, and a pose per action so jump/duck/run read at a glance."""
    body_color = (80, 200, 255)
    shade_color = (35, 110, 150)
    outline_color = (15, 55, 85)

    head_r = max(4, int(size * 0.22))

    if action == "duck":
        torso_w, torso_h = int(size * 1.15), int(size * 0.55)
        torso_rect = pygame.Rect(0, 0, torso_w, torso_h)
        torso_rect.center = (x, y + int(size * 0.15))
        head_center = (x + int(size * 0.35), torso_rect.top - head_r * 0.6)
    elif action == "jump":
        torso_w, torso_h = int(size * 0.62), int(size * 0.68)
        torso_rect = pygame.Rect(0, 0, torso_w, torso_h)
        torso_rect.center = (x, y + int(size * 0.05))
        head_center = (x, torso_rect.top - head_r * 0.7)
    else:  # run
        torso_w, torso_h = int(size * 0.5), int(size * 0.78)
        torso_rect = pygame.Rect(0, 0, torso_w, torso_h)
        torso_rect.center = (x, y + int(size * 0.05))
        head_center = (x, torso_rect.top - head_r * 0.7)

    # Legs, drawn behind the torso.
    hip = (torso_rect.centerx, torso_rect.bottom - 2)
    if action == "run":
        stride = size * 0.28
        foot_y = y + size // 2
        pygame.draw.line(surface, shade_color, hip, (hip[0] - stride, foot_y), 8)
        pygame.draw.line(surface, shade_color, hip, (hip[0] + stride * 0.6, foot_y), 8)
    elif action == "jump":
        pygame.draw.line(surface, shade_color, hip, (hip[0] - size * 0.22, hip[1] + size * 0.3), 8)
        pygame.draw.line(surface, shade_color, hip, (hip[0] + size * 0.22, hip[1] + size * 0.3), 8)
    else:  # duck -- legs tucked, barely visible under the flattened torso
        pygame.draw.line(surface, shade_color, hip, (hip[0] - size * 0.3, hip[1] + size * 0.12), 8)
        pygame.draw.line(surface, shade_color, hip, (hip[0] + size * 0.3, hip[1] + size * 0.12), 8)

    pygame.draw.rect(surface, body_color, torso_rect, border_radius=int(min(torso_w, torso_h) * 0.35))
    pygame.draw.rect(surface, outline_color, torso_rect, width=2, border_radius=int(min(torso_w, torso_h) * 0.35))

    pygame.draw.circle(surface, body_color, (int(head_center[0]), int(head_center[1])), head_r)
    pygame.draw.circle(surface, outline_color, (int(head_center[0]), int(head_center[1])), head_r, width=2)
