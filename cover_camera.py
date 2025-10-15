import pygame
import numpy as np
import sys
import math
import os
import datetime

# Quaternion class (from book 4.8: q = s + i x + j y + k z)
class Quaternion:
    def __init__(self, s=1.0, v=np.zeros(3)):
        self.s = s
        self.v = v

    def __mul__(self, other):
        s1, v1 = self.s, self.v
        s2, v2 = other.s, other.v
        s = s1 * s2 - np.dot(v1, v2)
        v = s1 * v2 + s2 * v1 + np.cross(v1, v2)
        return Quaternion(s, v)

    def conjugate(self):
        return Quaternion(self.s, -self.v)

    def normalize(self):
        norm = np.sqrt(self.s**2 + np.dot(self.v, self.v))
        if norm > 0:
            self.s /= norm
            self.v /= norm

    def to_rotation_matrix(self):
        s, x, y, z = self.s, self.v[0], self.v[1], self.v[2]
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*s*z, 2*x*z + 2*s*y],
            [2*x*y + 2*s*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*s*x],
            [2*x*z - 2*s*y, 2*y*z + 2*s*x, 1 - 2*x**2 - 2*y**2]
        ])

# Vector3 for points (homogeneous as [x,y,z,1])
class Vector3:
    def __init__(self, x, y, z):
        self.data = np.array([x, y, z, 1.0])

# Polygon class for faces
class Polygon:
    def __init__(self, vertices, color):
        self.vertices = vertices  # list of np arrays
        self.color = color

# Cuboid class with faces
class Cuboid:
    def __init__(self, pos, size, color):
        half = np.array(size) / 2
        offsets = [-1, 1]
        verts = [Vector3(pos[0] + ox * half[0], pos[1] + oy * half[1], pos[2] + oz * half[2]).data
                 for ox in offsets for oy in offsets for oz in offsets]
        self.faces = [
            Polygon([verts[0], verts[1], verts[3], verts[2]], color),  # bottom
            Polygon([verts[4], verts[5], verts[7], verts[6]], color),  # top
            Polygon([verts[0], verts[1], verts[5], verts[4]], color),  # front
            Polygon([verts[2], verts[3], verts[7], verts[6]], color),  # back
            Polygon([verts[0], verts[2], verts[6], verts[4]], color),  # left
            Polygon([verts[1], verts[3], verts[7], verts[5]], color),  # right
        ]
        self.edges = [(0,1), (1,3), (3,2), (2,0), (4,5), (5,7), (7,6), (6,4), (0,4), (1,5), (2,6), (3,7)]
        self.vertices = verts

# Camera class
class Camera:
    def __init__(self, pos=(0, 0, 10), fov=60.0):
        self.pos = np.array(pos, dtype=float)
        self.quat = Quaternion()  # Identity (no rotation)
        self.fov = fov

    def rotate(self, axis, angle_deg):
        angle_rad = math.radians(angle_deg / 2)
        s = math.cos(angle_rad)
        v = math.sin(angle_rad) * axis / np.linalg.norm(axis)
        delta = Quaternion(s, v)
        delta.normalize()
        self.quat = delta * self.quat
        self.quat.normalize()

    def translate(self, delta_local):
        rot_mat = self.quat.to_rotation_matrix()
        delta_world = rot_mat @ delta_local
        self.pos += delta_world

    def get_view_matrix(self):
        rot_mat3 = self.quat.to_rotation_matrix().T  # Inverse rotation (3x3)
        view = np.eye(4, dtype=float)
        view[:3, :3] = rot_mat3
        view[:3, 3] = -rot_mat3 @ self.pos
        return view

# Perspective projection matrix
def perspective_matrix(fov, aspect, near=0.1, far=100.0):
    f = 1.0 / math.tan(math.radians(fov / 2))
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0]
    ])

# Painter's algorithm with average z
def painters_average_z(polygons, view):
    sorted_polygons = sorted(polygons, key=lambda p: sum((view @ v)[2] for v in p.vertices) / len(p.vertices))
    return sorted_polygons

# Painter's algorithm with max z
def painters_max_z(polygons, view):
    sorted_polygons = sorted(polygons, key=lambda p: max((view @ v)[2] for v in p.vertices))
    return sorted_polygons

# Main program
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Virtual Camera with HSR - Zadanie 2a")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 20)
help_visible = True
screenshot_msg = ""
msg_timer = 0.0

camera = Camera(pos=(0, 0, 10), fov=60.0)

# Scene
cuboids = [
    Cuboid((-4, -2, -5), (2, 4, 10), (255, 0, 0)),
    Cuboid((-4, -2, -20), (2, 6, 10), (0, 255, 0)),
    Cuboid((4, -2, -5), (2, 5, 10), (0, 0, 255)),
    Cuboid((4, -2, -20), (2, 4, 10), (255, 255, 0)),
    Cuboid((0, -3, -10), (1, 1, 30), (255, 0, 255))
]

# All polygons
all_polygons = [face for cub in cuboids for face in cub.faces]

running = True
hsr_mode = 0  # 0: off, 1: Painter's average Z, 2: Painter's max Z

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_h:
                help_visible = not help_visible
            if event.key == pygame.K_r:
                camera.pos = np.array((0.0, 0.0, 10.0), dtype=float)
                camera.quat = Quaternion()
                camera.fov = 60.0
            if event.key == pygame.K_SPACE:
                try:
                    folder = os.path.dirname(os.path.abspath(__file__))
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"screenshot_{ts}.png"
                    path = os.path.join(folder, filename)
                    pygame.image.save(screen, path)
                    screenshot_msg = f"Saved: {filename}"
                    msg_timer = 2.5
                except Exception as e:
                    screenshot_msg = f"Save failed: {e}"
                    msg_timer = 3.0
            if event.key == pygame.K_0:
                hsr_mode = 0
            if event.key == pygame.K_1:
                hsr_mode = 1
            if event.key == pygame.K_2:
                hsr_mode = 2

    keys = pygame.key.get_pressed()
    move_speed = 0.2
    rot_speed = 2.0
    fov_speed = 5.0

    forward = np.array([0, 0, -move_speed]) if keys[pygame.K_w] else np.array([0, 0, move_speed]) if keys[pygame.K_s] else np.zeros(3)
    right = np.array([move_speed, 0, 0]) if keys[pygame.K_d] else np.array([-move_speed, 0, 0]) if keys[pygame.K_a] else np.zeros(3)
    up = np.array([0, move_speed, 0]) if keys[pygame.K_e] else np.array([0, -move_speed, 0]) if keys[pygame.K_q] else np.zeros(3)
    camera.translate(forward + right + up)

    if keys[pygame.K_LEFT]: camera.rotate(np.array([0,1,0]), -rot_speed)
    if keys[pygame.K_RIGHT]: camera.rotate(np.array([0,1,0]), rot_speed)
    if keys[pygame.K_UP]: camera.rotate(np.array([1,0,0]), -rot_speed)
    if keys[pygame.K_DOWN]: camera.rotate(np.array([1,0,0]), rot_speed)
    if keys[pygame.K_z]: camera.rotate(np.array([0,0,1]), -rot_speed)
    if keys[pygame.K_x]: camera.rotate(np.array([0,0,1]), rot_speed)

    if keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS]: camera.fov = max(10, camera.fov - fov_speed)
    if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]: camera.fov = min(120, camera.fov + fov_speed)

    screen.fill((0, 0, 0))
    proj = perspective_matrix(camera.fov, width / height)
    view = camera.get_view_matrix()

    if hsr_mode == 0:  # Wireframe
        for cub in cuboids:
            for edge in cub.edges:
                v1 = proj @ (view @ cub.vertices[edge[0]])
                v2 = proj @ (view @ cub.vertices[edge[1]])
                if v1[3] > 0.001 and v2[3] > 0.001:
                    p1 = (int(width / 2 + (v1[0] / v1[3]) * (width / 2)), int(height / 2 - (v1[1] / v1[3]) * (height / 2)))
                    p2 = (int(width / 2 + (v2[0] / v2[3]) * (width / 2)), int(height / 2 - (v2[1] / v2[3]) * (height / 2)))
                    pygame.draw.line(screen, (255, 255, 255), p1, p2)
    else:
        if hsr_mode == 1:
            sorted_poly = painters_average_z(all_polygons, view)
        elif hsr_mode == 2:
            sorted_poly = painters_max_z(all_polygons, view)

        for poly in sorted_poly:
            projected = [proj @ (view @ v) for v in poly.vertices]
            projected_2d = []
            skip = False
            for v in projected:
                if v[3] <= 0.001:
                    skip = True
                    break
                x = int(width / 2 + (v[0] / v[3]) * (width / 2))
                y = int(height / 2 - (v[1] / v[3]) * (height / 2))
                projected_2d.append((x, y))
            if not skip and len(projected_2d) >= 3:
                pygame.draw.polygon(screen, poly.color, projected_2d)

    # HUD and help (same as before)
    if help_visible:
        lines = [
            "Controls:",
            "W/S: forward/back",
            "A/D: left/right",
            "Q/E: down/up",
            "Arrow keys: pitch/yaw",
            "Z/X: roll",
            "+/-: zoom (change FOV)",
            "0/1/2: HSR off/avg Z/max Z",
            "Space: take screenshot",
            "H: toggle this help",
            "R: reset camera",
            "Esc: quit"
        ]
        padding = 8
        line_height = font.get_linesize()
        box_w = 240
        box_h = padding * 2 + line_height * len(lines)
        surf = pygame.Surface((box_w, box_h), flags=pygame.SRCALPHA)
        surf.fill((0, 0, 0, 150))
        for i, txt in enumerate(lines):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            txt_surf = font.render(txt, True, color)
            surf.blit(txt_surf, (padding, padding + i * line_height))
        screen.blit(surf, (10, 10))

    status = f"FOV: {camera.fov:.1f}  Pos: ({camera.pos[0]:.2f}, {camera.pos[1]:.2f}, {camera.pos[2]:.2f})  HSR: {hsr_mode}"
    stat_surf = font.render(status, True, (255, 255, 0))
    screen.blit(stat_surf, (10, height - 40))

    if msg_timer > 0.0 and screenshot_msg:
        msg_surf = font.render(screenshot_msg, True, (0, 255, 0))
        screen.blit(msg_surf, (width // 2 - msg_surf.get_width() // 2, 10))

    pygame.display.flip()
    ms = clock.tick(60)
    if msg_timer > 0.0:
        msg_timer -= ms / 1000.0
        if msg_timer <= 0.0:
            screenshot_msg = ""
            msg_timer = 0.0

pygame.quit()
sys.exit()