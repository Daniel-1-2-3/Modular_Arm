import pygame
import math
import numpy as np
from calc_bearing import CalcBearing

class Car:
    def __init__(self, name, pos_px, px_per_cm, angle_rad=0.0, white_side=1):
        self.name = name
        self.x, self.y = float(pos_px[0]), float(pos_px[1])
        self.angle = float(angle_rad)

        self.px_per_cm = float(px_per_cm)

        self.width_cm = 20.0
        self.length_cm = 20.0

        self.width_px = self.width_cm * self.px_per_cm
        self.length_px = self.length_cm * self.px_per_cm

        self.dot_back_from_front_cm = 7.0
        self.white_offset_cm = 1.9
        self.white_side = white_side

        self.v = 0.0
        self.omega = 0.0

    def set_speed(self, v_px_s):
        self.v = float(v_px_s)

    def set_angular_velocity(self, omega_rad_s):
        self.omega = float(omega_rad_s)

    def update(self, dt_s):
        half_l = self.length_px / 2.0
        back_px = self.dot_back_from_front_cm * self.px_per_cm
        pivot_local = (half_l - back_px, 0.0)

        c0 = math.cos(self.angle)
        s0 = math.sin(self.angle)

        pivot_world_x = self.x + (c0 * pivot_local[0] - s0 * pivot_local[1])
        pivot_world_y = self.y + (s0 * pivot_local[0] + c0 * pivot_local[1])

        self.angle += self.omega * dt_s
        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi

        c1 = math.cos(self.angle)
        s1 = math.sin(self.angle)

        self.x = pivot_world_x - (c1 * pivot_local[0] - s1 * pivot_local[1])
        self.y = pivot_world_y - (s1 * pivot_local[0] + c1 * pivot_local[1])

        self.x += math.cos(self.angle) * self.v * dt_s
        self.y += math.sin(self.angle) * self.v * dt_s

    def _corners_world(self):
        half_l = self.length_px / 2.0
        half_w = self.width_px / 2.0

        local = [
            (+half_l, -half_w),
            (+half_l, +half_w),
            (-half_l, +half_w),
            (-half_l, -half_w),
        ]

        c = math.cos(self.angle)
        s = math.sin(self.angle)

        corners = []
        for lx, ly in local:
            rx = c * lx - s * ly
            ry = s * lx + c * ly
            corners.append((self.x + rx, self.y + ry))

        return corners

    def _red_dot_world(self):
        half_l = self.length_px / 2.0
        back_px = self.dot_back_from_front_cm * self.px_per_cm
        lx, ly = (half_l - back_px, 0.0)

        c = math.cos(self.angle)
        s = math.sin(self.angle)

        return (self.x + (c * lx - s * ly), self.y + (s * lx + c * ly))

    def _white_dot_world(self):
        half_l = self.length_px / 2.0
        offset_px = self.white_offset_cm * self.px_per_cm * self.white_side
        lx, ly = (half_l, offset_px)

        c = math.cos(self.angle)
        s = math.sin(self.angle)

        return (self.x + (c * lx - s * ly), self.y + (s * lx + c * ly))

    def draw(self, surface, body_color, red_color):
        pts = [(int(px), int(py)) for (px, py) in self._corners_world()]
        pygame.draw.polygon(surface, body_color, pts, width=0)

        rx, ry = self._red_dot_world()
        pygame.draw.circle(surface, red_color, (int(rx), int(ry)), 5)

        wx, wy = self._white_dot_world()
        pygame.draw.circle(surface, (255, 255, 255), (int(wx), int(wy)), 5)

def draw_grid(surface, W, H, spacing, color_major, color_minor):
    for x in range(0, W + 1, spacing):
        col = color_major if x % (spacing * 5) == 0 else color_minor
        pygame.draw.line(surface, col, (x, 0), (x, H), 1)

    for y in range(0, H + 1, spacing):
        col = color_major if y % (spacing * 5) == 0 else color_minor
        pygame.draw.line(surface, col, (0, y), (W, y), 1)

def get_center_from_white_dot_px(car):
    wx, wy = car._white_dot_world()

    half_l = car.length_px / 2.0
    offset_px = car.white_offset_cm * car.px_per_cm * car.white_side
    lx, ly = (half_l, offset_px)

    c = math.cos(car.angle)
    s = math.sin(car.angle)

    dx = c * lx - s * ly
    dy = s * lx + c * ly

    return (wx - dx, wy - dy)

def simulate_aruco_rvec_tvec(car_stationary, car_movable, px_per_cm):
    px_per_m = px_per_cm * 100.0

    cam_px = np.array(car_movable._white_dot_world(), dtype=np.float64)
    tag_px = np.array(car_stationary._white_dot_world(), dtype=np.float64)

    cam_m = cam_px / px_per_m
    tag_m = tag_px / px_per_m

    theta_cam = car_movable.angle
    theta_tag = car_stationary.angle

    forward = np.array([math.cos(theta_cam), math.sin(theta_cam)], dtype=np.float64)
    right = np.array([-math.sin(theta_cam), math.cos(theta_cam)], dtype=np.float64)

    rel = tag_m - cam_m

    tx = float(np.dot(right, rel))
    tz = float(np.dot(forward, rel))
    ty = 0.0

    yaw_rel = (theta_tag - theta_cam + math.pi) % (2 * math.pi) - math.pi
    rvec = (0.0, yaw_rel, 0.0)

    return rvec, (tx, ty, tz)

def draw_bearing_visual(surface, font, origin_px, heading_rad, target_px, bearing, color):
    cx, cy = origin_px

    forward_len = 80
    fx = cx + math.cos(heading_rad) * forward_len
    fy = cy + math.sin(heading_rad) * forward_len
    pygame.draw.line(surface, color, (cx, cy), (fx, fy), 3)

    tx, ty = target_px
    pygame.draw.line(surface, color, (cx, cy), (tx, ty), 2)

    radius = 60
    mid_a = heading_rad + 0.5 * bearing
    lx = cx + math.cos(mid_a) * (radius + 10)
    ly = cy + math.sin(mid_a) * (radius + 10)

    deg = math.degrees(bearing)
    text = font.render(f"{deg:+.1f}°", True, color)
    surface.blit(text, (int(lx), int(ly)))

pygame.init()

W, H = 900, 600
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)

px_per_cm = 6
car_static = Car("A", (300, 300), px_per_cm, 0.0, white_side=-1)
car_move = Car("B", (600, 300), px_per_cm, math.pi, white_side=1)
speed = 220
rot_speed = math.radians(120)
L_m = car_move.length_cm / 100.0

center_to_camera_cam_m = (
    +L_m / 2.0,
    0.0,
    (car_move.white_offset_cm / 100.0) * car_move.white_side,
)

tag_to_center_tag_m = (
    -L_m / 2.0,
    0.0,
    -(car_static.white_offset_cm / 100.0) * car_static.white_side,
)

running = True
while running:
    dt = clock.tick(60) / 1000.0

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    v = 0.0
    omega = 0.0

    if keys[pygame.K_w]:
        v += speed
    if keys[pygame.K_s]:
        v -= speed
    if keys[pygame.K_a]:
        omega += rot_speed
    if keys[pygame.K_d]:
        omega -= rot_speed

    car_move.set_speed(v)
    car_move.set_angular_velocity(omega)
    car_move.update(dt)

    rvec, tvec = simulate_aruco_rvec_tvec(car_static, car_move, px_per_cm)

    bearing_cam = CalcBearing.get_camera_bearing_from_tvec(tvec)
    bearing_real = CalcBearing.get_bearing(rvec, tvec)

    screen.fill((244, 246, 250))
    draw_grid(screen, W, H, 24, (210, 215, 225), (228, 232, 240))

    car_static.draw(screen, (100, 100, 100), (235, 60, 60))
    car_move.draw(screen, (100, 100, 100), (235, 60, 60))

    cam_origin = car_move._white_dot_world()
    tag_target = car_static._white_dot_world()

    mov_center = get_center_from_white_dot_px(car_move)
    sta_center = get_center_from_white_dot_px(car_static)

    draw_bearing_visual(screen, font, cam_origin, car_move.angle, tag_target, bearing_cam, (235, 60, 60))
    draw_bearing_visual(screen, font, mov_center, car_move.angle, sta_center, bearing_real, (0, 120, 255))

    pygame.display.flip()

pygame.quit()