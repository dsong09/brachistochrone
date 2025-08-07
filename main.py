import pygame
import numpy as np
from scipy.interpolate import CubicSpline

FPS = 120
START_POINT = (0.0, 0.0)
END_POINT = (9.0, 6.0)
NUM_CONTROL_POINTS = 20
WIDTH = 800
HEIGHT = 600

WHITE = (255, 255, 255)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

X_PADDING = 1.0
Y_PADDING = 1.0
GRAVITY = 9.8

class PathGenerator:
    def __init__(self, start, end, num_points):
        self.start = start
        self.end = end
        self.num_points = num_points

        self.x_control = np.array([])
        self.y_control = np.array([])
        self.spline = None
        self.x_path = np.array([])
        self.y_path = np.array([])
        self.spline_deriv = None

        self.x_brach = np.array([])
        self.y_brach = np.array([])
        self.cumulative_dist_cubic = []
        self.cumulative_dist_brach = []
        self.total_cubic_length = 0.0
        self.total_brach_length = 0.0

        self.generate_brachistochrone()
        self.generate_random_path()

    @staticmethod
    def cumulative_distances(x_points, y_points):
        dx = np.diff(x_points)
        dy = np.diff(y_points)
        distances = np.sqrt(dx**2 + dy**2)
        return np.concatenate([[0.0], np.cumsum(distances)])

    def generate_brachistochrone(self):
        theta_max = 1.8
        a = 6.0 / (1 - np.cos(theta_max))
        t = np.linspace(0, theta_max, 10000)
        x = a * (t - np.sin(t))
        y = a * (1 - np.cos(t))

        x_scale = self.end[0] / x[-1]
        y_scale = self.end[1] / y[-1]

        self.x_brach = x * x_scale
        self.y_brach = y * y_scale
        self.cumulative_dist_brach = self.cumulative_distances(self.x_brach, self.y_brach)
        self.total_brach_length = self.cumulative_dist_brach[-1]

    def generate_random_path(self):
        self.x_control = np.linspace(self.start[0], self.end[0], self.num_points + 2)
        self.y_control = np.random.uniform(0.0, 8.0, self.num_points + 2)
        self.y_control[0] = self.start[1]
        self.y_control[-1] = self.end[1]

        self.spline = CubicSpline(self.x_control, self.y_control)
        self.x_path = np.linspace(self.start[0], self.end[0], 1000)
        self.y_path = self.spline(self.x_path)
        self.spline_deriv = self.spline.derivative()

        self.cumulative_dist_cubic = self.cumulative_distances(self.x_path, self.y_path)
        self.total_cubic_length = self.cumulative_dist_cubic[-1]

    def get_position_from_distance(self, dist, path_type):
        if path_type == "CUBIC":
            cumulative = self.cumulative_dist_cubic
            x_path, y_path = self.x_path, self.y_path
        else:
            cumulative = self.cumulative_dist_brach
            x_path, y_path = self.x_brach, self.y_brach

        idx = int(np.searchsorted(cumulative, dist, side="right")) - 1
        if idx >= len(cumulative) - 1:
            return self.end

        frac = (dist - cumulative[idx]) / (cumulative[idx + 1] - cumulative[idx])
        x = x_path[idx] + frac * (x_path[idx + 1] - x_path[idx])
        y = y_path[idx] + frac * (y_path[idx + 1] - y_path[idx])
        return x, y

    def get_slope_from_distance(self, dist, path_type):
        if path_type == "CUBIC":
            x, _ = self.get_position_from_distance(dist, path_type)
            return np.arctan(self.spline_deriv(x))
        else:
            cumulative = self.cumulative_dist_brach
            idx = np.searchsorted(cumulative, dist, side="right") - 1
            if idx >= len(self.x_brach) - 1:
                idx = len(self.x_brach) - 2
            dx = self.x_brach[idx + 1] - self.x_brach[idx]
            dy = self.y_brach[idx + 1] - self.y_brach[idx]
            return np.arctan2(dy, dx)

def world_to_screen(x, y, scale_x, scale_y, offset_x, offset_y):
    screen_x = int((x - offset_x) * scale_x)
    screen_y = int((y - offset_y) * scale_y)
    return screen_x, screen_y

def animation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("brachistochrone")
    clock = pygame.time.Clock()
    font = pygame.font.Font("fonts/DeterminationSansWebRegular-369X.ttf", 20)

    brach_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    pathgen = PathGenerator(START_POINT, END_POINT, NUM_CONTROL_POINTS)

    x_min = -1.0
    x_max = 10.0
    y_min = -1.0
    y_max = 8.0

    x_range = (x_max - x_min) + 2 * X_PADDING
    y_range = (y_max - y_min) + 2 * Y_PADDING

    scale_x = WIDTH / x_range
    scale_y = HEIGHT / y_range

    offset_x = x_min - X_PADDING
    offset_y = y_min - Y_PADDING

    current_distance = 0.0
    ball_velocity = 0.0
    physics_time = 0.0
    has_reached_end = False
    current_path_type = "CUBIC"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    pathgen.generate_random_path()
                    current_distance = 0.0
                    ball_velocity = 0.0
                    physics_time = 0.0
                    has_reached_end = False
                    current_path_type = "CUBIC"

                elif event.key == pygame.K_t:
                    if current_path_type == "CUBIC":
                        current_path_type = "BRACHISTOCHRONE"
                    else:
                        current_path_type = "CUBIC"
                    current_distance = 0.0
                    ball_velocity = 0.0
                    physics_time = 0.0
                    has_reached_end = False

        if has_reached_end:
            dt = 0
        else:
            dt = 1.0 / FPS
            prev_distance = current_distance
            prev_velocity = ball_velocity

            slope = pathgen.get_slope_from_distance(current_distance, current_path_type)
            acceleration = GRAVITY * np.sin(slope)
            ball_velocity += acceleration * dt

            new_distance = current_distance + ball_velocity * dt
            total_path_length = (pathgen.total_cubic_length if current_path_type == "CUBIC" else pathgen.total_brach_length)

            if new_distance < 0:
                ball_velocity = 0
                physics_time += dt
            elif new_distance > total_path_length:
                distance_to_end = total_path_length - prev_distance
                partial_time = distance_to_end / prev_velocity if prev_velocity > 0 else 0
                current_distance = total_path_length
                has_reached_end = True
                physics_time += partial_time
            else:
                current_distance = new_distance
                physics_time += dt

        ball_x, ball_y = pathgen.get_position_from_distance(current_distance, current_path_type)
        ball_screen_x, ball_screen_y = world_to_screen(ball_x, ball_y, scale_x, scale_y, offset_x, offset_y)

        screen.fill(BLACK)
        screen.blit(brach_surface, (0, 0))

        cubic_points = [(world_to_screen(x, y, scale_x, scale_y, offset_x, offset_y)) for x, y in zip(pathgen.x_path, pathgen.y_path)]
        if len(cubic_points) > 1:
            pygame.draw.lines(screen, WHITE, False, cubic_points, 3)

        spline = CubicSpline(pathgen.x_brach, pathgen.y_brach)
        x_spline = np.linspace(pathgen.x_brach[0], pathgen.x_brach[-1], 500)
        y_spline = spline(x_spline)
        brach_points = [(world_to_screen(x, y, scale_x, scale_y, offset_x, offset_y)) for x, y in zip(x_spline, y_spline)]
        if len(brach_points) > 1:
            pygame.draw.lines(screen, ORANGE, False, brach_points, 3)

        start_screen = world_to_screen(START_POINT[0], START_POINT[1], scale_x, scale_y, offset_x, offset_y)
        end_screen = world_to_screen(END_POINT[0], END_POINT[1], scale_x, scale_y, offset_x, offset_y)
        pygame.draw.circle(screen, WHITE, start_screen, 4)
        pygame.draw.circle(screen, WHITE, end_screen, 4)
        pygame.draw.circle(screen, RED, (ball_screen_x, ball_screen_y), 10)

        slope_angle = pathgen.get_slope_from_distance(current_distance, current_path_type)
        vx = ball_velocity * np.cos(slope_angle)
        vy = ball_velocity * np.sin(slope_angle)

        elapsed = physics_time

        text_x = WIDTH - 220
        text_y = 10

        path_text = f"PATH: {current_path_type}"
        screen.blit(font.render(path_text, True, WHITE), (10, 10))

        info_lines = [
            f"TIME: {elapsed:.4f}s",
            f"POSITION: ({ball_x:.4f}, {ball_y:.4f})",
            f"Vx: {vx:.4f} m/s",
            f"Vy: {vy:.4f} m/s"
        ]

        for i, line in enumerate(info_lines):
            screen.blit(font.render(line, True, WHITE), (text_x, text_y + i * 20))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    animation()
