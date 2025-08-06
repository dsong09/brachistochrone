import pygame
import numpy as np
import time
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

def animation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("brachistochrone")
    clock = pygame.time.Clock()
    font = pygame.font.Font("fonts/DeterminationSansWebRegular-369X.ttf", 20)

    # Create a surface for the brachistochrone path with transparency
    brachisto_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    # Brachistochrone parameters to exactly match endpoint (9, 6)
    a = 6.0 / (1 - np.cos(1.8))  # Calculated to hit y=6
    theta_max = 1.8  # Angle that gives x=9 when scaled

    # Generate cycloid points
    t = np.linspace(0, theta_max, 100)
    x_brach = a * (t - np.sin(t))
    y_brach = a * (1 - np.cos(t))

    # Scale to exactly match endpoint
    x_scale = END_POINT[0] / x_brach[-1]
    y_scale = END_POINT[1] / y_brach[-1]
    x_brach = x_brach * x_scale
    y_brach = y_brach * y_scale

    # Precompute cumulative path distance for brachistochrone
    cumulative_dist_brach = [0.0]
    for i in range(1, len(x_brach)):
        dx = x_brach[i] - x_brach[i - 1]
        dy = y_brach[i] - y_brach[i - 1]
        cumulative_dist_brach.append(cumulative_dist_brach[-1] + np.sqrt(dx ** 2 + dy ** 2))
    total_brach_length = cumulative_dist_brach[-1]

    def create_smooth_path():
        # Create random control points with fixed start and end
        x_control = np.linspace(START_POINT[0], END_POINT[0], NUM_CONTROL_POINTS + 2)
        y_control = np.random.uniform(0.0, 8.0, NUM_CONTROL_POINTS + 2)

        # Fix start and end points
        y_control[0] = START_POINT[1]
        y_control[-1] = END_POINT[1]

        # Create cubic spline
        spline = CubicSpline(x_control, y_control)
        x_smooth = np.linspace(START_POINT[0], END_POINT[0], 500)
        y_smooth = spline(x_smooth)
        return x_smooth, y_smooth, spline, x_control, y_control

    x_path, y_path, spline, x_control, y_control = create_smooth_path()
    spline_deriv = spline.derivative()

    # Precompute cumulative path distance for cubic spline
    cumulative_dist_cubic = [0.0]
    for i in range(1, len(x_path)):
        dx = x_path[i] - x_path[i - 1]
        dy = y_path[i] - y_path[i - 1]
        cumulative_dist_cubic.append(cumulative_dist_cubic[-1] + np.sqrt(dx ** 2 + dy ** 2))
    total_cubic_length = cumulative_dist_cubic[-1]

    # Function to get position from distance
    def get_position_from_distance(dist, path_type):
        if path_type == "CUBIC":
            cumulative_dist = cumulative_dist_cubic
            x_path_curr = x_path
            y_path_curr = y_path
        else:  # brachistochrone
            cumulative_dist = cumulative_dist_brach
            x_path_curr = x_brach
            y_path_curr = y_brach

        # Find the segment containing the distance
        idx = 0
        while idx < len(cumulative_dist) - 1 and cumulative_dist[idx + 1] < dist:
            idx += 1

        if idx >= len(cumulative_dist) - 1:
            return END_POINT[0], END_POINT[1]

        # Calculate position within the segment
        seg_start_dist = cumulative_dist[idx]
        seg_end_dist = cumulative_dist[idx + 1]
        seg_length = seg_end_dist - seg_start_dist
        fraction = (dist - seg_start_dist) / seg_length

        x = x_path_curr[idx] + fraction * (x_path_curr[idx + 1] - x_path_curr[idx])
        y = y_path_curr[idx] + fraction * (y_path_curr[idx + 1] - y_path_curr[idx])
        return x, y

    def get_slope_from_distance(dist, path_type):
        if path_type == "CUBIC":
            x, y = get_position_from_distance(dist, path_type)
            return np.arctan(spline_deriv(x))
        else:
            idx = 0
            while idx < len(cumulative_dist_brach) - 1 and cumulative_dist_brach[idx + 1] < dist:
                idx += 1

            if idx >= len(cumulative_dist_brach) - 1:
                idx = len(cumulative_dist_brach) - 2

            dx = x_brach[idx + 1] - x_brach[idx]
            dy = y_brach[idx + 1] - y_brach[idx]
            return np.arctan2(dy, dx)

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

    def world_to_screen(x, y):
        screen_x = int((x - offset_x) * scale_x)
        screen_y = int((y - offset_y) * scale_y)
        return screen_x, screen_y

    brachisto_points = [world_to_screen(x, y) for x, y in zip(x_brach, y_brach)]

    if len(brachisto_points) > 1:
        pygame.draw.lines(brachisto_surface, ORANGE, False, brachisto_points, 3)

    path_points = [world_to_screen(x, y) for x, y in zip(x_path, y_path)]
    control_points_screen = [world_to_screen(x, y) for x, y in zip(x_control, y_control)]

    current_distance = 0.0
    ball_velocity = 0.0
    start_time = time.time()
    end_time = None
    has_reached_end = False

    current_path_type = "CUBIC"
    physics_time = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:  # Reset simulation
                    # Generate a new random path
                    x_path, y_path, spline, x_control, y_control = create_smooth_path()
                    spline_deriv = spline.derivative()

                    # Recompute cumulative path distance for cubic
                    cumulative_dist_cubic = [0.0]
                    for i in range(1, len(x_path)):
                        dx = x_path[i] - x_path[i - 1]
                        dy = y_path[i] - y_path[i - 1]
                        cumulative_dist_cubic.append(cumulative_dist_cubic[-1] + np.sqrt(dx ** 2 + dy ** 2))
                    total_cubic_length = cumulative_dist_cubic[-1]

                    # Update path points
                    path_points = [world_to_screen(x, y) for x, y in zip(x_path, y_path)]
                    control_points_screen = [world_to_screen(x, y) for x, y in zip(x_control, y_control)]

                    # Reset ball
                    current_distance = 0.0
                    ball_velocity = 0.0
                    start_time = time.time()
                    end_time = None
                    has_reached_end = False
                    physics_time = 0.0

                elif event.key == pygame.K_t:
                    if current_path_type == "CUBIC":
                        current_path_type = "BRACHISTOCHRONE"
                    else:
                        current_path_type = "CUBIC"

                    current_distance = 0.0
                    ball_velocity = 0.0
                    has_reached_end = False
                    physics_time = 0.0

        if has_reached_end:
            dt = 0
        else:
            dt = 1.0 / FPS

            prev_distance = current_distance
            prev_velocity = ball_velocity

            slope = get_slope_from_distance(current_distance, current_path_type)

            acceleration = GRAVITY * np.sin(slope)
            ball_velocity += acceleration * dt

            new_distance = current_distance + ball_velocity * dt

            total_path_length = total_cubic_length if current_path_type == "CUBIC" else total_brach_length

            if new_distance < 0:
                new_distance = 0
                ball_velocity = 0  # Stop at start
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

        ball_x, ball_y = get_position_from_distance(current_distance, current_path_type)
        ball_screen_x, ball_screen_y = world_to_screen(ball_x, ball_y)

        screen.fill(BLACK)

        screen.blit(brachisto_surface, (0, 0))

        for i in range(len(path_points) - 1):
            pygame.draw.line(screen, WHITE, path_points[i], path_points[i + 1], 2)

        start_screen = world_to_screen(START_POINT[0], START_POINT[1])
        end_screen = world_to_screen(END_POINT[0], END_POINT[1])
        pygame.draw.circle(screen, WHITE, start_screen, 4)
        pygame.draw.circle(screen, WHITE, end_screen, 4)

        ball_radius = 10
        pygame.draw.circle(screen, RED, (ball_screen_x, ball_screen_y), ball_radius)

        slope_angle = get_slope_from_distance(current_distance, current_path_type)
        vx = ball_velocity * np.cos(slope_angle)  # x-component (horizontal)
        vy = ball_velocity * np.sin(slope_angle)  # y-component (vertical)

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