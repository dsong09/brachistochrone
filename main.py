import pygame
import numpy as np
from path_generator import PathGenerator, START_POINT, END_POINT, NUM_CONTROL_POINTS
from physics import update_motion, FPS
from agent import Agent

WHITE = (255, 255, 255)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
BLACK = (0, 0, 0)

WIDTH = 800
HEIGHT = 600

X_PADDING = 1.0
Y_PADDING = 1.0

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

    brachistochrone_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pathgen = PathGenerator(START_POINT, END_POINT, NUM_CONTROL_POINTS)

    # for now, manually set limit for agent y-values
    agent = Agent(NUM_CONTROL_POINTS, -1.0, 6.0)
    agent.reset()
    pathgen.generate_agent_path(agent.action())

    x_min, x_max = -1.0, 10.0
    y_min, y_max = -1.0, 8.0

    x_range = (x_max - x_min) + 2 * X_PADDING
    y_range = (y_max - y_min) + 2 * Y_PADDING

    scale_x = WIDTH / x_range
    scale_y = HEIGHT / y_range

    offset_x = x_min - X_PADDING
    offset_y = y_min - Y_PADDING

    current_distance = 0.0
    ball_velocity = 0.0
    physics_time = 0.0
    reached_end = False
    printed_time_check = False
    current_path_type = "AGENT"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    agent.reset()
                    pathgen.generate_agent_path(agent.action())
                    current_distance = 0.0
                    ball_velocity = 0.0
                    physics_time = 0.0
                    reached_end = False
                    current_path_type = "AGENT"

                elif event.key == pygame.K_t:
                    current_path_type = "BRACHISTOCHRONE" if current_path_type == "AGENT" else "AGENT"
                    current_distance = 0.0
                    ball_velocity = 0.0
                    physics_time = 0.0
                    reached_end = False

        dt = 0 if reached_end else 1.0 / FPS

        if not reached_end:
            prev_distance = current_distance
            prev_velocity = ball_velocity

            current_distance, ball_velocity, reached_end, partial_dt = update_motion(
                current_distance, ball_velocity, pathgen, current_path_type, dt
            )

            physics_time += partial_dt

        ball_x, ball_y = pathgen.position_from_distance(current_distance, current_path_type)
        ball_screen_x, ball_screen_y = world_to_screen(ball_x, ball_y, scale_x, scale_y, offset_x, offset_y)

        screen.fill(BLACK)
        screen.blit(brachistochrone_surface, (0, 0))

        cubic_points = [world_to_screen(x, y, scale_x, scale_y, offset_x, offset_y) for x, y in zip(pathgen.x_path, pathgen.y_path)]
        if len(cubic_points) > 1:
            pygame.draw.lines(screen, WHITE, False, cubic_points, 3)

        brach_points = [world_to_screen(x, y, scale_x, scale_y, offset_x, offset_y) for x, y in zip(pathgen.x_brach, pathgen.y_brach)]
        if len(brach_points) > 1:
            pygame.draw.lines(screen, ORANGE, False, brach_points, 3)

        start_screen = world_to_screen(START_POINT[0], START_POINT[1], scale_x, scale_y, offset_x, offset_y)
        end_screen = world_to_screen(END_POINT[0], END_POINT[1], scale_x, scale_y, offset_x, offset_y)
        pygame.draw.circle(screen, WHITE, start_screen, 4)
        pygame.draw.circle(screen, WHITE, end_screen, 4)
        pygame.draw.circle(screen, RED, (ball_screen_x, ball_screen_y), 10)

        slope_angle = pathgen.slope_from_distance(current_distance, current_path_type)
        vx = ball_velocity * np.cos(slope_angle)
        vy = ball_velocity * np.sin(slope_angle)

        elapsed = physics_time

        # view parameters and test simulation time vs. actual time
        if reached_end and current_path_type == "BRACHISTOCHRONE" and not printed_time_check:
            print("Brachistochrone Max Angle:", pathgen.theta_max)
            print("Cycloid Parameter A:", pathgen.a_brach)
            print("Simulation Time:", physics_time)
            print("Analytical Time:", pathgen.travel_time_brach())
            printed_time_check = True

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

