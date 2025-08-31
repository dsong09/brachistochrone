import pygame
import numpy as np
from path_generator import PathGenerator, START_POINT, END_POINT
from physics import update_motion, FPS

WHITE  = (255, 255, 255)
RED    = (255,   0,   0)
ORANGE = (255, 165,   0)
BLACK  = (  0,   0,   0)

WIDTH, HEIGHT = 800, 600
X_PADDING, Y_PADDING = 1.0, 1.0

SEGMENT_LENGTH = 0.1

def world_to_screen(x, y, scale_x, scale_y, offset_x, offset_y):
    return int((x - offset_x) * scale_x), int((y - offset_y) * scale_y)

def animation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("brachistochrone")
    clock = pygame.time.Clock()
    font = pygame.font.Font("fonts/DeterminationSansWebRegular-369X.ttf", 20)

    # Path generator with sampling resolution ds
    pathgen = PathGenerator(ds=0.05)

    # --- setup screen scaling ---
    x_min = min(START_POINT[0], END_POINT[0]) - 2.0
    x_max = max(START_POINT[0], END_POINT[0]) + 2.0
    y_min = min(START_POINT[1], END_POINT[1]) - 2.0
    y_max = max(START_POINT[1], END_POINT[1]) + 2.0
    x_range = (x_max - x_min) + 2 * X_PADDING
    y_range = (y_max - y_min) + 2 * Y_PADDING
    scale_x = WIDTH / x_range
    scale_y = HEIGHT / y_range
    offset_x = x_min - X_PADDING
    offset_y = y_min - Y_PADDING

    current_path_type = "AGENT"
    current_distance  = 0.0
    ball_velocity     = 0.0
    sim_time          = 0.0

    first_cp1, first_cp2, first_p3 = pathgen.random_segment(segment_length=SEGMENT_LENGTH)
    pathgen.add_segment(first_cp1, first_cp2, first_p3)
    last_endpoint = first_p3

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # manual reset
                    pathgen = PathGenerator(ds=0.05)
                    current_distance  = 0.0
                    ball_velocity     = 0.0
                    sim_time          = 0.0
                    first_cp1, first_cp2, first_p3 = pathgen.random_segment(segment_length=SEGMENT_LENGTH)
                    pathgen.add_segment(first_cp1, first_cp2, first_p3)
                    last_endpoint = first_p3
                elif event.key == pygame.K_t:
                    # toggle path type
                    current_path_type = "BRACHISTOCHRONE" if current_path_type == "AGENT" else "AGENT"
                    current_distance  = 0.0
                    ball_velocity     = 0.0
                    sim_time          = 0.0

        dt = 1.0 / FPS
        prev_distance = current_distance
        current_distance, ball_velocity = update_motion(
            current_distance, ball_velocity, pathgen, current_path_type, dt
        )
        delta_d = current_distance - prev_distance
        if ball_velocity > 0:
            sim_time += delta_d / ball_velocity

        # --- grow agent path continuously ---
        if current_path_type == "AGENT":
            buffer_ahead = 1.0
            if pathgen.total_length("AGENT") - current_distance < buffer_ahead:
                cp1, cp2, p3 = pathgen.random_segment(segment_length=SEGMENT_LENGTH)
                pathgen.add_segment(cp1, cp2, p3)
                last_endpoint = p3

        # --------- Draw ---------
        screen.fill(BLACK)

        # Agent path
        total_len_agent = pathgen.total_length("AGENT")
        if total_len_agent > 0:
            ss = np.linspace(0.0, total_len_agent, 400)
            pts = [pathgen.position_from_distance(s, "AGENT") for s in ss]
            pts_scr = [world_to_screen(x, y, scale_x, scale_y, offset_x, offset_y) for (x, y) in pts]
            if len(pts_scr) > 1:
                pygame.draw.lines(screen, WHITE, False, pts_scr, 3)

        # Brachistochrone path
        total_len_b = pathgen.total_length("BRACHISTOCHRONE")
        if total_len_b > 0:
            ssb = np.linspace(0.0, total_len_b, 500)
            pts_b = [pathgen.position_from_distance(s, "BRACHISTOCHRONE") for s in ssb]
            pts_b_scr = [world_to_screen(x, y, scale_x, scale_y, offset_x, offset_y) for (x, y) in pts_b]
            if len(pts_b_scr) > 1:
                pygame.draw.lines(screen, ORANGE, False, pts_b_scr, 3)

        # Start/end markers
        start_scr = world_to_screen(START_POINT[0], START_POINT[1], scale_x, scale_y, offset_x, offset_y)
        end_scr   = world_to_screen(END_POINT[0],   END_POINT[1],   scale_x, scale_y, offset_x, offset_y)
        pygame.draw.circle(screen, WHITE, start_scr, 4)
        pygame.draw.circle(screen, WHITE, end_scr,   4)

        # Ball
        bx, by = pathgen.position_from_distance(current_distance, current_path_type)
        bx_s, by_s = world_to_screen(bx, by, scale_x, scale_y, offset_x, offset_y)
        pygame.draw.circle(screen, RED, (bx_s, by_s), 10)

        # Info
        slope_angle = pathgen.slope_from_distance(current_distance, current_path_type)
        vx = ball_velocity * np.cos(slope_angle)
        vy = ball_velocity * np.sin(slope_angle)

        screen.blit(font.render(f"PATH: {current_path_type}", True, WHITE), (10, 10))
        info_lines = [
            f"TIME: {sim_time:.4f}s",
            f"POS: ({bx:.3f}, {by:.3f})",
            f"Vx:  {vx:.3f} m/s",
            f"Vy:  {vy:.3f} m/s",
        ]
        for i, line in enumerate(info_lines):
            screen.blit(font.render(line, True, WHITE), (WIDTH - 220, 10 + i * 20))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    animation()
