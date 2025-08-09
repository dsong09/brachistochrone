import numpy as np

FPS = 120
GRAVITY = 9.81

def update_motion(current_distance, ball_velocity, path, path_type, dt=1.0/FPS):
    # update distance/velocity after dt seconds
    # path: PathGenerator
    # path_type: "BRACHISTOCHRONE" or "AGENT"
    slope = path.slope_from_distance(current_distance, path_type)
    acceleration = GRAVITY * np.sin(slope)

    ball_velocity += acceleration * dt
    new_distance = current_distance + ball_velocity * dt

    # allows for end time to not be multiple of dt = 1 / FPS
    done = False
    partial_dt = dt

    if new_distance >= (path.total_cubic_length if path_type == "AGENT" else path.total_brach_length):
        total_path_length = path.total_cubic_length if path_type == "AGENT" else path.total_brach_length
        distance_to_end = total_path_length - current_distance
        partial_dt = distance_to_end / ball_velocity if ball_velocity > 0 else 0
        new_distance = total_path_length
        done = True
    elif new_distance < 0:
        ball_velocity = 0.0
        new_distance = 0.0

    return new_distance, ball_velocity, done, partial_dt