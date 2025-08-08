import numpy as np

FPS = 120
GRAVITY = 9.8

def update_motion(current_distance, ball_velocity, path, path_type, dt=1.0 / FPS):
    # update distance/velocity after dt seconds
    # path: PathGenerator
    # path_type: "BRACHISTOCHRONE" or "CUBIC"
    slope = path.slope_from_distance(current_distance, path_type)
    acceleration = GRAVITY * np.sin(slope)

    ball_velocity += acceleration * dt
    new_distance = current_distance + ball_velocity * dt

    total_path_length = path.total_cubic_length if path_type == "AGENT" else path.total_brach_length

    done = False
    if new_distance >= total_path_length:
        new_distance = total_path_length
        done = True
    elif new_distance < 0:
        ball_velocity = 0.0
        new_distance = 0.0

    return new_distance, ball_velocity, done
