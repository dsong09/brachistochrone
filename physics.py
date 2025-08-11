import numpy as np

FPS = 120
GRAVITY = 9.81

def update_motion(current_distance, ball_velocity, pathgen, path_type, dt=1.0/FPS):
    # update distance/velocity after dt seconds
    slope = pathgen.slope_from_distance(current_distance, path_type)

    # acceleration component along slope
    acceleration = GRAVITY * np.sin(slope)
    ball_velocity += acceleration * dt
    new_distance = current_distance + ball_velocity * dt

    # clamp distance to total path length
    max_distance = pathgen.total_length(path_type)
    if new_distance >= max_distance:
        new_distance = max_distance
        ball_velocity = 0.0

    return new_distance, ball_velocity