import numpy as np

FPS = 120
GRAVITY = 9.81
EPS = 1e-4  # small velocity threshold

def update_motion(current_distance, ball_velocity, pathgen, path_type, dt=1.0 / FPS):
    slope = pathgen.slope_from_distance(current_distance, path_type)
    acceleration = GRAVITY * np.sin(slope)

    # Update velocity
    ball_velocity += acceleration * dt
    # Clamp velocity to zero if too small
    if ball_velocity < EPS and acceleration <= 0:
        ball_velocity = 0.0

    # Update distance
    new_distance = current_distance + ball_velocity * dt

    # Clamp distance to [0, total_length]
    max_distance = pathgen.total_length(path_type)
    new_distance = np.clip(new_distance, 0.0, max_distance)

    return new_distance, ball_velocity
