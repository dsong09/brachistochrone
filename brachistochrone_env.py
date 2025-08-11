from path_generator import PathGenerator
from physics import update_motion, FPS
import numpy as np

class BrachistochroneEnv:
    def __init__(self, pathgen: PathGenerator):
        self.pathgen = pathgen
        self.reset()
        self.time_elapsed = 0.0
        self.current_distance = 0.0
        self.ball_velocity = 0.0
        self.done = None

    def reset(self):
        self.current_distance = 0.0
        self.ball_velocity = 0.0
        self.done = False
        self.time_elapsed = 0.0
        # agent needs observation of environment before first action
        return self.get_observation()

    def get_observation(self):
        x, y = self.pathgen.position_from_distance(self.current_distance, "AGENT")
        return np.array([x, y, self.ball_velocity] + list(self.pathgen.y_control[1:-1]), dtype=np.float32)

    def step(self, action_control_points):
        # current state before action
        prev_state = self.get_observation()

        # action changes path
        self.pathgen.generate_agent_path(action_control_points)

        # physics step: update motion after dt
        dt = 1.0 / FPS
        self.current_distance, self.ball_velocity, self.done, partial_dt = update_motion(
            self.current_distance,
            self.ball_velocity,
            self.pathgen,
            "AGENT",
            dt
        )

        if not self.done:
            self.time_elapsed += dt
        else:
            self.time_elapsed += partial_dt

        # reward function
        reward = -self.time_elapsed

        # update state after action
        next_state = self.get_observation()

        return next_state, reward, self.done, {}
