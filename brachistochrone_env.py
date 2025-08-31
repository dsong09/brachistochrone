from path_generator import PathGenerator, END_POINT
from physics import update_motion, FPS
import numpy as np


class BrachistochroneEnv:
    def __init__(self, pathgen: PathGenerator, end_point_tolerance=0.01):
        self.pathgen = pathgen
        self.end_point_tolerance = end_point_tolerance
        self.reset()

        self.current_distance = 0.0
        self.ball_velocity = 0.0
        self.done = None
        self.time_elapsed = 0.0

    def reset(self):
        self.current_distance = 0.0
        self.ball_velocity = 0.0
        self.done = False
        self.time_elapsed = 0.0

        # Reset PathGenerator segments and control points, keep only start point
        self.pathgen.control_points = [np.array(self.pathgen.start, dtype=float)]
        self.pathgen.segments_built = 0
        self.pathgen.total_agent_length = 0.0
        self.pathgen.cumulative_segment_lengths = np.array([], dtype=float)

        return self.get_observation()

    def get_observation(self):
        x, y = self.pathgen.position_from_distance(self.current_distance, "AGENT")
        return np.array(
            [x, y, self.ball_velocity] + list(pt[1] for pt in self.pathgen.control_points[1:-1]),
            dtype=np.float32
        )

    def check_termination(self):
        """Check if episode should end."""
        x, y = self.pathgen.position_from_distance(self.current_distance, "AGENT")

        # Case 1: Ball stuck (no velocity and not at start)
        stuck = abs(self.ball_velocity) < 1e-5 and self.current_distance > 0.0

        # Case 2: Out of bounds
        out_of_bounds = not (0 <= x <= 1 and 0 <= y <= 1)

        # Case 3: Close enough to endpoint
        end_x, end_y = END_POINT
        near_goal = np.hypot(x - end_x, y - end_y) < self.end_point_tolerance

        return stuck or out_of_bounds or near_goal

    def step(self, action_control_points):
        prev_state = self.get_observation()

        # action_control_points expected to be (cp1, cp2, cp3) for one segment
        cp1, cp2, cp3 = action_control_points
        self.pathgen.add_segment(cp1, cp2, cp3)

        # Physics update
        dt = 1.0 / FPS
        self.current_distance, self.ball_velocity, done_from_physics, partial_dt = update_motion(
            self.current_distance,
            self.ball_velocity,
            self.pathgen,
            "AGENT",
            dt
        )

        # Accumulate time
        if not done_from_physics:
            self.time_elapsed += dt
        else:
            self.time_elapsed += partial_dt

        # Check all termination conditions
        self.done = done_from_physics or self.check_termination()

        # Reward: negative elapsed time (faster = better)
        reward = -self.time_elapsed

        next_state = self.get_observation()

        return next_state, reward, self.done, {}
