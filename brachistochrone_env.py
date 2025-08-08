from path_generator import PathGenerator

class BrachistochroneEnv:
    def __init__(self, path: PathGenerator):
        self.path = path
        self.reset()
        self.current_distance = 0.0
        self.ball_velocity = 0.0
        self.done = None

    def reset(self):
        self.current_distance = 0.0
        self.ball_velocity = 0.0
        self.done = False
        # agent needs observation of environment before first action
        return self.get_observation()

    def get_observation(self):
        x, y = self.path.position_from_distance(self.current_distance, "CUBIC")
        return {
            "position": (x, y),
            "velocity": self.ball_velocity,
            "y_control": self.path.y_control[1:-1].copy()
        }


