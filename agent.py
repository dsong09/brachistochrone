import numpy as np

class Agent:
    def __init__(self, y_min, y_max):
        self.y_min = y_min
        self.y_max = y_max
        self.current_segment = 0

        self.last_endpoint = None
        self.last_handle = None

    def reset(self):
        self.current_segment = 0
        self.last_endpoint = np.array([0.0, self.y_min])
        self.last_handle = self.last_endpoint + np.array([0.5, 0.0])

    def action(self, prev_endpoint, prev_handle):
        cp0 = prev_endpoint

        # Add jitter to the reflected point to break symmetry more strongly
        reflected = 2 * cp0 - prev_handle + np.random.uniform(-1.0, 1.0, size=2)

        # cp1 can vary widely around reflected (both x and y)
        cp1 = reflected + np.random.uniform(-1.0, 1.0, size=2)

        # cp2 moves roughly forward but allow backward x up to small amount for curve complexity
        cp2_x = cp1[0] + np.random.uniform(-0.3, 1.5)
        cp2_y = cp1[1] + np.random.uniform(-2.0, 2.0)
        cp2 = np.array([cp2_x, cp2_y])

        # cp3 can move forward or slightly backward x, large vertical shifts allowed for steep slopes
        cp3_x = cp2[0] + np.random.uniform(-0.5, 2.0)
        cp3_y = cp2[1] + np.random.uniform(-3.0, 1.0)
        cp3 = np.array([cp3_x, cp3_y])

        # Clamp y to allowed bounds
        cp3[1] = np.clip(cp3[1], self.y_min, self.y_max)

        self.current_segment += 1
        self.last_endpoint = cp3
        self.last_handle = cp2

        return cp1, cp2, cp3

    def update_after_episode(self, reward):
        print(f"Episode finished with reward: {reward:.4f}")

    def train_on_batch(self, states, actions, rewards, dones, next_states):
        print(f"Training on batch of size: {states.shape[0]}")
