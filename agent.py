import numpy as np

class Agent:
    def __init__(self, num_control_points, y_min, y_max):
        self.num_control_points = num_control_points
        self.control_points = None
        self.y_min = y_min
        self.y_max = y_max

    def reset(self):
        # random for now
        self.control_points = np.random.uniform(self.y_min, self.y_max, self.num_control_points)

    def action(self):
        return self.control_points

    def update_after_episode(self, reward):
        # maybe print total reward in each episode for plotting?
        print(f"Episode finished with reward: {reward:.4f}")

    def train_on_batch(self, states, actions, rewards, dones, next_states):
        # training logic for a batch here
        print(f"Training on batch of size: {states.shape[0]}")

