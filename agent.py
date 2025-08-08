import numpy as np

class Agent:
    def __init__(self, num_control_points):
        self.num_control_points = num_control_points
        self.control_points = np.zeros(num_control_points)

    def action(self):
        return self.control_points