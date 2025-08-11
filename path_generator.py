import numpy as np
from physics import GRAVITY
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

START_POINT = (0.0, 0.0)
END_POINT = (9.0, 6.0)
NUM_CONTROL_POINTS = 32

class PathGenerator:
    def __init__(self, start=START_POINT, end=END_POINT, num_points=NUM_CONTROL_POINTS):
        self.start = start
        self.end = end
        self.num_points = num_points

        self.x_control = np.linspace(self.start[0], self.end[0], self.num_points + 2)
        self.y_control = np.zeros(self.num_points + 2)
        self.y_control[0] = self.start[1]
        self.y_control[-1] = self.end[1]

        self.spline = None
        self.x_path = np.array([])
        self.y_path = np.array([])
        self.spline_deriv = None

        self.x_brach = np.array([])
        self.y_brach = np.array([])
        self.a_brach = 0.0
        self.theta_max = 0.0

        self.cumulative_dist_cubic = []
        self.cumulative_dist_brach = []
        self.total_cubic_length = 0.0
        self.total_brach_length = 0.0

        self.generate_brachistochrone()
        self.update_spline()

    @staticmethod
    def cumulative_distances(x_points, y_points):
        dx = np.diff(x_points)
        dy = np.diff(y_points)
        distances = np.sqrt(dx**2 + dy**2)
        return np.concatenate([[0.0], np.cumsum(distances)])

    def generate_brachistochrone(self):
        def solve_theta(theta):
            return (theta - np.sin(theta)) / (1 - np.cos(theta)) - (self.end[0] / self.end[-1])

        # solving for cycloid parameter a
        self.theta_max = fsolve(solve_theta, 3.0)[0]
        self.a_brach = self.end[-1] / (1 - np.cos(self.theta_max))
        t = np.linspace(0, self.theta_max, 10000)

        # analytical solution: cycloid
        x = self.a_brach * (t - np.sin(t))
        y = self.a_brach * (1.0 - np.cos(t))

        x_scale = self.end[0] / x[-1]
        y_scale = self.end[1] / y[-1]

        self.x_brach = x * x_scale
        self.y_brach = y * y_scale
        self.cumulative_dist_brach = self.cumulative_distances(self.x_brach, self.y_brach)
        self.total_brach_length = self.cumulative_dist_brach[-1]

    def travel_time_brach(self):
        return np.sqrt(self.a_brach / GRAVITY) * self.theta_max

    def generate_agent_path(self, y_control_points: np.ndarray):
        if len(y_control_points) != self.num_points:
            raise ValueError(f"Expected {self.num_points} control points, got {len(y_control_points)}")

        self.y_control[1:-1] = y_control_points
        self.update_spline()

    def update_spline(self):
        self.spline = CubicSpline(self.x_control, self.y_control)
        self.x_path = np.linspace(self.start[0], self.end[0], 1000)
        self.y_path = self.spline(self.x_path)
        self.spline_deriv = self.spline.derivative()

        self.cumulative_dist_cubic = self.cumulative_distances(self.x_path, self.y_path)
        self.total_cubic_length = self.cumulative_dist_cubic[-1]

    def position_from_distance(self, dist, path_type):
        if path_type == "AGENT":
            cumulative = self.cumulative_dist_cubic
            x_path, y_path = self.x_path, self.y_path
        else:
            cumulative = self.cumulative_dist_brach
            x_path, y_path = self.x_brach, self.y_brach

        idx = int(np.searchsorted(cumulative, dist, side="right")) - 1
        if idx >= len(cumulative) - 1:
            return self.end

        frac = (dist - cumulative[idx]) / (cumulative[idx + 1] - cumulative[idx])
        x = x_path[idx] + frac * (x_path[idx + 1] - x_path[idx])
        y = y_path[idx] + frac * (y_path[idx + 1] - y_path[idx])
        return x, y

    def slope_from_distance(self, dist, path_type):
        if path_type == "AGENT":
            x, _ = self.position_from_distance(dist, path_type)
            return np.arctan(self.spline_deriv(x))
        else:
            cumulative = self.cumulative_dist_brach
            idx = np.searchsorted(cumulative, dist, side="right") - 1
            if idx >= len(self.x_brach) - 1:
                idx = len(self.x_brach) - 2
            dx = self.x_brach[idx + 1] - self.x_brach[idx]
            dy = self.y_brach[idx + 1] - self.y_brach[idx]
            return np.arctan2(dy, dx)
