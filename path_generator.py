import numpy as np
from physics import GRAVITY
from math import comb
from scipy.optimize import fsolve

START_POINT = (0.0, 0.0)
END_POINT = (9.0, 6.0)
NUM_CONTROL_POINTS = 32

def bernstein_polynomial(i, n, t):
    return comb(n, i) * (t**i) * ((1 - t)**(n - i))

class PathGenerator:
    def __init__(self, start=START_POINT, end=END_POINT, max_segments=NUM_CONTROL_POINTS):
        self.start = start
        self.end = end
        self.max_segments = max_segments

        # 4 points per segment in cubic Bezier, last point of segment becomes first of next
        # for the first segment, one of the control points should be the starting point
        self.control_points = [self.start]
        self.segments_built = 0

        # interval between points set by number of points
        self.x_control = np.linspace(self.start[0], self.end[0], self.max_segments + 2)

        # endpoints are fixed, but points (and path segments) are set incrementally
        self.y_control = np.zeros(self.max_segments + 2)
        self.y_control[0] = self.start[1]
        self.y_control[-1] = self.end[1]

        self.fixed_segments = []
        self.current_segment_x = None
        self.current_segment_y = None

        self.total_agent_length = 0.0

        # brachistochrone parameters
        self.x_brach = np.array([])
        self.y_brach = np.array([])
        self.a_brach = 0.0
        self.theta_max = 0.0
        self.cumulative_dist_brach = []
        self.total_brach_length = 0.0

        self.generate_brachistochrone()

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

    def add_segment(self, cp1, cp2, end_point):
        if self.segments_built >= self.max_segments:
            raise ValueError("Maximum number of segments reached")

        if self.segments_built > 0:
            cp1 = 2 * self.control_points[-1] - self.control_points[-2]

        self.control_points.extend([cp1, cp2, end_point])
        self.segments_built += 1

        # update total agent path length whenever a new segment is added
        self.compute_total_agent_length()

    def compute_total_agent_length(self, num_samples_per_segment=100):
        total_length = 0.0
        for seg in range(self.segments_built):
            ts = np.linspace(0, 1, num_samples_per_segment)
            points = np.array([self.evaluate_segment(seg, t) for t in ts])
            dx = np.diff(points[:, 0])
            dy = np.diff(points[:, 1])
            seg_length = np.sum(np.sqrt(dx**2 + dy**2))
            total_length += seg_length
        self.total_agent_length = total_length

    def evaluate_segment(self, segment_index, t):
        i = segment_index * 3
        p0 = self.control_points[i]
        p1 = self.control_points[i + 1]
        p2 = self.control_points[i + 2]
        p3 = self.control_points[i + 3]

        point = (
            (1 - t)**3 * p0 +
            3 * (1 - t)**2 * t * p1 +
            3 * (1 - t) * t**2 * p2 +
            t**3 * p3
        )

        return point

    @staticmethod
    def clip_t(t_global):
        return np.clip(t_global, 0.0, 1.0)

    def evaluate_path(self, t_global):
        if self.segments_built == 0:
            return self.start

        t_global = self.clip_t(t_global)

        total_segments = self.segments_built
        scaled_t = t_global * total_segments
        segment_index = int(np.floor(scaled_t))
        if segment_index >= total_segments:
            segment_index = total_segments - 1

        local_t = scaled_t - segment_index
        return self.evaluate_segment(segment_index, local_t)

    def slope_segment(self, segment_index, t):
        i = segment_index * 3
        p0 = self.control_points[i]
        p1 = self.control_points[i + 1]
        p2 = self.control_points[i + 2]
        p3 = self.control_points[i + 3]

        dp = (
            3 * (1 - t)**2 * (p1 - p0) +
            6 * (1 - t) * t * (p2 - p1) +
            3 * t**2 * (p3 - p2)
        )

        return np.arctan2(dp[1], dp[0])

    def slope_path(self, t_global):
        if self.segments_built == 0:
            return 0.0

        t_global = self.clip_t(t_global)
        total_segments = self.segments_built
        scaled_t = t_global * total_segments
        segment_index = int(np.floor(scaled_t))
        if segment_index >= total_segments:
            segment_index = total_segments - 1

        local_t = scaled_t - segment_index
        return self.slope_segment(segment_index, local_t)

    def find_segment_and_local_t(self, dist):
        remaining = dist
        num_samples = 100

        for seg in range(self.segments_built):
            ts = np.linspace(0, 1, num_samples)
            points = np.array([self.evaluate_segment(seg, t) for t in ts])
            dx = np.diff(points[:, 0])
            dy = np.diff(points[:, 1])
            seg_length = np.sum(np.sqrt(dx ** 2 + dy ** 2))

            if remaining <= seg_length:
                cum_lengths = np.cumsum(np.sqrt(dx ** 2 + dy ** 2))
                t_idx = int(np.searchsorted(cum_lengths, remaining))
                t_idx = min(t_idx, len(cum_lengths) - 1)

                t0 = t_idx / (num_samples - 1)
                t1 = min(t0 + 1 / (num_samples - 1), 1.0)

                if t_idx == 0:
                    frac = 0.0
                else:
                    denominator = cum_lengths[t_idx] - cum_lengths[t_idx - 1]
                    frac = (remaining - cum_lengths[t_idx - 1]) / denominator if denominator > 0 else 0.0

                t = t0 + frac * (t1 - t0)
                return seg, t
            else:
                remaining -= seg_length

        if self.segments_built > 0:
            return self.segments_built - 1, 1.0
        else:
            return None, None

    def position_from_distance(self, dist, path_type):
        if path_type == "AGENT":
            seg_t = self.find_segment_and_local_t(dist)
            if seg_t == (None, None):
                return self.control_points[0]
            seg, t = seg_t
            return self.evaluate_segment(seg, t)
        elif path_type == "BRACHISTOCHRONE":
            cumulative = self.cumulative_dist_brach
            idx = np.searchsorted(cumulative, dist, side="right") - 1
            if idx >= len(self.x_brach) - 1:
                idx = len(self.x_brach) - 2
            x = self.x_brach[idx] + (dist - cumulative[idx]) / (cumulative[idx + 1] - cumulative[idx]) * (
                        self.x_brach[idx + 1] - self.x_brach[idx])
            y = self.y_brach[idx] + (dist - cumulative[idx]) / (cumulative[idx + 1] - cumulative[idx]) * (
                        self.y_brach[idx + 1] - self.y_brach[idx])
            return np.array([x, y])
        else:
            raise ValueError(f"Unknown path_type: {path_type}")
    def slope_from_distance(self, dist, path_type):
        if path_type == "AGENT":
            seg_t = self.find_segment_and_local_t(dist)
            if seg_t == (None, None):
                return 0.0
            seg, t = seg_t
            return self.slope_segment(seg, t)

        elif path_type == "BRACHISTOCHRONE":
            cumulative = self.cumulative_dist_brach
            idx = np.searchsorted(cumulative, dist, side="right") - 1
            if idx >= len(self.x_brach) - 1:
                idx = len(self.x_brach) - 2
            dx = self.x_brach[idx + 1] - self.x_brach[idx]
            dy = self.y_brach[idx + 1] - self.y_brach[idx]
            return np.arctan2(dy, dx)

        else:
            raise ValueError(f"Unknown path_type: {path_type}")

    def total_length(self, path_type):
        if path_type == "AGENT":
            return self.total_agent_length
        elif path_type == "BRACHISTOCHRONE":
            return self.total_brach_length
        else:
            raise ValueError(f"Unknown path_type: {path_type}")

    def partial_agent_path(self, current_distance, num_samples_per_segment=50):
        xs = []
        ys = []
        remaining = current_distance
        for seg in range(self.segments_built):
            ts = np.linspace(0, 1, num_samples_per_segment)
            segment_points = np.array([self.evaluate_segment(seg, t) for t in ts])
            dx = np.diff(segment_points[:, 0])
            dy = np.diff(segment_points[:, 1])
            seg_lengths = np.sqrt(dx ** 2 + dy ** 2)
            cum_lengths = np.cumsum(seg_lengths)
            seg_length = cum_lengths[-1]

            if remaining < seg_length:
                idx = np.searchsorted(cum_lengths, remaining)
                idx = np.maximum(idx, 0)
                xs.extend(segment_points[:idx + 1, 0].tolist())
                ys.extend(segment_points[:idx + 1, 1].tolist())

                # interpolate last point to exactly current_distance
                if idx > 0:
                    excess = remaining - cum_lengths[idx - 1]
                    segment_fraction = excess / seg_lengths[idx - 1] if seg_lengths[idx - 1] > 0 else 0
                    x_last = segment_points[idx - 1, 0] + segment_fraction * dx[idx - 1]
                    y_last = segment_points[idx - 1, 1] + segment_fraction * dy[idx - 1]
                    xs.append(x_last)
                    ys.append(y_last)
                break
            else:
                xs.extend(segment_points[:, 0].tolist())
                ys.extend(segment_points[:, 1].tolist())
                remaining -= seg_length

        return np.array(xs), np.array(ys)