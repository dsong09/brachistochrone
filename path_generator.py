import numpy as np
from scipy.optimize import fsolve

START_POINT = (0.0, 0.0)
END_POINT = (9.0, 6.0)

class PathGenerator:
    def __init__(self, start=START_POINT, end=END_POINT, samples_per_seg=100, ds=0.01):
        self.start = np.array(start, dtype=float)
        self.end   = np.array(end, dtype=float)

        # Agent path (piecewise cubic Beziers)
        self.segments = []   # list of (4,2) arrays: [p0, p1, p2, p3]
        self.arc_table = []  # per-segment lookup: {"ts","cumlen","length"}
        self.total_agent_length = 0.0
        self.samples_per_seg = int(samples_per_seg)
        self.ds = float(ds)  # target arc table resolution

        # Brachistochrone (precomputed once)
        self.brach_pts, self.brach_cumlen, self.theta_max, self.a_brach = self._build_brachistochrone(samples=1200)

    # -------------------- Agent path (Bezier) --------------------
    def add_segment(self, cp1, cp2, p3):
        cp1 = np.array(cp1, dtype=float)
        cp2 = np.array(cp2, dtype=float)
        p3  = np.array(p3, dtype=float)

        if not self.segments:
            p0 = self.start.copy()
        else:
            p0 = self.segments[-1][3].copy()

        seg = np.vstack([p0, cp1, cp2, p3])
        self.segments.append(seg)
        self._rebuild_agent_arc_table()

    def random_segment(self, segment_length):
        """
        Generate a random cubic Bézier segment with:
          - fixed segment_length
          - maximum randomness (can go backward, up, or down)
          - G¹ continuity with previous segment
        """
        # Determine last endpoint
        if self.segments:
            last_endpoint = self.segments[-1][3].copy()
        else:
            last_endpoint = self.start.copy()

        # Random direction anywhere (-pi to pi)
        angle = np.random.uniform(-np.pi, np.pi)
        p3 = last_endpoint + segment_length * np.array([np.cos(angle), np.sin(angle)])

        # G¹ continuity for cp1
        if self.segments:
            prev = self.segments[-1]
            cp1 = 2 * last_endpoint - prev[2]  # reflect previous p2
            # Add some jitter for exploration
            cp1 += np.random.uniform(-0.5, 0.5, size=2) * segment_length
        else:
            # First segment: completely random control point
            cp1 = last_endpoint + np.random.uniform(-0.5, 0.5, size=2) * segment_length

        # cp2: near midpoint, add more jitter
        mid = 0.5 * (last_endpoint + p3)
        cp2 = mid + np.random.uniform(-0.5, 0.5, size=2) * segment_length

        return cp1, cp2, p3

    def _rebuild_agent_arc_table(self):
        self.arc_table = []
        total = 0.0
        for seg in self.segments:
            ts = np.linspace(0.0, 1.0, self.samples_per_seg)
            pts = self._evaluate_many(seg, ts)
            deltas = np.diff(pts, axis=0)
            seglens = np.sqrt((deltas ** 2).sum(axis=1))
            cumlen = np.concatenate([[0.0], np.cumsum(seglens)])
            self.arc_table.append({"ts": ts, "cumlen": cumlen, "length": float(cumlen[-1])})
            total += float(cumlen[-1])
        self.total_agent_length = total

    @staticmethod
    def _evaluate(seg, t):
        p0, p1, p2, p3 = seg
        omt = 1.0 - t
        return (omt**3) * p0 + 3*(omt**2)*t * p1 + 3*omt*(t**2) * p2 + (t**3) * p3

    @staticmethod
    def _evaluate_many(seg, ts):
        ts = np.asarray(ts)
        OM = (1.0 - ts)[:, None]
        T  = ts[:, None]
        return (OM**3) * seg[0] + 3*(OM**2)*T * seg[1] + 3*OM*(T**2) * seg[2] + (T**3) * seg[3]

    # -------------------- Brachistochrone (unchanged) --------------------
    def _build_brachistochrone(self, samples=1200):
        x0, y0 = self.start
        x1, y1 = self.end
        dx = x1 - x0
        dy = y1 - y0
        if dy <= 0 or dx == 0:
            pts = np.column_stack([np.linspace(x0, x1, samples), np.linspace(y0, y1, samples)])
            deltas = np.diff(pts, axis=0)
            cumlen = np.concatenate([[0.0], np.cumsum(np.sqrt((deltas**2).sum(axis=1)))])
            return pts, cumlen, 0.0, 0.0

        ratio = dx / dy
        def eq(theta): return (theta - np.sin(theta)) / (1 - np.cos(theta)) - ratio
        theta_guess = 3.0
        try:
            theta_max = float(fsolve(eq, theta_guess, maxfev=10000)[0])
        except Exception:
            theta_max = None
        if theta_max is None or not np.isfinite(theta_max) or theta_max <= 1e-6:
            grid = np.linspace(1e-4, 30.0, 40000)
            vals = (grid - np.sin(grid)) / (1 - np.cos(grid))
            idx = np.argmin(np.abs(vals - ratio))
            theta_max = float(grid[idx])
        a = dy / (1 - np.cos(theta_max))
        thetas = np.linspace(0.0, theta_max, samples)
        x = x0 + a * (thetas - np.sin(thetas))
        y = y0 + a * (1 - np.cos(thetas))
        pts = np.column_stack([x, y])
        deltas = np.diff(pts, axis=0)
        cumlen = np.concatenate([[0.0], np.cumsum(np.sqrt((deltas**2).sum(axis=1)))])
        return pts, cumlen, theta_max, a

    # -------------------- Unified API --------------------
    def total_length(self, path_type="AGENT"):
        pt = path_type.upper()
        if pt == "AGENT": return float(self.total_agent_length)
        elif pt == "BRACHISTOCHRONE": return float(self.brach_cumlen[-1])
        else: raise ValueError(f"Unknown path_type: {path_type}")

    def position_from_distance(self, s, path_type="AGENT"):
        pt = path_type.upper()
        if pt == "AGENT":
            if not self.segments: return self.start.copy()
            s = float(np.clip(s, 0.0, self.total_agent_length))
            rem = s
            for seg_idx, seg_data in enumerate(self.arc_table):
                L = seg_data["length"]
                if rem <= L:
                    cum = seg_data["cumlen"]
                    ts = seg_data["ts"]
                    i = int(np.searchsorted(cum, rem) - 1)
                    i = int(np.clip(i, 0, len(cum)-2))
                    frac = 0.0 if cum[i+1]==cum[i] else (rem-cum[i])/(cum[i+1]-cum[i])
                    t = ts[i] + frac*(ts[i+1]-ts[i])
                    return self._evaluate(self.segments[seg_idx], t)
                rem -= L
            return self.segments[-1][3].copy()
        elif pt == "BRACHISTOCHRONE":
            s = float(np.clip(s, 0.0, self.brach_cumlen[-1]))
            i = int(np.searchsorted(self.brach_cumlen, s)-1)
            i = int(np.clip(i, 0, len(self.brach_cumlen)-2))
            c0, c1 = self.brach_cumlen[i], self.brach_cumlen[i+1]
            frac = 0.0 if c1==c0 else (s-c0)/(c1-c0)
            return (1-frac)*self.brach_pts[i] + frac*self.brach_pts[i+1]
        else:
            raise ValueError(f"Unknown path_type: {path_type}")

    def slope_from_distance(self, s, path_type="AGENT", h=1e-4):
        p1 = self.position_from_distance(s, path_type)
        s2 = min(s+h, self.total_length(path_type))
        p2 = self.position_from_distance(s2, path_type)
        dx, dy = (p2 - p1)
        return np.arctan2(dy, dx)
