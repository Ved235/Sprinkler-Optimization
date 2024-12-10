import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import direct
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from Polynomial_Interpolation import SprinklerAnalysis


class Opt:
    def __init__(self, verts, w, h, rate=40, num=3, analyz:SprinklerAnalysis=None):
        self.verts = verts
        self.path = Path(verts)
        self.w = w
        self.h = h
        self.rate = rate
        self.num = num
        self.analyz = analyz

    def ar(self, dists, ht):
        dists = np.asarray(dists)
        ints = np.vectorize(self.analyz.height_adjusted_intensity_function)(x=dists, y=0, h=ht)
        return np.maximum(0, ints)

    def ent(self, vals):
        eps = 1e-10
        tot = np.sum(vals)
        if tot == 0:
            return 1e6
        probs = vals / tot
        return -np.sum(probs * np.log(probs + eps))

    def std(self, vals):
        devs = vals - self.rate
        sqr = devs ** 2
        var = np.mean(sqr)
        return np.sqrt(var)

    def cuc(self, vals):
        tot_dev = np.sum(np.abs(vals - self.rate))
        return 1 - (tot_dev / (len(vals) * self.rate))

    def cov(self, vals):
        return self.std(vals) / np.mean(vals)

    def obj(self, params):
        coords = params[:2 * self.num].reshape((-1, 2))
        heights = params[2 * self.num:]
        for coord in coords:
            if not self.path.contains_point(coord):
                return 1e6
        x_vals = np.linspace(0, self.w, 50)
        y_vals = np.linspace(0, self.h, 50)
        X, Y = np.meshgrid(x_vals, y_vals)
        points = np.column_stack((X.ravel(), Y.ravel()))
        inside = np.array([self.path.contains_point(pt) for pt in points])
        points = points[inside]
        vals = np.zeros(points.shape[0])
        for i, (xi, yi) in enumerate(coords):
            hi = heights[i]
            dists = np.linalg.norm(points - np.array([xi, yi]), axis=1)
            vals += self.ar(dists, hi)
        return -self.cuc(vals)

    def opt(self):
        bounds = [(0, self.w), (0, self.h)] * self.num + [(0.4826, 0.6858)] * self.num
        result = differential_evolution(
            self.obj,
            bounds,
            strategy='best1bin',
            maxiter=500,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=None,
            workers=-1
        )
        return result

    def viz(self, result):
        params = result.x
        coords = params[:2 * self.num].reshape((-1, 2))
        heights = params[2 * self.num:]
        print("\nResults:")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.nit}")
        print(f"Objective: {result.fun:.4f}")
        print("\nPositions and Heights:")
        for i, (x, y, h) in enumerate(zip(coords[:, 0], coords[:, 1], heights)):
            print(f"Sprinkler {i + 1}: ({x:.2f}, {y:.2f}), Height={h:.2f}")
        fig, ax = plt.subplots(figsize=(10, 8))
        patch = patches.Polygon(self.verts, closed=True, edgecolor='black', facecolor='none', lw=2)
        ax.add_patch(patch)
        x_vals = np.linspace(0, self.w, 50)
        y_vals = np.linspace(0, self.h, 50)
        X, Y = np.meshgrid(x_vals, y_vals)
        points = np.column_stack((X.ravel(), Y.ravel()))
        inside = np.array([self.path.contains_point(pt) for pt in points])
        points = points[inside]
        vals = np.zeros(points.shape[0])
        for i, (xi, yi) in enumerate(coords):
            dists = np.linalg.norm(points - np.array([xi, yi]), axis=1)
            vals += self.ar(dists, heights[i])
        imap = np.zeros(X.shape)
        imap[inside.reshape(X.shape)] = vals
        print(f"\nEntropy: {self.ent(vals):.4f}")
        print(f"\nStd Dev: {self.std(vals):.4f}")
        print(f"\nCOV: {self.cov(vals):.4f}")
        print(f"\nSD: {np.std(vals):.4f}")
        print(f"\CUC: {self.cuc(vals):.4f}")
        contour = ax.contourf(X, Y, imap, levels=20, cmap="viridis", alpha=0.7)
        fig.colorbar(contour, ax=ax, label="Rate (mm/hr)")
        for i, (x, y) in enumerate(coords):
            ax.scatter(x, y, label=f"Sprinkler {i + 1} (H={heights[i]:.2f})", s=100)
        ax.set_xlim(0, self.w)
        ax.set_ylim(0, self.h)
        ax.set_title("Optimized Positions and Distribution")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.hist(vals, bins=20, edgecolor='black', color='skyblue')
        ax2.axvline(x=self.rate, color='red', linestyle='--', label=f'Target ({self.rate} mm/hr)')
        ax2.set_title("Distribution of Rates")
        ax2.set_xlabel("Rate (mm/hr)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        plt.show()

if __name__ == "__main__":

    verts = [(0, 0), (12.192, 0), (12.192, 15.24), (0, 15.24)]
    w = 12.192
    h = 15.24
    analyz = SprinklerAnalysis(base_height=0.4826)
    data = [
        {'x': 0, 'y': 0, 'volume_ml': 0, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
        {'x': 1, 'y': 0, 'volume_ml': 250, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
        {'x': 3, 'y': 0, 'volume_ml': 300, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
        {'x': 4, 'y': 0, 'volume_ml': 350, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
        {'x': 6, 'y': 0, 'volume_ml': 550, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
        {'x': 7, 'y': 0, 'volume_ml': 0, 'can_area_cm2': 100, 'duration_hr': 1.0},
    ]
    analyz.fit_model(data)
    opt = Opt(verts, w, h, analyz=analyz)
    res = opt.opt()
    opt.viz(res)

