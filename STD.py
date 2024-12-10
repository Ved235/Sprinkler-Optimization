import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from Polynomial_Interpolation import SprinklerAnalysis
import time

class WaterDistribution:
    def __init__(self, vertices, width, height, target_rate=40, num_sprinklers=3, evaluator: SprinklerAnalysis = None):
        self.vertices = vertices
        self.boundary = Path(vertices)
        self.width = width
        self.height = height
        self.target_rate = target_rate
        self.num_sprinklers = num_sprinklers
        self.evaluator = evaluator

    def distribution_rate(self, dists, h):
        dists = np.asarray(dists)
        rates = np.vectorize(self.evaluator.height_adjusted_intensity_function)(x=dists, y=0, h=h)
        return np.maximum(0, rates)

    def calc_entropy(self, vals):
        eps = 1e-10
        total = np.sum(vals)
        if total == 0:
            return 1e6
        probs = vals / total
        return -np.sum(probs * np.log(probs + eps))

    def deviation(self, vals):
        return np.sqrt(np.mean((vals - self.target_rate) ** 2))

    def coeff_variance(self, vals):
        return self.deviation(vals) / np.mean(vals)

    def obj_func(self, params):
        points = params[:2 * self.num_sprinklers].reshape((-1, 2))
        heights = params[2 * self.num_sprinklers:]
        for point in points:
            if not self.boundary.contains_point(point):
                return 1e6
        x_grid = np.linspace(0, self.width, 50)
        y_grid = np.linspace(0, self.height, 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid = np.column_stack((X.ravel(), Y.ravel()))
        in_field = np.array([self.boundary.contains_point(p) for p in grid])
        grid = grid[in_field]
        rates = np.zeros(grid.shape[0])
        for i, (x, y) in enumerate(points):
            rates += self.distribution_rate(np.linalg.norm(grid - np.array([x, y]), axis=1), heights[i])
        return self.deviation(rates)

    def optimize_layout(self):
        bounds = [(0, self.width), (0, self.height)] * self.num_sprinklers + [(0.5, 0.7)] * self.num_sprinklers
        return differential_evolution(self.obj_func, bounds, strategy='best1bin', maxiter=200, popsize=15, tol=0.01,
                                      mutation=(0.5, 1.5), recombination=0.7, workers=-1)

    def show_results(self, res):
        params = res.x
        points = params[:2 * self.num_sprinklers].reshape((-1, 2))
        heights = params[2 * self.num_sprinklers:]
        print("\nResults:")
        print(f"Success: {res.success}")
        print(f"Iterations: {res.nit}")
        print(f"Objective: {res.fun:.4f}")
        print("\nSprinklers:")
        for i, (x, y, h) in enumerate(zip(points[:, 0], points[:, 1], heights)):
            print(f"Sprinkler {i + 1}: ({x:.2f}, {y:.2f}), Height={h:.2f}")
        fig, ax = plt.subplots(figsize=(10, 8))
        field_patch = patches.Polygon(self.vertices, closed=True, edgecolor='black', facecolor='none', lw=2)
        ax.add_patch(field_patch)
        x_grid = np.linspace(0, self.width, 50)
        y_grid = np.linspace(0, self.height, 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid = np.column_stack((X.ravel(), Y.ravel()))
        in_field = np.array([self.boundary.contains_point(p) for p in grid])
        grid = grid[in_field]
        rates = np.zeros(grid.shape[0])
        for i, (x, y) in enumerate(points):
            rates += self.distribution_rate(np.linalg.norm(grid - np.array([x, y]), axis=1), heights[i])
        intensity_map = np.zeros(X.shape)
        intensity_map[in_field.reshape(X.shape)] = rates
        print(f"\nEntropy: {self.calc_entropy(rates):.4f}")
        print(f"\nDeviation: {self.deviation(rates):.4f}")
        print(f"\nCoeff of Variance: {self.coeff_variance(rates):.4f}")
        print(f"\nStd Dev: {np.std(rates):.4f}")
        contour = ax.contourf(X, Y, intensity_map, levels=20, cmap="viridis", alpha=0.7)
        fig.colorbar(contour, ax=ax, label="Rate (mm/hr)")
        for i, (x, y) in enumerate(points):
            ax.scatter(x, y, label=f"Sprinkler {i + 1} (Height={heights[i]:.2f})", s=100)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_title("Optimized Sprinkler Layout")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.hist(rates, bins=20, edgecolor='black', color='skyblue')
        ax2.axvline(x=self.target_rate, color='red', linestyle='--', label=f'Target ({self.target_rate} mm/hr)')
        ax2.set_title("Rate Distribution")
        ax2.set_xlabel("Rate (mm/hr)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        plt.show()


if __name__ == "__main__":
    start = time.time()
    vertices = [(0, 0), (10, 0), (10, 5), (5, 5), (3, 7), (0, 5)]
    width = 10
    height = 7
    evaluator = SprinklerAnalysis(base_height=0.4826)
    data = [
        {'x': 0, 'y': 0, 'volume_ml': 0, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
        {'x': 1, 'y': 0, 'volume_ml': 250, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
        {'x': 3, 'y': 0, 'volume_ml': 300, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
        {'x': 4, 'y': 0, 'volume_ml': 350, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
        {'x': 6, 'y': 0, 'volume_ml': 550, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
        {'x': 7, 'y': 0, 'volume_ml': 0, 'can_area_cm2': 100, 'duration_hr': 1.0},
    ]
    evaluator.fit_model(data)
    dist_optimizer = WaterDistribution(vertices, width, height, evaluator=evaluator)
    res = dist_optimizer.optimize_layout()
    dist_optimizer.show_results(res)
