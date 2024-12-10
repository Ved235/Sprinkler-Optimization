# Sprinkler System Optimization

This project aims to optimize the layout and intensity distribution of sprinklers in a given area using polynomial interpolation and differential evolution. The project includes classes and methods to analyze sprinkler data, fit intensity models, and optimize sprinkler positions and heights.

## Project Structure

```
.gitattributes
CUC.py
Polynomial_Interpolation.py
STD.py
```

### Files

- **CUC.py**: Contains the `Opt` class for optimizing sprinkler layout using differential evolution.
- **Polynomial_Interpolation.py**: Contains the `SprinklerAnalysis` class for analyzing sprinkler data and fitting polynomial intensity models.
- **STD.py**: Contains the `WaterDistribution` class for optimizing and visualizing the sprinkler distribution.

## Installation

1. Clone the repository.
2. Install the required dependencies using pip:

    ```sh
    pip install numpy scipy matplotlib
    ```

## Usage

### Polynomial Interpolation

The `SprinklerAnalysis` class in `Polynomial_Interpolation.py` is used to analyze sprinkler data and fit polynomial intensity models.

#### Example

```python
from Polynomial_Interpolation import SprinklerAnalysis

raw_measurements = [
    {'x': 0, 'y': 0, 'volume_ml': 0, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
    {'x': 1, 'y': 0, 'volume_ml': 250, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
    # ...
]

analyzer = SprinklerAnalysis(base_height=0.4826)
analyzer.fit_model(raw_measurements)
analyzer.plot_intensity_distribution(height=0.4826, max_range=10)
```

### Optimization

The `Opt` class in `CUC.py` and the `WaterDistribution` class in `STD.py` are used to optimize the sprinkler layout.

## Classes and Methods

### `SprinklerAnalysis` (in `Polynomial_Interpolation.py`)

- **`__init__(self, base_height: float)`**: Initialize the analysis with a base height.
- **`convert_to_intensity(self, volume_ml: float, can_area_cm2: float, test_duration_hr: float) -> float`**: Convert catch can measurements to intensity.
- **`radial_polynomial(self, r: np.ndarray, *coeffs) -> np.ndarray`**: Calculate radial polynomial value.
- **`create_intensity_function(self, experimental_data: np.ndarray, degree: int = 4) -> Callable`**: Create an intensity function using polynomial fit.
- **`intensity_at_point(self, x: float, y: float) -> float`**: Calculate intensity at a given point.
- **`create_height_adjusted_intensity_function(self) -> Callable`**: Create a height-adjusted intensity function.
- **`height_adjusted_intensity(self, x: float, y: float, h: float) -> float`**: Calculate height-adjusted intensity at a given point.
- **`process_catch_can_data(self, measurements: List[Dict]) -> List[List[float]]`**: Process raw catch can measurements.
- **`fit_model(self, catch_can_measurements: List[Dict]) -> None`**: Fit the intensity model using catch can measurements.
- **`plot_intensity_distribution(self, height: float, max_range: float = 5)`**: Plot the intensity distribution.

### `Opt` (in `CUC.py`)

- **`__init__(self, verts, w, h, rate=40, num=3, analyz: SprinklerAnalysis = None)`**: Initialize the optimizer.
- **`ar(self, dists, ht)`**: Calculate adjusted rates.
- **`ent(self, vals)`**: Calculate entropy.
- **`std(self, vals)`**: Calculate standard deviation.
- **`cuc(self, vals)`**: Calculate coefficient of uniformity.
- **`cov(self, vals)`**: Calculate coefficient of variance.
- **`obj(self, params)`**: Objective function for optimization.
- **`opt(self)`**: Perform optimization using differential evolution.
- **`viz(self, result)`**: Visualize the optimization results.

### `WaterDistribution` (in `STD.py`)

- **`__init__(self, vertices, width, height, target_rate=40, num_sprinklers=3, evaluator: SprinklerAnalysis = None)`**: Initialize the water distribution optimizer.
- **`distribution_rate(self, dists, h)`**: Calculate distribution rate.
- **`calc_entropy(self, vals)`**: Calculate entropy.
- **`deviation(self, vals)`**: Calculate deviation.
- **`coeff_variance(self, vals)`**: Calculate coefficient of variance.
- **`obj_func(self, params)`**: Objective function for optimization.
- **`optimize_layout(self)`**: Optimize the sprinkler layout.
- **`show_results(self, res)`**: Show the optimization results.

## Acknowledgements

- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Matplotlib](https://matplotlib.org/)
