import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from typing import List, Dict, Callable
from dataclasses import dataclass

@dataclass
class CatchCanMeasurement:
    x: float
    y: float
    volume_ml: float
    can_area_cm2: float
    duration_hr: float

class SprinklerAnalysis:
    def __init__(self, base_height: float):
        """
        Initialize the SprinklerAnalysis with a base height.
        
        Args:
            base_height (float): The base height (h0) in meters for intensity calculations
        """
        self.base_height = base_height
        self.base_intensity_function = None
        self.height_adjusted_intensity_function = None
        
    def convert_to_intensity(self, volume_ml: float, can_area_cm2: float, 
                           test_duration_hr: float) -> float:
        """
        Convert catch can measurements to intensity (mm/hr).
        
        Args:
            volume_ml: Volume collected in milliliters
            can_area_cm2: Catch can area in square centimeters
            test_duration_hr: Test duration in hours
            
        Returns:
            float: Intensity in mm/hr
        """
        depth_mm = (volume_ml / can_area_cm2) * 10
        return depth_mm / test_duration_hr

    def radial_polynomial(self, r: np.ndarray, *coeffs) -> np.ndarray:
        """Calculate radial polynomial value."""
        
        return np.sum([c * r**i for i, c in enumerate(coeffs)], axis=0)

    def create_intensity_function(self, experimental_data: np.ndarray, degree: int = 4) -> Callable:
        """
        Creates an intensity function using polynomial fit.
        
        Args:
            experimental_data: Array of [x, y, intensity] measurements
            degree: Degree of polynomial fit
            
        Returns:
            Callable: Function that returns intensity at any point (x, y)
        """
        points = np.array(experimental_data)
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        intensities = points[:, 2]
        print("intensities", intensities)
    
        self.coeffs, _ = curve_fit(self.radial_polynomial, distances, intensities, 
                                 p0=np.zeros(degree + 1))
        self.max_distance = np.max(distances)
        
        return self.intensity_at_point
    
    def intensity_at_point(self, x: float, y: float) -> float:
        """Calculate intensity at a given point using stored coefficients."""
        r = np.sqrt(x**2 + y**2)
        if r > self.max_distance:
            return 0
        return max(0, self.radial_polynomial(r, *self.coeffs))

    def create_height_adjusted_intensity_function(self) -> Callable:
        """
        Creates a height-adjusted intensity function based on the base intensity function.
        
        Returns:
            Callable: Function that returns height-adjusted intensity at any point (x, y, h)
        """
        return self.height_adjusted_intensity

    def height_adjusted_intensity(self, x: float, y: float, h: float) -> float:
        """Calculate height-adjusted intensity at a given point."""
        r = np.sqrt(x**2 + y**2)
        scaled_r = r * np.sqrt(self.base_height / h)
        base_intensity = self.base_intensity_function(scaled_r, 0)
        return (self.base_height / h) * base_intensity

    def process_catch_can_data(self, 
                             measurements: List[Dict]) -> List[List[float]]:
        """
        Process raw catch can measurements into format needed for intensity function.
        
        Args:
            measurements: List of dictionaries containing measurement data
            
        Returns:
            List[List[float]]: Processed data in format [[x, y, intensity], ...]
        """
        processed_data = []
        for measurement in measurements:
            intensity = self.convert_to_intensity(
                volume_ml=measurement['volume_ml'],
                can_area_cm2=measurement['can_area_cm2'],
                test_duration_hr=measurement['duration_hr']
            )
            processed_data.append([measurement['x'], measurement['y'], intensity])
        return processed_data

    def fit_model(self, catch_can_measurements: List[Dict]) -> None:
        """
        Fit the intensity model using catch can measurements.
        
        Args:
            catch_can_measurements: List of dictionaries containing measurement data
        """
        processed_data = self.process_catch_can_data(catch_can_measurements)
        self.base_intensity_function = self.create_intensity_function(processed_data)
        self.height_adjusted_intensity_function = \
            self.create_height_adjusted_intensity_function()

    def plot_intensity_distribution(self, height: float, max_range: float = 5):
        if self.height_adjusted_intensity_function is None:
            raise ValueError("Model must be fit before plotting")
            
        x = np.linspace(-max_range, max_range, 200)
        y = np.linspace(-max_range, max_range, 200)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.vectorize(lambda x, y: self.height_adjusted_intensity_function(x, y, height))(X, Y)

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(contour, label='Intensity (mm/hr)')
        plt.xlabel('Distance (m)')
        plt.ylabel('Distance (m)')
        plt.title(f'Intensity Distribution at Height h={height}m')
        plt.show()


if __name__ == "__main__":
    raw_measurements = [
    {'x': 0, 'y': 0, 'volume_ml': 0, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
    {'x': 1, 'y': 0, 'volume_ml': 250, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
    {'x': 3, 'y': 0, 'volume_ml': 300, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
    {'x': 4, 'y': 0, 'volume_ml': 350, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
    {'x': 6, 'y': 0, 'volume_ml': 550, 'can_area_cm2': 530.9, 'duration_hr': 0.25},
    {'x': 7, 'y': 0, 'volume_ml': 0, 'can_area_cm2': 100, 'duration_hr': 1.0},
    ]
    
    analyzer = SprinklerAnalysis(base_height=0.4826)
    
    analyzer.fit_model(raw_measurements)
    rounded_coeffs = [round(c, 2) for c in analyzer.coeffs]
    print("Rounded Coefficients:", rounded_coeffs)

    analyzer.plot_intensity_distribution(height=0.4826, max_range=10)
