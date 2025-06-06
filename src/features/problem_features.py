"""
Problem feature extraction for optimization problems.

This module provides functionality to extract various features from optimization
problems that can be used to characterize the problem landscape and difficulty.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Callable
import warnings

import sys
import os

# Add the src directory to the Python path
current_dir = os.path.dirname(__file__)
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from benchmarks.continuous_functions import OptimizationFunction


class ProblemFeatureExtractor:
    """
    Extract features from optimization problems to characterize their properties.
    
    This class implements various feature extraction methods to analyze the
    landscape and characteristics of optimization problems.
    """
    
    def __init__(self, n_samples: int = 1000, random_state: int = 42):
        """
        Initialize the problem feature extractor.
        
        Args:
            n_samples: Number of samples to use for landscape analysis
            random_state: Random seed for reproducible results
        """
        self.n_samples = n_samples
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def extract_features(self, problem: OptimizationFunction, dimension: int) -> Dict[str, float]:
        """
        Extract comprehensive features from a problem.
        
        Args:
            problem: The optimization function to analyze
            dimension: Dimensionality of the problem
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        # Basic problem properties
        features.update(self._extract_basic_features(problem, dimension))
        
        # Statistical features from sampling
        features.update(self._extract_statistical_features(problem, dimension))
        
        # Landscape features
        features.update(self._extract_landscape_features(problem, dimension))
        
        # Meta-features
        features.update(self._extract_meta_features(problem, dimension))
        
        return features
    
    def _extract_basic_features(self, problem: OptimizationFunction, dimension: int) -> Dict[str, float]:
        """Extract basic problem properties."""
        features = {
            'dimension': float(dimension),
            'separable': float(problem.separable),
            'global_optimum': float(problem.global_optimum),
            'bound_width': float(problem.bounds[1] - problem.bounds[0]),
            'bound_min': float(problem.bounds[0]),
            'bound_max': float(problem.bounds[1])
        }
        
        return features
    
    def _extract_statistical_features(self, problem: OptimizationFunction, dimension: int) -> Dict[str, float]:
        """Extract statistical features from random sampling."""
        # Generate random samples within bounds
        bounds = problem.bounds
        samples = self.rng.uniform(bounds[0], bounds[1], (self.n_samples, dimension))
        
        # Evaluate function at all samples
        values = []
        for sample in samples:
            try:
                value = problem(sample)
                if np.isfinite(value):
                    values.append(value)
            except:
                continue
        
        if len(values) == 0:
            return {'stats_error': 1.0}
        
        values = np.array(values)
        
        features = {
            'fitness_mean': float(np.mean(values)),
            'fitness_std': float(np.std(values)),
            'fitness_min': float(np.min(values)),
            'fitness_max': float(np.max(values)),
            'fitness_median': float(np.median(values)),
            'fitness_skewness': float(self._skewness(values)),
            'fitness_kurtosis': float(self._kurtosis(values)),
            'fitness_range': float(np.max(values) - np.min(values)),
            'fitness_iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
            'valid_samples_ratio': float(len(values) / self.n_samples)
        }
        
        return features
    
    def _extract_landscape_features(self, problem: OptimizationFunction, dimension: int) -> Dict[str, float]:
        """Extract landscape characterization features."""
        features = {}
        
        # Sample points for landscape analysis
        bounds = problem.bounds
        n_landscape_samples = min(500, self.n_samples // 2)
        samples = self.rng.uniform(bounds[0], bounds[1], (n_landscape_samples, dimension))
        
        # Calculate gradients (finite differences)
        gradients = self._estimate_gradients(problem, samples, dimension)
        
        if len(gradients) > 0:
            gradient_norms = np.linalg.norm(gradients, axis=1)
            
            features.update({
                'gradient_mean_norm': float(np.mean(gradient_norms)),
                'gradient_std_norm': float(np.std(gradient_norms)),
                'gradient_max_norm': float(np.max(gradient_norms)),
                'gradient_min_norm': float(np.min(gradient_norms)),
                'ruggedness': float(np.std(gradient_norms) / (np.mean(gradient_norms) + 1e-10))
            })
        
        # Multimodality estimation
        features.update(self._estimate_multimodality(problem, dimension))
        
        return features
    
    def _extract_meta_features(self, problem: OptimizationFunction, dimension: int) -> Dict[str, float]:
        """Extract meta-features that describe problem characteristics."""
        features = {}
        
        # Problem complexity indicators
        features['problem_complexity'] = float(dimension * np.log(dimension + 1))
        features['search_space_volume'] = float((problem.bounds[1] - problem.bounds[0]) ** dimension)
        
        # Problem type encoding (based on function name)
        problem_name = problem.name.lower()
        features['is_sphere'] = float('sphere' in problem_name)
        features['is_rastrigin'] = float('rastrigin' in problem_name)
        features['is_rosenbrock'] = float('rosenbrock' in problem_name)
        features['is_ackley'] = float('ackley' in problem_name)
        features['is_griewank'] = float('griewank' in problem_name)
        
        return features
    
    def _estimate_gradients(self, problem: OptimizationFunction, samples: np.ndarray, dimension: int) -> np.ndarray:
        """Estimate gradients using finite differences."""
        gradients = []
        epsilon = 1e-6
        
        for i, sample in enumerate(samples[:100]):  # Limit for computational efficiency
            try:
                grad = np.zeros(dimension)
                f_x = problem(sample)
                
                if not np.isfinite(f_x):
                    continue
                
                for d in range(dimension):
                    sample_plus = sample.copy()
                    sample_plus[d] += epsilon
                    
                    # Check bounds
                    if sample_plus[d] > problem.bounds[1]:
                        sample_plus[d] = problem.bounds[1]
                    
                    try:
                        f_x_plus = problem(sample_plus)
                        if np.isfinite(f_x_plus):
                            grad[d] = (f_x_plus - f_x) / epsilon
                    except:
                        grad[d] = 0.0
                
                gradients.append(grad)
                
            except:
                continue
        
        return np.array(gradients) if gradients else np.array([])
    
    def _estimate_multimodality(self, problem: OptimizationFunction, dimension: int) -> Dict[str, float]:
        """Estimate multimodality characteristics."""
        features = {}
        
        # Use fewer samples for efficiency
        n_modal_samples = min(200, self.n_samples // 5)
        bounds = problem.bounds
        samples = self.rng.uniform(bounds[0], bounds[1], (n_modal_samples, dimension))
        
        values = []
        for sample in samples:
            try:
                value = problem(sample)
                if np.isfinite(value):
                    values.append(value)
            except:
                continue
        
        if len(values) < 10:
            return {'multimodality_error': 1.0}
        
        values = np.array(values)
        
        # Estimate number of modes using simple binning
        n_bins = min(20, len(values) // 5)
        hist, bin_edges = np.histogram(values, bins=n_bins)
        
        # Count local maxima in histogram (potential modes)
        local_maxima = 0
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0:
                local_maxima += 1
        
        features['estimated_modes'] = float(max(1, local_maxima))
        features['modality_ratio'] = float(local_maxima / n_bins)
        
        return features
    
    def _skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of the distribution."""
        try:
            mean = np.mean(values)
            std = np.std(values)
            if std == 0:
                return 0.0
            return float(np.mean(((values - mean) / std) ** 3))
        except:
            return 0.0
    
    def _kurtosis(self, values: np.ndarray) -> float:
        """Calculate kurtosis of the distribution."""
        try:
            mean = np.mean(values)
            std = np.std(values)
            if std == 0:
                return 0.0
            return float(np.mean(((values - mean) / std) ** 4) - 3)
        except:
            return 0.0 