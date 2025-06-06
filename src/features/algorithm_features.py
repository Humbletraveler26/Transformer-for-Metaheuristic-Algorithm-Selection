"""
Algorithm feature extraction for metaheuristic performance analysis.

This module provides functionality to extract features from algorithm performance
data that can be used to characterize algorithm behavior and effectiveness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import asdict

import sys
import os

# Add the src directory to the Python path
current_dir = os.path.dirname(__file__)
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from metaheuristics.base import OptimizationResult


class AlgorithmFeatureExtractor:
    """
    Extract features from algorithm performance data.
    
    This class implements various feature extraction methods to analyze
    algorithm performance characteristics and behavior patterns.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize the algorithm feature extractor.
        
        Args:
            window_size: Window size for convergence analysis
        """
        self.window_size = window_size
    
    def extract_features(self, results: List[OptimizationResult], 
                        algorithm_name: str = None) -> Dict[str, float]:
        """
        Extract comprehensive features from algorithm results.
        
        Args:
            results: List of optimization results from multiple runs
            algorithm_name: Name of the algorithm (optional)
            
        Returns:
            Dictionary containing extracted features
        """
        if not results:
            return {'no_results_error': 1.0}
        
        features = {}
        
        # Performance features
        features.update(self._extract_performance_features(results))
        
        # Convergence features
        features.update(self._extract_convergence_features(results))
        
        # Robustness features
        features.update(self._extract_robustness_features(results))
        
        # Efficiency features
        features.update(self._extract_efficiency_features(results))
        
        # Meta-features
        features.update(self._extract_meta_features(results, algorithm_name))
        
        return features
    
    def _extract_performance_features(self, results: List[OptimizationResult]) -> Dict[str, float]:
        """Extract performance-related features."""
        fitness_values = [result.best_fitness for result in results]
        
        features = {
            'best_fitness_mean': float(np.mean(fitness_values)),
            'best_fitness_std': float(np.std(fitness_values)),
            'best_fitness_min': float(np.min(fitness_values)),
            'best_fitness_max': float(np.max(fitness_values)),
            'best_fitness_median': float(np.median(fitness_values)),
            'best_fitness_range': float(np.max(fitness_values) - np.min(fitness_values)),
            'best_fitness_iqr': float(np.percentile(fitness_values, 75) - np.percentile(fitness_values, 25)),
            'best_fitness_cv': float(np.std(fitness_values) / (np.mean(fitness_values) + 1e-10))
        }
        
        # Success rate
        success_count = sum(1 for result in results if result.success)
        features['success_rate'] = float(success_count / len(results))
        
        # Target achievement analysis
        if success_count > 0:
            successful_results = [result for result in results if result.success]
            success_evaluations = [result.total_evaluations for result in successful_results]
            features['success_evaluations_mean'] = float(np.mean(success_evaluations))
            features['success_evaluations_std'] = float(np.std(success_evaluations))
        else:
            features['success_evaluations_mean'] = float(np.nan)
            features['success_evaluations_std'] = float(np.nan)
        
        return features
    
    def _extract_convergence_features(self, results: List[OptimizationResult]) -> Dict[str, float]:
        """Extract convergence-related features."""
        features = {}
        
        # Analyze convergence history if available
        convergence_curves = []
        for result in results:
            if hasattr(result, 'fitness_history') and result.fitness_history:
                convergence_curves.append(result.fitness_history)
        
        if convergence_curves:
            # Average convergence behavior
            min_length = min(len(curve) for curve in convergence_curves)
            if min_length > 1:
                # Truncate all curves to same length
                truncated_curves = [curve[:min_length] for curve in convergence_curves]
                avg_curve = np.mean(truncated_curves, axis=0)
                
                # Convergence rate (improvement per iteration)
                if len(avg_curve) > 1:
                    improvements = np.diff(avg_curve)
                    negative_improvements = improvements[improvements < 0]  # Actual improvements
                    
                    if len(negative_improvements) > 0:
                        features['avg_improvement_rate'] = float(np.mean(negative_improvements))
                        features['improvement_consistency'] = float(len(negative_improvements) / len(improvements))
                    else:
                        features['avg_improvement_rate'] = 0.0
                        features['improvement_consistency'] = 0.0
                
                # Early vs late convergence
                early_portion = avg_curve[:min_length//4] if min_length >= 4 else avg_curve[:1]
                late_portion = avg_curve[-min_length//4:] if min_length >= 4 else avg_curve[-1:]
                
                features['early_convergence_rate'] = float(np.std(early_portion) / (np.mean(early_portion) + 1e-10))
                features['late_convergence_rate'] = float(np.std(late_portion) / (np.mean(late_portion) + 1e-10))
                
                # Stagnation detection
                if len(avg_curve) >= self.window_size:
                    windows = [avg_curve[i:i+self.window_size] for i in range(len(avg_curve) - self.window_size + 1)]
                    stagnation_windows = sum(1 for window in windows if np.std(window) < 1e-8)
                    features['stagnation_ratio'] = float(stagnation_windows / len(windows))
                else:
                    features['stagnation_ratio'] = 0.0
        
        # Evaluation efficiency
        total_evaluations = [result.total_evaluations for result in results]
        features['evaluations_mean'] = float(np.mean(total_evaluations))
        features['evaluations_std'] = float(np.std(total_evaluations))
        features['evaluations_efficiency'] = float(np.mean([
            (result.best_fitness + 1e-10) / (result.total_evaluations + 1) 
            for result in results
        ]))
        
        return features
    
    def _extract_robustness_features(self, results: List[OptimizationResult]) -> Dict[str, float]:
        """Extract robustness and reliability features."""
        features = {}
        
        # Variance in performance
        fitness_values = [result.best_fitness for result in results]
        features['performance_variance'] = float(np.var(fitness_values))
        features['performance_reliability'] = float(1.0 / (1.0 + np.std(fitness_values)))
        
        # Consistency across runs
        if len(results) > 1:
            # Coefficient of variation
            cv = np.std(fitness_values) / (np.mean(fitness_values) + 1e-10)
            features['consistency_cv'] = float(cv)
            
            # Outlier detection (simple z-score based)
            z_scores = np.abs((fitness_values - np.mean(fitness_values)) / (np.std(fitness_values) + 1e-10))
            outlier_count = np.sum(z_scores > 2.0)
            features['outlier_ratio'] = float(outlier_count / len(results))
        
        # Execution time variance
        execution_times = [result.execution_time for result in results]
        features['time_mean'] = float(np.mean(execution_times))
        features['time_std'] = float(np.std(execution_times))
        features['time_reliability'] = float(1.0 / (1.0 + np.std(execution_times) / (np.mean(execution_times) + 1e-10)))
        
        return features
    
    def _extract_efficiency_features(self, results: List[OptimizationResult]) -> Dict[str, float]:
        """Extract efficiency-related features."""
        features = {}
        
        # Solution quality vs computational cost
        fitness_values = np.array([result.best_fitness for result in results])
        evaluations = np.array([result.total_evaluations for result in results])
        times = np.array([result.execution_time for result in results])
        
        # Quality per evaluation
        quality_per_eval = fitness_values / (evaluations + 1e-10)
        features['quality_per_evaluation'] = float(np.mean(quality_per_eval))
        
        # Quality per time unit
        quality_per_time = fitness_values / (times + 1e-10)
        features['quality_per_time'] = float(np.mean(quality_per_time))
        
        # Evaluation efficiency (lower is better for minimization)
        eval_efficiency = 1.0 / ((fitness_values + 1e-10) * (evaluations + 1))
        features['evaluation_efficiency'] = float(np.mean(eval_efficiency))
        
        # Time efficiency
        time_efficiency = 1.0 / ((fitness_values + 1e-10) * (times + 1e-10))
        features['time_efficiency'] = float(np.mean(time_efficiency))
        
        # Resource utilization
        max_evals = max(evaluations) if len(evaluations) > 0 else 1
        features['evaluation_utilization'] = float(np.mean(evaluations) / max_evals)
        
        return features
    
    def _extract_meta_features(self, results: List[OptimizationResult], 
                              algorithm_name: Optional[str] = None) -> Dict[str, float]:
        """Extract meta-features about algorithm characteristics."""
        features = {}
        
        # Algorithm type encoding
        if algorithm_name:
            algo_name = algorithm_name.lower()
            features['is_genetic'] = float('genetic' in algo_name or 'ga' in algo_name)
            features['is_swarm'] = float('swarm' in algo_name or 'pso' in algo_name)
            features['is_evolutionary'] = float('evolution' in algo_name or 'de' in algo_name)
            features['is_annealing'] = float('annealing' in algo_name or 'sa' in algo_name)
        
        # Run characteristics
        features['num_runs'] = float(len(results))
        
        # Convergence behavior classification
        fitness_values = [result.best_fitness for result in results]
        if len(fitness_values) > 1:
            # Classify convergence pattern
            sorted_fitness = sorted(fitness_values)
            best_quartile = sorted_fitness[:len(sorted_fitness)//4] if len(sorted_fitness) >= 4 else sorted_fitness[:1]
            worst_quartile = sorted_fitness[-len(sorted_fitness)//4:] if len(sorted_fitness) >= 4 else sorted_fitness[-1:]
            
            quartile_ratio = np.mean(worst_quartile) / (np.mean(best_quartile) + 1e-10)
            features['performance_spread'] = float(quartile_ratio)
        
        # Solution diversity (if solutions are available)
        if hasattr(results[0], 'best_solution') and results[0].best_solution is not None:
            solutions = [result.best_solution for result in results if result.best_solution is not None]
            if len(solutions) > 1:
                solution_distances = []
                for i in range(len(solutions)):
                    for j in range(i+1, len(solutions)):
                        dist = np.linalg.norm(np.array(solutions[i]) - np.array(solutions[j]))
                        solution_distances.append(dist)
                
                if solution_distances:
                    features['solution_diversity'] = float(np.mean(solution_distances))
                    features['solution_consistency'] = float(1.0 / (1.0 + np.std(solution_distances)))
        
        return features
    
    def extract_from_dataframe(self, df: pd.DataFrame, 
                             algorithm_column: str = 'algorithm_name',
                             group_columns: List[str] = None) -> pd.DataFrame:
        """
        Extract features from a pandas DataFrame of results.
        
        Args:
            df: DataFrame containing optimization results
            algorithm_column: Column name containing algorithm names
            group_columns: Additional columns to group by (e.g., problem_name, dimension)
            
        Returns:
            DataFrame with extracted features
        """
        if group_columns is None:
            group_columns = []
        
        # Group by algorithm and other specified columns
        group_cols = [algorithm_column] + group_columns
        feature_data = []
        
        for group_values, group_df in df.groupby(group_cols):
            if isinstance(group_values, str):
                group_values = [group_values]
            
            # Convert DataFrame rows to OptimizationResult objects
            results = []
            for _, row in group_df.iterrows():
                # Create a minimal OptimizationResult from available columns
                result = OptimizationResult(
                    best_fitness=row.get('best_fitness', float('inf')),
                    best_solution=row.get('best_solution', None),
                    total_evaluations=row.get('total_evaluations', 0),
                    execution_time=row.get('execution_time', 0.0),
                    success=row.get('success', False),
                    convergence_history=row.get('convergence_history', None)
                )
                results.append(result)
            
            # Extract features
            algorithm_name = group_values[0] if group_values else None
            features = self.extract_features(results, algorithm_name)
            
            # Add group information
            feature_dict = {group_cols[i]: group_values[i] for i in range(len(group_values))}
            feature_dict.update(features)
            
            feature_data.append(feature_dict)
        
        return pd.DataFrame(feature_data) 