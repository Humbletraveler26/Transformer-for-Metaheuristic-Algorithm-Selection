#!/usr/bin/env python3
"""
Massive Data Collection Script for Research Publication (Option A)
Target: 10,000+ optimization runs for comprehensive algorithm selection research.

This script implements Phase 1 of Path 1: Massive Dataset Expansion
- 50+ optimization problems from expanded benchmark suite
- Multiple dimensions (2, 5, 10, 20, 30) 
- 30+ runs per configuration for statistical significance
- Enhanced feature extraction and metadata collection
"""

import numpy as np
import pandas as pd
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from benchmarks.continuous_functions import (
    get_all_functions, get_function_by_name, 
    generate_problem_instances, get_function_characteristics
)

from metaheuristics.differential_evolution import DifferentialEvolution
from metaheuristics.particle_swarm import ParticleSwarmOptimization
from metaheuristics.genetic_algorithm import GeneticAlgorithm
from metaheuristics.simulated_annealing import SimulatedAnnealing

from features.problem_features import ProblemFeatureExtractor


class MassiveDataCollector:
    """
    Massive data collection system for research-grade algorithm selection dataset.
    
    Target: Generate 10,000+ optimization runs across diverse problem instances
    with comprehensive feature extraction and statistical robustness.
    """
    
    def __init__(self, 
                 target_runs: int = 30,
                 dimensions: List[int] = [2, 5, 10, 20, 30],
                 max_evaluations: int = 1000,
                 output_dir: str = "research_data",
                 parallel_workers: int = None):
        """
        Initialize the massive data collector.
        
        Args:
            target_runs: Number of runs per problem-algorithm combination
            dimensions: List of problem dimensions to test
            max_evaluations: Maximum function evaluations per run
            output_dir: Directory to store results
            parallel_workers: Number of parallel workers (None = auto)
        """
        self.target_runs = target_runs
        self.dimensions = dimensions
        self.max_evaluations = max_evaluations
        self.output_dir = output_dir
        self.parallel_workers = parallel_workers or mp.cpu_count()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.feature_extractor = ProblemFeatureExtractor()
        self.algorithms = self._initialize_algorithms()
        self.problem_instances = self._generate_problem_instances()
        
        print(f"🚀 Massive Data Collector Initialized")
        print(f"📊 Target runs per config: {target_runs}")
        print(f"📏 Dimensions: {dimensions}")
        print(f"🧮 Problem instances: {len(self.problem_instances)}")
        print(f"⚡ Algorithms: {len(self.algorithms)}")
        print(f"🔄 Parallel workers: {self.parallel_workers}")
        print(f"🎯 Estimated total runs: {len(self.problem_instances) * len(self.algorithms) * target_runs:,}")
    
    def _initialize_algorithms(self) -> Dict[str, Any]:
        """Initialize all metaheuristic algorithms with default parameters."""
        return {
            'differential_evolution': DifferentialEvolution,
            'particle_swarm': ParticleSwarmOptimization, 
            'genetic_algorithm': GeneticAlgorithm,
            'simulated_annealing': SimulatedAnnealing
        }
    
    def _generate_problem_instances(self) -> List[Any]:
        """Generate comprehensive problem instance suite."""
        print("🏗️  Generating comprehensive problem suite...")
        
        # Get all available functions
        all_functions = list(get_all_functions().keys())
        print(f"📋 Available functions: {len(all_functions)}")
        print(f"📝 Functions: {', '.join(all_functions)}")
        
        # Generate instances across all dimensions
        instances = []
        for dim in self.dimensions:
            for func_name in all_functions:
                try:
                    func = get_function_by_name(func_name, dim)
                    instances.append(func)
                except Exception as e:
                    print(f"⚠️  Skipping {func_name} dim={dim}: {e}")
        
        print(f"✅ Generated {len(instances)} problem instances")
        
        # Show distribution
        dim_counts = {}
        func_counts = {}
        for instance in instances:
            dim = instance.dimension
            func_name = instance.name
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
            func_counts[func_name] = func_counts.get(func_name, 0) + 1
        
        print("📊 Distribution by dimension:")
        for dim, count in sorted(dim_counts.items()):
            print(f"  Dimension {dim}: {count} problems")
        
        return instances
    
    def _extract_enhanced_features(self, problem_instance: Any) -> Dict[str, Any]:
        """Extract comprehensive features from problem instance."""
        try:
            # Basic metadata
            features = {
                'problem_name': problem_instance.name,
                'dimension': problem_instance.dimension,
                'bounds_lower': problem_instance.bounds[0],
                'bounds_upper': problem_instance.bounds[1],
                'bounds_range': problem_instance.bounds[1] - problem_instance.bounds[0],
                'global_optimum': problem_instance.global_optimum,
                'separable': problem_instance.separable,
            }
            
            # Sample points for landscape analysis
            np.random.seed(42)  # Reproducible sampling
            n_samples = min(100, 50 * problem_instance.dimension)  # Scale with dimension
            
            sample_points = []
            for _ in range(n_samples):
                x = np.random.uniform(
                    problem_instance.bounds[0], 
                    problem_instance.bounds[1], 
                    problem_instance.dimension
                )
                sample_points.append(x)
            
            # Evaluate samples
            sample_values = [problem_instance(x) for x in sample_points]
            
            # Statistical features
            features.update({
                'landscape_mean': np.mean(sample_values),
                'landscape_std': np.std(sample_values),
                'landscape_min': np.min(sample_values),
                'landscape_max': np.max(sample_values),
                'landscape_range': np.max(sample_values) - np.min(sample_values),
                'landscape_skewness': self._calculate_skewness(sample_values),
                'landscape_kurtosis': self._calculate_kurtosis(sample_values),
            })
            
            # Gradient-based features (finite differences)
            if problem_instance.dimension <= 20:  # Only for reasonable dimensions
                gradients = []
                for x in sample_points[:20]:  # Sample subset for efficiency
                    grad = self._estimate_gradient(problem_instance, x)
                    if grad is not None:
                        gradients.append(np.linalg.norm(grad))
                
                if gradients:
                    features.update({
                        'gradient_norm_mean': np.mean(gradients),
                        'gradient_norm_std': np.std(gradients),
                        'gradient_norm_max': np.max(gradients),
                    })
            
            # Multimodality indicators
            local_minima_count = self._estimate_local_minima(sample_values)
            features['estimated_local_minima'] = local_minima_count
            features['multimodality_indicator'] = local_minima_count > 1
            
            # Problem characteristics from registry
            try:
                from benchmarks.continuous_functions import _get_function_properties
                properties = _get_function_properties(problem_instance.name)
                features['problem_properties'] = properties
                
                # One-hot encode common properties
                common_properties = [
                    'unimodal', 'multimodal', 'separable', 'non_separable',
                    'convex', 'smooth', 'deceptive', 'ill_conditioned'
                ]
                for prop in common_properties:
                    features[f'property_{prop}'] = prop in properties or prop.replace('_', ' ') in properties
                    
            except Exception as e:
                print(f"⚠️  Could not extract properties for {problem_instance.name}: {e}")
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features for {problem_instance.name}: {e}")
            return {'error': str(e)}
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of values."""
        try:
            values = np.array(values)
            mean = np.mean(values)
            std = np.std(values)
            if std == 0:
                return 0.0
            return np.mean(((values - mean) / std) ** 3)
        except:
            return 0.0
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate kurtosis of values."""
        try:
            values = np.array(values)
            mean = np.mean(values)
            std = np.std(values)
            if std == 0:
                return 0.0
            return np.mean(((values - mean) / std) ** 4) - 3
        except:
            return 0.0
    
    def _estimate_gradient(self, func, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """Estimate gradient using finite differences."""
        try:
            grad = np.zeros_like(x)
            f_x = func(x)
            
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += h
                f_plus = func(x_plus)
                grad[i] = (f_plus - f_x) / h
            
            return grad
        except:
            return None
    
    def _estimate_local_minima(self, values: List[float], window: int = 5) -> int:
        """Estimate number of local minima in the sample."""
        try:
            values = np.array(values)
            if len(values) < window:
                return 1
            
            # Sort values and look for valleys
            sorted_idx = np.argsort(values)
            local_minima = 0
            
            # Simple heuristic: count significant valleys
            for i in range(window, len(values) - window):
                left_higher = np.all(values[sorted_idx[i-window:i]] > values[sorted_idx[i]])
                right_higher = np.all(values[sorted_idx[i+1:i+window+1]] > values[sorted_idx[i]])
                if left_higher and right_higher:
                    local_minima += 1
            
            return max(1, local_minima)  # At least one minimum
        except:
            return 1
    
    def run_single_optimization(self, 
                              problem_instance: Any, 
                              algorithm_name: str, 
                              run_id: int) -> Dict[str, Any]:
        """Run a single optimization and collect comprehensive data."""
        try:
            # Reset function evaluation counter
            problem_instance.reset_counter()
            
            # Initialize algorithm with correct parameters
            algorithm_class = self.algorithms[algorithm_name]
            
            # Create algorithm instance with proper configuration parameters
            if algorithm_name == 'differential_evolution':
                algorithm = algorithm_class(
                    population_size=30,
                    F=0.8,
                    CR=0.9
                )
            elif algorithm_name == 'particle_swarm':
                algorithm = algorithm_class(
                    swarm_size=30,
                    inertia_weight=0.9,
                    cognitive_coeff=2.0,
                    social_coeff=2.0
                )
            elif algorithm_name == 'genetic_algorithm':
                algorithm = algorithm_class(
                    population_size=30,
                    crossover_rate=0.8,
                    mutation_rate=0.1
                )
            elif algorithm_name == 'simulated_annealing':
                algorithm = algorithm_class(
                    initial_temp=100.0,
                    cooling_rate=0.95
                )
            else:
                # Fallback generic initialization
                algorithm = algorithm_class()
            
            # Record start time
            start_time = time.time()
            
            # Run optimization with proper method call
            result = algorithm.optimize(
                objective_function=problem_instance,
                dimension=problem_instance.dimension,
                bounds=problem_instance.bounds,
                max_evaluations=self.max_evaluations,
                target_fitness=1e-8
            )
            
            # Record end time
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Extract results from OptimizationResult
            best_solution = result.best_solution
            best_fitness = result.best_fitness
            convergence_history = result.convergence_history
            actual_evaluations = result.total_evaluations
            
            # Calculate performance metrics
            
            # Distance to global optimum (if at origin)
            if hasattr(problem_instance, 'global_optimum'):
                distance_to_global = abs(best_fitness - problem_instance.global_optimum)
            else:
                distance_to_global = None
            
            # Convergence analysis
            if convergence_history:
                initial_fitness = convergence_history[0] if convergence_history else best_fitness
                improvement = initial_fitness - best_fitness
                improvement_rate = improvement / len(convergence_history) if convergence_history else 0
                
                # Convergence speed (evaluations to reach 90% of final improvement)
                target_fitness = initial_fitness - 0.9 * improvement
                convergence_speed = None
                for i, fitness in enumerate(convergence_history):
                    if fitness <= target_fitness:
                        convergence_speed = i + 1
                        break
                if convergence_speed is None:
                    convergence_speed = len(convergence_history)
            else:
                improvement = 0
                improvement_rate = 0
                convergence_speed = actual_evaluations
            
            # Compile results
            result_dict = {
                # Problem identification
                'problem_name': problem_instance.name,
                'algorithm_name': algorithm_name,
                'run_id': run_id,
                
                # Performance metrics
                'best_fitness': float(best_fitness),
                'final_fitness': float(best_fitness),  # Same as best for final
                'distance_to_global_optimum': distance_to_global,
                'execution_time_seconds': execution_time,
                'actual_evaluations': actual_evaluations,
                'evaluations_per_second': actual_evaluations / execution_time if execution_time > 0 else 0,
                
                # Convergence metrics
                'improvement': improvement,
                'improvement_rate': improvement_rate,
                'convergence_speed': convergence_speed,
                'convergence_rate': convergence_speed / actual_evaluations if actual_evaluations > 0 else 0,
                'convergence_efficiency': convergence_speed / actual_evaluations if actual_evaluations > 0 else 0,
                
                # Solution quality
                'solution_found': best_solution.tolist() if best_solution is not None else None,
                'solution_norm': float(np.linalg.norm(best_solution)) if best_solution is not None else None,
                
                # Success indicators  
                'success': result.success,
                'success_rate': 1.0 if result.success else 0.0,
                'budget_sufficient': actual_evaluations < self.max_evaluations,
                
                # Metadata
                'timestamp': datetime.now().isoformat(),
                'max_evaluations_budget': self.max_evaluations,
                'budget_used_fraction': actual_evaluations / self.max_evaluations,
                
                # No error if we reach here
                'error': None
            }
            
            return result_dict
            
        except Exception as e:
            # Return error data
            return {
                'problem_name': problem_instance.name if hasattr(problem_instance, 'name') else 'unknown',
                'algorithm_name': algorithm_name,
                'run_id': run_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'best_fitness': None,
                'final_fitness': None,
                'convergence_rate': None,
                'success_rate': 0.0
            }
    
    def collect_batch_data(self, 
                          batch_problems: List[Any],
                          save_interval: int = 100) -> pd.DataFrame:
        """Collect data for a batch of problems with all algorithms."""
        all_results = []
        completed_runs = 0
        
        total_runs = len(batch_problems) * len(self.algorithms) * self.target_runs
        print(f"🎯 Starting batch collection: {total_runs:,} total runs")
        
        # Extract features for all problems first (cached)
        print("🔍 Extracting problem features...")
        problem_features = {}
        for problem in batch_problems:
            key = f"{problem.name}_dim{problem.dimension}"
            problem_features[key] = self._extract_enhanced_features(problem)
        
        # Run optimizations
        for problem_idx, problem in enumerate(batch_problems):
            print(f"\n📊 Problem {problem_idx + 1}/{len(batch_problems)}: {problem.name} (dim={problem.dimension})")
            
            problem_key = f"{problem.name}_dim{problem.dimension}"
            features = problem_features[problem_key]
            
            for algorithm_name in self.algorithms:
                print(f"  🔧 Algorithm: {algorithm_name}")
                
                for run_id in range(self.target_runs):
                    # Run optimization
                    result = self.run_single_optimization(problem, algorithm_name, run_id)
                    
                    # Add problem features to result
                    result.update(features)
                    
                    # Label the best algorithm (placeholder - would be determined by analysis)
                    result['is_best_algorithm'] = False  # Will be determined later
                    
                    all_results.append(result)
                    completed_runs += 1
                    
                    # Progress update
                    if completed_runs % 10 == 0:
                        progress = completed_runs / total_runs * 100
                        print(f"    📈 Progress: {completed_runs:,}/{total_runs:,} ({progress:.1f}%)")
                    
                    # Save intermediate results
                    if completed_runs % save_interval == 0:
                        temp_df = pd.DataFrame(all_results)
                        temp_filename = os.path.join(self.output_dir, f"temp_results_{completed_runs}.csv")
                        temp_df.to_csv(temp_filename, index=False)
                        print(f"    💾 Saved intermediate results: {temp_filename}")
        
        print(f"\n✅ Batch collection completed: {completed_runs:,} runs")
        return pd.DataFrame(all_results)
    
    def determine_best_algorithms(self, df):
        """
        Determine the best algorithm for each problem based on performance.
        Uses best fitness values when available, falls back to lowest error count.
        """
        print("🏆 Determining best algorithms for each problem...")
        
        # Group by problem and dimension
        problem_groups = df.groupby(['problem_name', 'dimension'])
        
        best_algorithms = []
        
        for (problem_name, dimension), group in problem_groups:
            # Check if we have any successful runs (no errors)
            successful_runs = group[group['error'].isna() | (group['error'] == '')]
            
            if len(successful_runs) > 0:
                # Use performance metrics for successful runs
                algo_performance = successful_runs.groupby('algorithm_name').agg({
                    'best_fitness': 'mean',
                    'final_fitness': 'mean',
                    'convergence_rate': 'mean',
                    'success_rate': 'mean'
                }).reset_index()
                
                # Best algorithm is the one with lowest average best_fitness
                best_algo = algo_performance.loc[algo_performance['best_fitness'].idxmin(), 'algorithm_name']
            else:
                # Fallback: algorithm with fewest errors
                algo_errors = group.groupby('algorithm_name')['error'].count()
                best_algo = algo_errors.idxmin()
            
            best_algorithms.append({
                'problem_name': problem_name,
                'dimension': dimension,
                'best_algorithm': best_algo
            })
        
        # Mark best algorithms in the dataframe
        df['is_best_algorithm'] = False
        
        for best_info in best_algorithms:
            mask = (
                (df['problem_name'] == best_info['problem_name']) & 
                (df['dimension'] == best_info['dimension']) & 
                (df['algorithm_name'] == best_info['best_algorithm'])
            )
            df.loc[mask, 'is_best_algorithm'] = True
        
        return df
    
    def save_research_dataset(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save the research dataset with comprehensive metadata."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_dataset_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Generate comprehensive metadata
        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_runs': len(df),
            'unique_problems': df.groupby(['problem_name', 'dimension']).ngroups,
            'unique_algorithms': df['algorithm_name'].nunique(),
            'algorithms_tested': sorted(df['algorithm_name'].unique()),
            'problems_tested': sorted(df['problem_name'].unique()),
            'dimensions_tested': sorted(df['dimension'].unique()),
            'runs_per_algorithm': df['algorithm_name'].value_counts().to_dict(),
            'success_rate_overall': df['success_rate'].mean() if 'success_rate' in df.columns else 0.0,
            'avg_best_fitness': df['best_fitness'].mean() if 'best_fitness' in df.columns else None,
            'collection_parameters': {
                'target_runs': self.target_runs,
                'max_evaluations': self.max_evaluations,
                'dimensions': self.dimensions,
                'parallel_workers': self.parallel_workers
            },
            'data_quality': {
                'error_rate': (df['error'].notna() & (df['error'] != '')).mean() if 'error' in df.columns else 0.0,
                'completion_rate': (df['best_fitness'].notna()).mean() if 'best_fitness' in df.columns else 0.0,
                'total_features': len(df.columns)
            }
        }
        
        # Save main dataset
        df.to_csv(filepath, index=False)
        
        # Save metadata
        metadata_path = filepath.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"📁 Dataset saved: {filepath}")
        print(f"📋 Metadata saved: {metadata_path}")
        print(f"📊 Dataset summary:")
        print(f"   • Total runs: {metadata['total_runs']:,}")
        print(f"   • Problems: {metadata['unique_problems']}")
        print(f"   • Algorithms: {metadata['unique_algorithms']}")
        print(f"   • Success rate: {metadata['success_rate_overall']:.2%}")
        print(f"   • Error rate: {metadata['data_quality']['error_rate']:.2%}")
        
        return filepath
    
    def run_massive_collection(self, batch_size: int = None) -> str:
        """Run the complete massive data collection process."""
        print("🚀 STARTING MASSIVE DATA COLLECTION FOR RESEARCH")
        print("=" * 60)
        
        start_time = time.time()
        
        # Determine batch size
        if batch_size is None:
            batch_size = min(20, len(self.problem_instances))  # Reasonable default
        
        print(f"📋 Collection Configuration:")
        print(f"  Total problems: {len(self.problem_instances)}")
        print(f"  Algorithms: {len(self.algorithms)}")
        print(f"  Runs per config: {self.target_runs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Estimated total runs: {len(self.problem_instances) * len(self.algorithms) * self.target_runs:,}")
        
        # Process in batches
        all_dataframes = []
        num_batches = (len(self.problem_instances) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.problem_instances))
            batch_problems = self.problem_instances[start_idx:end_idx]
            
            print(f"\n🔄 Processing Batch {batch_idx + 1}/{num_batches}")
            print(f"   Problems {start_idx + 1}-{end_idx} ({len(batch_problems)} problems)")
            
            # Collect batch data
            batch_df = self.collect_batch_data(batch_problems)
            all_dataframes.append(batch_df)
            
            # Save batch results
            batch_filename = f"batch_{batch_idx + 1}_results.csv"
            batch_filepath = os.path.join(self.output_dir, batch_filename)
            batch_df.to_csv(batch_filepath, index=False)
            print(f"💾 Saved batch results: {batch_filename}")
        
        # Combine all results
        print(f"\n🔗 Combining all batches...")
        final_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Determine best algorithms
        final_df = self.determine_best_algorithms(final_df)
        
        # Save final dataset
        final_filepath = self.save_research_dataset(final_df)
        
        # Final summary
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 MASSIVE DATA COLLECTION COMPLETED!")
        print(f"=" * 50)
        print(f"📊 Total runs: {len(final_df):,}")
        print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/3600:.1f} hours)")
        print(f"🚀 Runs per second: {len(final_df)/total_time:.2f}")
        print(f"💾 Dataset saved: {final_filepath}")
        print(f"🎯 Ready for transformer training and research publication!")
        
        return final_filepath


def main():
    """Main function to run massive data collection."""
    print("🧬 MASSIVE RESEARCH DATA COLLECTION")
    print("Option A: Scale for Research Publication")
    print("=" * 50)
    
    # Configuration for research-grade dataset
    collector = MassiveDataCollector(
        target_runs=30,  # Statistical significance
        dimensions=[2, 5, 10, 20, 30],  # Multi-scale problems
        max_evaluations=1000,  # Reasonable budget
        output_dir="research_data_massive",
        parallel_workers=4  # Conservative for stability
    )
    
    # Run collection
    dataset_path = collector.run_massive_collection(batch_size=10)
    
    print(f"\n✅ Research dataset ready: {dataset_path}")
    print("🎯 Next steps:")
    print("  1. Train advanced transformer models")
    print("  2. Implement ensemble methods")
    print("  3. Conduct statistical analysis")
    print("  4. Prepare research paper")


if __name__ == "__main__":
    main() 