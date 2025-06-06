"""
Performance data collection system for metaheuristic algorithms.

This module provides tools to systematically collect performance data
across different optimization problems and metaheuristic algorithms.
"""

import os
import time
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from ..benchmarks import generate_problem_instances, get_function_by_name
from ..metaheuristics import (
    GeneticAlgorithm, 
    ParticleSwarmOptimization,
    DifferentialEvolution,
    SimulatedAnnealing,
    OptimizationResult
)


class ExperimentRunner:
    """Handles individual experiment execution."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.algorithms = self._initialize_algorithms()
    
    def _initialize_algorithms(self) -> Dict[str, Any]:
        """Initialize metaheuristic algorithms based on config."""
        algorithms = {}
        
        metaheuristic_config = self.config.get('metaheuristics', {})
        
        # Genetic Algorithm
        if metaheuristic_config.get('genetic_algorithm', {}).get('enabled', False):
            ga_params = metaheuristic_config['genetic_algorithm']['default_params']
            algorithms['genetic_algorithm'] = GeneticAlgorithm(**ga_params)
        
        # Particle Swarm Optimization
        if metaheuristic_config.get('particle_swarm', {}).get('enabled', False):
            pso_params = metaheuristic_config['particle_swarm']['default_params']
            algorithms['particle_swarm'] = ParticleSwarmOptimization(**pso_params)
        
        # Differential Evolution
        if metaheuristic_config.get('differential_evolution', {}).get('enabled', False):
            de_params = metaheuristic_config['differential_evolution']['default_params']
            algorithms['differential_evolution'] = DifferentialEvolution(**de_params)
        
        # Simulated Annealing
        if metaheuristic_config.get('simulated_annealing', {}).get('enabled', False):
            sa_params = metaheuristic_config['simulated_annealing']['default_params']
            algorithms['simulated_annealing'] = SimulatedAnnealing(**sa_params)
        
        return algorithms
    
    def run_single_experiment(self, 
                            problem_name: str,
                            dimension: int,
                            algorithm_name: str,
                            run_id: int,
                            random_seed: int) -> Dict[str, Any]:
        """Run a single optimization experiment."""
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Get problem instance
        problem = get_function_by_name(problem_name, dimension)
        problem.reset_counter()
        
        # Get algorithm
        algorithm = self.algorithms[algorithm_name]
        algorithm.random_state = random_seed
        
        # Run optimization
        experiment_config = self.config['experiment']
        
        try:
            result = algorithm.optimize(
                objective_function=problem,
                dimension=dimension,
                bounds=problem.bounds,
                max_evaluations=experiment_config['max_evaluations'],
                target_fitness=self.config['metrics']['success_threshold']
            )
            
            # Convert result to dictionary for storage
            result_dict = asdict(result)
            
            # Add experiment metadata
            result_dict.update({
                'problem_name': problem_name,
                'dimension': dimension,
                'algorithm_name': algorithm_name,
                'run_id': run_id,
                'random_seed': random_seed,
                'timestamp': time.time(),
                'problem_metadata': problem.get_metadata()
            })
            
            return result_dict
            
        except Exception as e:
            # Return error information
            return {
                'problem_name': problem_name,
                'dimension': dimension,
                'algorithm_name': algorithm_name,
                'run_id': run_id,
                'random_seed': random_seed,
                'error': str(e),
                'timestamp': time.time()
            }


class PerformanceCollector:
    """Main class for collecting performance data across experiments."""
    
    def __init__(self, config_path: str = "configs/project_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.experiment_runner = ExperimentRunner(self.config)
        
        # Create data directories
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load project configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_directories(self):
        """Create necessary data directories."""
        data_paths = [
            "data/raw",
            "data/processed", 
            "data/synthetic"
        ]
        
        for path in data_paths:
            os.makedirs(path, exist_ok=True)
    
    def generate_experiment_plan(self) -> List[Dict[str, Any]]:
        """Generate a list of all experiments to run."""
        experiments = []
        
        # Get enabled problem types and algorithms
        problem_config = self.config['problem_types']['continuous']
        metaheuristic_config = self.config['metaheuristics']
        experiment_config = self.config['experiment']
        
        if not problem_config['enabled']:
            return experiments
        
        # Get enabled algorithms
        enabled_algorithms = [
            name for name, config in metaheuristic_config.items()
            if config.get('enabled', False)
        ]
        
        # Generate experiments
        for function_name in problem_config['functions']:
            if function_name == 'cec2017':  # Skip CEC for now
                continue
                
            for dimension in problem_config['dimensions']:
                for algorithm_name in enabled_algorithms:
                    for run_id in range(experiment_config['runs_per_combination']):
                        for seed_idx, random_seed in enumerate(experiment_config['random_seeds']):
                            experiments.append({
                                'problem_name': function_name,
                                'dimension': dimension,
                                'algorithm_name': algorithm_name,
                                'run_id': run_id,
                                'random_seed': random_seed
                            })
        
        return experiments
    
    def run_experiments(self, 
                       max_workers: int = 4,
                       save_interval: int = 100) -> pd.DataFrame:
        """Run all experiments and collect results."""
        
        experiments = self.generate_experiment_plan()
        print(f"Generated {len(experiments)} experiments to run")
        
        results = []
        
        # Run experiments with progress bar
        with tqdm(total=len(experiments), desc="Running experiments") as pbar:
            for i, experiment in enumerate(experiments):
                try:
                    result = self.experiment_runner.run_single_experiment(**experiment)
                    results.append(result)
                    
                    # Save intermediate results
                    if (i + 1) % save_interval == 0:
                        self._save_results(results, f"intermediate_{i+1}")
                    
                except Exception as e:
                    print(f"Error in experiment {i}: {e}")
                    results.append({
                        **experiment,
                        'error': str(e),
                        'timestamp': time.time()
                    })
                
                pbar.update(1)
        
        # Convert to DataFrame and save final results
        df_results = pd.DataFrame(results)
        self._save_results(results, "final")
        
        return df_results
    
    def run_experiments_parallel(self, 
                                max_workers: int = 4,
                                save_interval: int = 100) -> pd.DataFrame:
        """Run experiments in parallel (use with caution - memory intensive)."""
        
        experiments = self.generate_experiment_plan()
        print(f"Generated {len(experiments)} experiments to run")
        
        results = []
        
        # Run experiments in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_experiment = {
                executor.submit(self.experiment_runner.run_single_experiment, **exp): exp
                for exp in experiments
            }
            
            # Collect results with progress bar
            with tqdm(total=len(experiments), desc="Running experiments") as pbar:
                for future in as_completed(future_to_experiment):
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Save intermediate results
                        if len(results) % save_interval == 0:
                            self._save_results(results, f"intermediate_{len(results)}")
                        
                    except Exception as e:
                        experiment = future_to_experiment[future]
                        print(f"Error in experiment {experiment}: {e}")
                        results.append({
                            **experiment,
                            'error': str(e),
                            'timestamp': time.time()
                        })
                    
                    pbar.update(1)
        
        # Convert to DataFrame and save final results
        df_results = pd.DataFrame(results)
        self._save_results(results, "final")
        
        return df_results
    
    def _save_results(self, results: List[Dict], suffix: str = ""):
        """Save results to CSV file."""
        df = pd.DataFrame(results)
        
        timestamp = int(time.time())
        filename = f"performance_results_{suffix}_{timestamp}.csv"
        filepath = os.path.join("data/raw", filename)
        
        df.to_csv(filepath, index=False)
        print(f"Saved {len(results)} results to {filepath}")
    
    def load_results(self, filepath: str) -> pd.DataFrame:
        """Load previously collected results."""
        return pd.read_csv(filepath)
    
    def get_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics from results."""
        
        # Filter out error results
        df_clean = df[df['error'].isna()] if 'error' in df.columns else df
        
        # Group by problem and algorithm
        summary = df_clean.groupby(['problem_name', 'dimension', 'algorithm_name']).agg({
            'best_fitness': ['mean', 'std', 'min', 'max'],
            'total_evaluations': ['mean', 'std'],
            'execution_time': ['mean', 'std'],
            'success': ['mean', 'sum', 'count']
        }).round(6)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        
        return summary.reset_index()


def main():
    """Main function to run data collection."""
    collector = PerformanceCollector()
    
    print("Starting performance data collection...")
    results_df = collector.run_experiments(max_workers=2)
    
    print(f"\nCollected {len(results_df)} experiment results")
    
    # Generate summary statistics
    summary = collector.get_summary_statistics(results_df)
    print("\nSummary Statistics:")
    print(summary)


if __name__ == "__main__":
    main() 