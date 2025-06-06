#!/usr/bin/env python3
"""
Initial data collection script for the Transformer Metaheuristic Selection project.

This script collects a small dataset to get started with the project.
It runs the Genetic Algorithm on a subset of benchmark problems.
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from dataclasses import asdict

# Add src to path
sys.path.append('src')

from benchmarks import get_function_by_name
from metaheuristics import GeneticAlgorithm


def collect_initial_data():
    """Collect initial performance data."""
    
    print("Starting initial data collection...")
    print("=" * 50)
    
    # Configuration for initial data collection
    problems = [
        ('sphere', 2), ('sphere', 5), ('sphere', 10),
        ('rastrigin', 2), ('rastrigin', 5), ('rastrigin', 10),
        ('rosenbrock', 2), ('rosenbrock', 5), ('rosenbrock', 10),
        ('ackley', 2), ('ackley', 5), ('ackley', 10)
    ]
    
    algorithms = {
        'genetic_algorithm': GeneticAlgorithm(
            population_size=30,
            crossover_rate=0.8,
            mutation_rate=0.1,
            tournament_size=3
        )
    }
    
    # Experiment parameters
    runs_per_combination = 5
    max_evaluations = 1000
    target_fitness = 1e-6
    random_seeds = [42, 123, 456, 789, 101112]
    
    results = []
    total_experiments = len(problems) * len(algorithms) * runs_per_combination
    experiment_count = 0
    
    print(f"Total experiments to run: {total_experiments}")
    print()
    
    # Run experiments
    for problem_name, dimension in problems:
        for algorithm_name, algorithm in algorithms.items():
            print(f"Running {algorithm_name} on {problem_name}({dimension}D)...")
            
            for run_id in range(runs_per_combination):
                experiment_count += 1
                random_seed = random_seeds[run_id]
                
                # Set random seed
                np.random.seed(random_seed)
                algorithm.random_state = random_seed
                
                # Get problem instance
                problem = get_function_by_name(problem_name, dimension)
                problem.reset_counter()
                
                # Run optimization
                start_time = time.time()
                
                try:
                    result = algorithm.optimize(
                        objective_function=problem,
                        dimension=dimension,
                        bounds=problem.bounds,
                        max_evaluations=max_evaluations,
                        target_fitness=target_fitness
                    )
                    
                    # Convert result to dictionary
                    result_dict = asdict(result)
                    
                    # Add experiment metadata
                    result_dict.update({
                        'experiment_id': experiment_count,
                        'problem_name': problem_name,
                        'dimension': dimension,
                        'algorithm_name': algorithm_name,
                        'run_id': run_id,
                        'random_seed': random_seed,
                        'timestamp': time.time(),
                        'problem_bounds': problem.bounds,
                        'problem_separable': problem.separable,
                        'problem_global_optimum': problem.global_optimum
                    })
                    
                    results.append(result_dict)
                    
                    print(f"  Run {run_id+1}/{runs_per_combination}: "
                          f"fitness={result.best_fitness:.6f}, "
                          f"evals={result.total_evaluations}, "
                          f"time={result.execution_time:.3f}s, "
                          f"success={result.success}")
                    
                except Exception as e:
                    print(f"  Run {run_id+1}/{runs_per_combination}: ERROR - {e}")
                    results.append({
                        'experiment_id': experiment_count,
                        'problem_name': problem_name,
                        'dimension': dimension,
                        'algorithm_name': algorithm_name,
                        'run_id': run_id,
                        'random_seed': random_seed,
                        'error': str(e),
                        'timestamp': time.time()
                    })
            
            print()
    
    # Save results
    print("Saving results...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    timestamp = int(time.time())
    filename = f"data/raw/initial_performance_data_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    print(f"Saved {len(results)} results to {filename}")
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    
    # Filter successful runs (no errors)
    df_clean = df[~df.get('error', pd.Series([None]*len(df))).notna()]
    
    if len(df_clean) > 0:
        summary = df_clean.groupby(['problem_name', 'dimension', 'algorithm_name']).agg({
            'best_fitness': ['mean', 'std', 'min', 'max'],
            'total_evaluations': ['mean', 'std'],
            'execution_time': ['mean', 'std'],
            'success': ['mean', 'sum', 'count']
        }).round(6)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        summary = summary.reset_index()
        
        # Save summary
        summary_filename = f"data/raw/initial_summary_{timestamp}.csv"
        summary.to_csv(summary_filename, index=False)
        
        print(f"Summary statistics:")
        print(summary.to_string())
        print(f"\nSaved summary to {summary_filename}")
    
    print("\n" + "=" * 50)
    print("Initial data collection completed!")
    print(f"Total experiments: {len(results)}")
    print(f"Successful experiments: {len(df_clean)}")
    print(f"Error rate: {(len(results) - len(df_clean)) / len(results) * 100:.1f}%")
    
    return df


def analyze_results(df):
    """Perform basic analysis of the collected results."""
    print("\nBasic Analysis:")
    print("=" * 30)
    
    # Filter clean results
    df_clean = df[~df.get('error', pd.Series([None]*len(df))).notna()]
    
    if len(df_clean) == 0:
        print("No successful experiments to analyze.")
        return
    
    # Success rates by problem
    print("Success rates by problem:")
    success_by_problem = df_clean.groupby(['problem_name', 'dimension'])['success'].agg(['mean', 'count'])
    print(success_by_problem)
    
    # Best fitness by problem
    print("\nBest fitness achieved by problem:")
    best_fitness = df_clean.groupby(['problem_name', 'dimension'])['best_fitness'].agg(['mean', 'min', 'std'])
    print(best_fitness)
    
    # Performance correlation with dimension
    print("\nCorrelation between dimension and performance:")
    correlation = df_clean[['dimension', 'best_fitness', 'total_evaluations', 'execution_time']].corr()
    print(correlation)


def main():
    """Main function."""
    print("TRANSFORMER METAHEURISTIC SELECTION - INITIAL DATA COLLECTION")
    print("=" * 70)
    print()
    
    # Collect data
    df = collect_initial_data()
    
    # Analyze results
    analyze_results(df)
    
    print("\nNext steps:")
    print("1. Implement additional metaheuristics (PSO, DE, SA)")
    print("2. Expand to more benchmark problems and dimensions")
    print("3. Implement feature extraction for problems")
    print("4. Begin developing the Transformer model")
    
    return 0


if __name__ == "__main__":
    exit(main()) 