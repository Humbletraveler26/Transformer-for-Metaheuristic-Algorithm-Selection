#!/usr/bin/env python3
"""
Comprehensive data collection script for all metaheuristic algorithms.

This script collects performance data for GA, PSO, DE, and SA across
multiple benchmark problems and dimensions to build a substantial dataset.
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
from metaheuristics import (
    GeneticAlgorithm,
    ParticleSwarmOptimization, 
    DifferentialEvolution,
    SimulatedAnnealing
)


def collect_comprehensive_data():
    """Collect comprehensive performance data for all algorithms."""
    
    print("COMPREHENSIVE DATA COLLECTION FOR ALL METAHEURISTICS")
    print("=" * 60)
    
    # Extended configuration for comprehensive data collection
    problems = [
        # 2D problems (quick testing)
        ('sphere', 2), ('rastrigin', 2), ('rosenbrock', 2), ('ackley', 2), ('griewank', 2),
        # 5D problems (medium complexity)
        ('sphere', 5), ('rastrigin', 5), ('rosenbrock', 5), ('ackley', 5), ('griewank', 5),
        # 10D problems (higher complexity)
        ('sphere', 10), ('rastrigin', 10), ('rosenbrock', 10), ('ackley', 10), ('griewank', 10),
    ]
    
    # All algorithms with optimized parameters
    algorithms = {
        'genetic_algorithm': GeneticAlgorithm(
            population_size=30,
            crossover_rate=0.8,
            mutation_rate=0.1,
            tournament_size=3,
            elitism=True
        ),
        'particle_swarm': ParticleSwarmOptimization(
            swarm_size=30,
            inertia_weight=0.9,
            inertia_decay=0.99,
            cognitive_coeff=2.0,
            social_coeff=2.0,
            velocity_clamp=True,
            max_velocity_factor=0.2
        ),
        'differential_evolution': DifferentialEvolution(
            population_size=30,
            F=0.5,
            CR=0.7,
            strategy="rand/1/bin",
            adaptive=False
        ),
        'simulated_annealing': SimulatedAnnealing(
            initial_temp=100.0,
            final_temp=0.01,
            cooling_schedule="exponential",
            cooling_rate=0.95,
            steps_per_temp=10,
            step_size=0.1,
            adaptive_step=True
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
    
    print(f"Problems: {len(problems)} (5 functions × 3 dimensions)")
    print(f"Algorithms: {len(algorithms)} (GA, PSO, DE, SA)")
    print(f"Runs per combination: {runs_per_combination}")
    print(f"Total experiments: {total_experiments}")
    print(f"Max evaluations per run: {max_evaluations}")
    print()
    
    start_time = time.time()
    
    # Run experiments
    for problem_name, dimension in problems:
        print(f"Problem: {problem_name}({dimension}D)")
        print("-" * 40)
        
        for algorithm_name, algorithm in algorithms.items():
            print(f"  Running {algorithm_name}...")
            
            problem_results = []
            
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
                    problem_results.append(result.best_fitness)
                    
                except Exception as e:
                    print(f"    Run {run_id+1}: ERROR - {e}")
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
            
            # Summary for this algorithm on this problem
            if problem_results:
                mean_fitness = np.mean(problem_results)
                std_fitness = np.std(problem_results)
                min_fitness = np.min(problem_results)
                print(f"    Results: mean={mean_fitness:.6f}, std={std_fitness:.6f}, min={min_fitness:.6f}")
        
        print()
    
    # Save results
    print("Saving comprehensive results...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    timestamp = int(time.time())
    filename = f"data/raw/comprehensive_performance_data_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    print(f"Saved {len(results)} results to {filename}")
    
    # Generate comprehensive summary statistics
    print("\nGenerating comprehensive summary statistics...")
    
    # Filter successful runs (no errors)
    df_clean = df[~df.get('error', pd.Series([None]*len(df))).notna()]
    
    if len(df_clean) > 0:
        # Overall summary by algorithm
        algo_summary = df_clean.groupby('algorithm_name').agg({
            'best_fitness': ['mean', 'std', 'min', 'max'],
            'total_evaluations': ['mean', 'std'],
            'execution_time': ['mean', 'std'],
            'success': ['mean', 'sum', 'count']
        }).round(6)
        
        # Summary by problem and algorithm
        detailed_summary = df_clean.groupby(['problem_name', 'dimension', 'algorithm_name']).agg({
            'best_fitness': ['mean', 'std', 'min', 'max'],
            'total_evaluations': ['mean', 'std'],
            'execution_time': ['mean', 'std'],
            'success': ['mean', 'sum', 'count']
        }).round(6)
        
        # Flatten column names
        algo_summary.columns = ['_'.join(col).strip() for col in algo_summary.columns]
        detailed_summary.columns = ['_'.join(col).strip() for col in detailed_summary.columns]
        
        algo_summary = algo_summary.reset_index()
        detailed_summary = detailed_summary.reset_index()
        
        # Save summaries
        algo_summary_file = f"data/raw/algorithm_summary_{timestamp}.csv"
        detailed_summary_file = f"data/raw/detailed_summary_{timestamp}.csv"
        
        algo_summary.to_csv(algo_summary_file, index=False)
        detailed_summary.to_csv(detailed_summary_file, index=False)
        
        print(f"\nAlgorithm Performance Summary:")
        print("=" * 50)
        print(algo_summary[['algorithm_name', 'best_fitness_mean', 'best_fitness_std', 'success_mean']].to_string(index=False))
        
        print(f"\nSaved algorithm summary to {algo_summary_file}")
        print(f"Saved detailed summary to {detailed_summary_file}")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE DATA COLLECTION COMPLETED!")
    print("=" * 60)
    print(f"Total experiments: {len(results)}")
    print(f"Successful experiments: {len(df_clean)}")
    print(f"Error rate: {(len(results) - len(df_clean)) / len(results) * 100:.1f}%")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average time per experiment: {total_time/len(results):.2f} seconds")
    
    return df


def analyze_algorithm_performance(df):
    """Analyze and compare algorithm performance."""
    print("\nDETAILED ALGORITHM PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Filter clean results
    df_clean = df[~df.get('error', pd.Series([None]*len(df))).notna()]
    
    if len(df_clean) == 0:
        print("No successful experiments to analyze.")
        return
    
    # Performance ranking by problem type
    print("1. Best Algorithm by Problem Type:")
    print("-" * 35)
    
    for problem in df_clean['problem_name'].unique():
        problem_data = df_clean[df_clean['problem_name'] == problem]
        
        # Average performance across all dimensions
        avg_performance = problem_data.groupby('algorithm_name')['best_fitness'].mean()
        best_algo = avg_performance.idxmin()
        best_fitness = avg_performance.min()
        
        print(f"  {problem:>12}: {best_algo} (avg fitness: {best_fitness:.6f})")
    
    # Performance scaling with dimension
    print("\n2. Performance Scaling with Dimension:")
    print("-" * 40)
    
    for algo in df_clean['algorithm_name'].unique():
        print(f"\n  {algo.upper()}:")
        algo_data = df_clean[df_clean['algorithm_name'] == algo]
        
        for dim in sorted(algo_data['dimension'].unique()):
            dim_data = algo_data[algo_data['dimension'] == dim]
            avg_fitness = dim_data['best_fitness'].mean()
            print(f"    {dim}D: {avg_fitness:.6f}")
    
    # Success rate analysis
    print("\n3. Success Rate Analysis (target: 1e-6):")
    print("-" * 45)
    
    success_rates = df_clean.groupby('algorithm_name')['success'].agg(['mean', 'count'])
    success_rates['success_percentage'] = success_rates['mean'] * 100
    
    print(success_rates[['success_percentage', 'count']].round(2).to_string())
    
    # Efficiency analysis (fitness vs evaluations)
    print("\n4. Efficiency Analysis:")
    print("-" * 25)
    
    efficiency = df_clean.groupby('algorithm_name').agg({
        'best_fitness': 'mean',
        'total_evaluations': 'mean',
        'execution_time': 'mean'
    }).round(4)
    
    print(efficiency.to_string())


def main():
    """Main function."""
    print("TRANSFORMER METAHEURISTIC SELECTION - COMPREHENSIVE DATA COLLECTION")
    print("=" * 75)
    print()
    
    # Collect comprehensive data
    df = collect_comprehensive_data()
    
    # Analyze results
    analyze_algorithm_performance(df)
    
    print("\nNext steps:")
    print("1. Implement feature extraction for problem characterization")
    print("2. Develop algorithm selection model")
    print("3. Begin Transformer architecture development")
    print("4. Implement cross-validation framework")
    
    return 0


if __name__ == "__main__":
    exit(main()) 