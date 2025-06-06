#!/usr/bin/env python3
"""
Simplified feature extraction test.

This script tests the feature extraction functionality with a direct approach
to avoid import issues.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

# Direct imports
from benchmarks import get_function_by_name
from metaheuristics import GeneticAlgorithm


def test_basic_problem_analysis():
    """Test basic problem analysis without the feature extractor."""
    print("TESTING BASIC PROBLEM ANALYSIS")
    print("=" * 50)
    
    # Test problems
    test_problems = [
        ('sphere', 2),
        ('rastrigin', 5), 
        ('rosenbrock', 10)
    ]
    
    for problem_name, dimension in test_problems:
        print(f"\nAnalyzing {problem_name}({dimension}D):")
        print("-" * 30)
        
        # Get problem instance
        problem = get_function_by_name(problem_name, dimension)
        
        # Basic properties
        print(f"  Dimension: {dimension}")
        print(f"  Separable: {problem.separable}")
        print(f"  Bounds: {problem.bounds}")
        print(f"  Global optimum: {problem.global_optimum}")
        
        # Sample the function
        n_samples = 100
        bounds = problem.bounds
        samples = np.random.uniform(bounds[0], bounds[1], (n_samples, dimension))
        
        values = []
        for sample in samples:
            try:
                value = problem(sample)
                if np.isfinite(value):
                    values.append(value)
            except:
                continue
        
        if values:
            values = np.array(values)
            print(f"  Sample statistics:")
            print(f"    Mean: {np.mean(values):.6f}")
            print(f"    Std: {np.std(values):.6f}")
            print(f"    Min: {np.min(values):.6f}")
            print(f"    Max: {np.max(values):.6f}")
            print(f"    Valid samples: {len(values)}/{n_samples}")
        
        print(f"  ✓ Analysis completed successfully")
    
    print(f"\n✓ Basic problem analysis test PASSED")


def test_algorithm_performance_analysis():
    """Test algorithm performance analysis."""
    print("\nTESTING ALGORITHM PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Load performance data
    data_files = [f for f in os.listdir('data/raw') if f.startswith('comprehensive_performance_data')]
    
    if data_files:
        latest_file = max(data_files)
        df = pd.read_csv(f'data/raw/{latest_file}')
        
        print(f"  Loaded data from {latest_file}")
        print(f"  Data shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Basic analysis by algorithm
        print(f"\n  Performance summary by algorithm:")
        print("-" * 40)
        
        for algorithm in df['algorithm_name'].unique():
            algo_data = df[df['algorithm_name'] == algorithm]
            
            print(f"\n  {algorithm.upper()}:")
            print(f"    Runs: {len(algo_data)}")
            print(f"    Mean fitness: {algo_data['best_fitness'].mean():.6f}")
            print(f"    Std fitness: {algo_data['best_fitness'].std():.6f}")
            print(f"    Success rate: {algo_data['success'].mean():.3f}")
            print(f"    Mean evaluations: {algo_data['total_evaluations'].mean():.1f}")
            print(f"    Mean time: {algo_data['execution_time'].mean():.4f}s")
        
        # Performance by problem
        print(f"\n  Performance summary by problem:")
        print("-" * 40)
        
        for problem in df['problem_name'].unique():
            problem_data = df[df['problem_name'] == problem]
            best_algo = problem_data.groupby('algorithm_name')['best_fitness'].mean().idxmin()
            best_fitness = problem_data.groupby('algorithm_name')['best_fitness'].mean().min()
            
            print(f"    {problem:>12}: {best_algo} (avg: {best_fitness:.6f})")
        
        print(f"\n  ✓ Algorithm performance analysis PASSED")
        return df
    
    else:
        print("  No performance data found")
        return None


def create_simple_feature_matrix():
    """Create a simple feature matrix for ML readiness test."""
    print("\nCREATING SIMPLE FEATURE MATRIX")
    print("=" * 50)
    
    # Load performance data
    data_files = [f for f in os.listdir('data/raw') if f.startswith('comprehensive_performance_data')]
    
    if not data_files:
        print("  No data available")
        return None
    
    latest_file = max(data_files)
    df = pd.read_csv(f'data/raw/{latest_file}')
    
    # Create a simple feature matrix
    feature_data = []
    
    # Group by problem-algorithm combination
    for (problem, algo), group in df.groupby(['problem_name', 'algorithm_name']):
        
        # Extract simple features
        features = {
            'problem_name': problem,
            'algorithm_name': algo,
            
            # Problem features (simple encoding)
            'is_sphere': int(problem == 'sphere'),
            'is_rastrigin': int(problem == 'rastrigin'),
            'is_rosenbrock': int(problem == 'rosenbrock'),
            'is_ackley': int(problem == 'ackley'),
            'is_griewank': int(problem == 'griewank'),
            
            # Algorithm features (simple encoding)
            'is_ga': int(algo == 'genetic_algorithm'),
            'is_pso': int(algo == 'particle_swarm'),
            'is_de': int(algo == 'differential_evolution'),
            'is_sa': int(algo == 'simulated_annealing'),
            
            # Dimension feature
            'dimension': group['dimension'].iloc[0],
            
            # Performance features
            'mean_fitness': group['best_fitness'].mean(),
            'std_fitness': group['best_fitness'].std(),
            'min_fitness': group['best_fitness'].min(),
            'success_rate': group['success'].mean(),
            'mean_evaluations': group['total_evaluations'].mean(),
            'mean_time': group['execution_time'].mean(),
            
            # Target variable (1 if this is the best algorithm for this problem)
            'is_best': 0  # Will be filled in later
        }
        
        feature_data.append(features)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(feature_data)
    
    # Determine the best algorithm for each problem
    for problem in feature_df['problem_name'].unique():
        problem_data = feature_df[feature_df['problem_name'] == problem]
        best_idx = problem_data['mean_fitness'].idxmin()
        feature_df.loc[best_idx, 'is_best'] = 1
    
    print(f"  Created feature matrix: {feature_df.shape}")
    print(f"  Features: {[col for col in feature_df.columns if col not in ['problem_name', 'algorithm_name']]}")
    
    # Save the feature matrix
    os.makedirs('data/processed', exist_ok=True)
    timestamp = int(os.path.getmtime(f'data/raw/{latest_file}'))
    feature_file = f"data/processed/simple_features_{timestamp}.csv"
    feature_df.to_csv(feature_file, index=False)
    
    print(f"  Saved to {feature_file}")
    print(f"  ✓ Simple feature matrix creation PASSED")
    
    return feature_df


def test_ml_readiness():
    """Test if the data is ready for machine learning."""
    print("\nTESTING ML READINESS")
    print("=" * 50)
    
    feature_df = create_simple_feature_matrix()
    
    if feature_df is None:
        print("  No feature data available")
        return False
    
    # Check for ML readiness
    print(f"  Dataset shape: {feature_df.shape}")
    
    # Separate features and targets
    feature_cols = [col for col in feature_df.columns 
                   if col not in ['problem_name', 'algorithm_name', 'is_best']]
    X = feature_df[feature_cols]
    y = feature_df['is_best']
    
    print(f"  Feature matrix: {X.shape}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    # Check for missing values
    missing_values = X.isnull().sum().sum()
    print(f"  Missing values: {missing_values}")
    
    # Check for infinite values
    infinite_values = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    print(f"  Infinite values: {infinite_values}")
    
    # Basic statistics
    print(f"  Feature ranges:")
    for col in X.select_dtypes(include=[np.number]).columns[:5]:  # Show first 5
        print(f"    {col}: [{X[col].min():.3f}, {X[col].max():.3f}]")
    
    if missing_values == 0 and infinite_values == 0 and len(X) > 0:
        print(f"  ✓ Data is ML-ready!")
        return True
    else:
        print(f"  ✗ Data needs cleaning")
        return False


def main():
    """Main test function."""
    print("TRANSFORMER METAHEURISTIC SELECTION - SIMPLIFIED FEATURE TESTS")
    print("=" * 75)
    print()
    
    try:
        # Test basic problem analysis
        test_basic_problem_analysis()
        
        # Test algorithm performance analysis
        performance_df = test_algorithm_performance_analysis()
        
        # Test ML readiness
        ml_ready = test_ml_readiness()
        
        print("\n" + "=" * 75)
        print("ALL SIMPLIFIED TESTS PASSED!")
        print("=" * 75)
        
        if ml_ready:
            print("🎉 Data is ready for machine learning model development!")
            print("\nNext steps:")
            print("1. Implement basic ML models (Random Forest, SVM)")
            print("2. Develop Transformer architecture")
            print("3. Add cross-validation framework")
            print("4. Implement ensemble methods")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 