#!/usr/bin/env python3
"""
Test feature extraction functionality.

This script tests the problem and algorithm feature extractors to ensure
they can successfully analyze optimization problems and algorithm performance.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from benchmarks import get_function_by_name
from features import ProblemFeatureExtractor, AlgorithmFeatureExtractor
from metaheuristics import GeneticAlgorithm


def test_problem_features():
    """Test problem feature extraction."""
    print("TESTING PROBLEM FEATURE EXTRACTION")
    print("=" * 50)
    
    # Initialize feature extractor
    extractor = ProblemFeatureExtractor(n_samples=500, random_state=42)
    
    # Test on different problems
    test_problems = [
        ('sphere', 2),
        ('rastrigin', 5), 
        ('rosenbrock', 10)
    ]
    
    for problem_name, dimension in test_problems:
        print(f"\nTesting {problem_name}({dimension}D):")
        print("-" * 30)
        
        # Get problem instance
        problem = get_function_by_name(problem_name, dimension)
        
        # Extract features
        features = extractor.extract_features(problem, dimension)
        
        # Display key features
        print(f"  Basic features:")
        print(f"    Dimension: {features.get('dimension', 'N/A')}")
        print(f"    Separable: {features.get('separable', 'N/A')}")
        print(f"    Bound width: {features.get('bound_width', 'N/A')}")
        
        print(f"  Statistical features:")
        print(f"    Fitness mean: {features.get('fitness_mean', 'N/A'):.6f}")
        print(f"    Fitness std: {features.get('fitness_std', 'N/A'):.6f}")
        print(f"    Valid samples: {features.get('valid_samples_ratio', 'N/A'):.3f}")
        
        print(f"  Landscape features:")
        print(f"    Modality index: {features.get('modality_index', 'N/A'):.3f}")
        print(f"    Ruggedness: {features.get('ruggedness', 'N/A'):.3f}")
        
        print(f"  Meta-features:")
        print(f"    Problem complexity: {features.get('problem_complexity', 'N/A'):.3f}")
        
        assert len(features) > 10, f"Expected multiple features, got {len(features)}"
        print(f"  ✓ Extracted {len(features)} features successfully")
    
    print(f"\n✓ Problem feature extraction test PASSED")


def test_algorithm_features():
    """Test algorithm feature extraction."""
    print("\nTESTING ALGORITHM FEATURE EXTRACTION")
    print("=" * 50)
    
    # Load the comprehensive performance data
    data_files = [f for f in os.listdir('data/raw') if f.startswith('comprehensive_performance_data')]
    
    if not data_files:
        print("  Warning: No comprehensive performance data found. Running mini experiment...")
        # Run a small experiment to generate test data
        run_mini_experiment()
        data_files = [f for f in os.listdir('data/raw') if f.startswith('comprehensive_performance_data')]
    
    if data_files:
        # Load the most recent data file
        latest_file = max(data_files)
        df = pd.read_csv(f'data/raw/{latest_file}')
        
        print(f"  Loaded data from {latest_file}")
        print(f"  Data shape: {df.shape}")
        
        # Initialize algorithm feature extractor
        extractor = AlgorithmFeatureExtractor()
        
        # Extract features by algorithm and problem
        features_df = extractor.extract_from_dataframe(
            df, 
            algorithm_column='algorithm_name',
            group_columns=['problem_name', 'dimension']
        )
        
        print(f"\n  Extracted features shape: {features_df.shape}")
        print(f"  Feature columns: {len([col for col in features_df.columns if col not in ['algorithm_name', 'problem_name', 'dimension']])}")
        
        # Display sample features for one algorithm-problem combination
        sample_row = features_df.iloc[0]
        print(f"\n  Sample features for {sample_row['algorithm_name']} on {sample_row['problem_name']}({sample_row['dimension']}D):")
        print("-" * 60)
        
        feature_cols = [col for col in features_df.columns if col not in ['algorithm_name', 'problem_name', 'dimension']]
        for col in feature_cols[:10]:  # Show first 10 features
            value = sample_row[col]
            if pd.notna(value):
                print(f"    {col}: {value:.6f}")
        
        print(f"    ... and {len(feature_cols) - 10} more features")
        
        print(f"\n  ✓ Algorithm feature extraction test PASSED")
        return features_df
    
    else:
        print("  ✗ No performance data available for testing")
        return None


def run_mini_experiment():
    """Run a mini experiment if no data is available."""
    print("  Running mini experiment to generate test data...")
    
    from metaheuristics import GeneticAlgorithm
    from benchmarks import get_function_by_name
    import time
    from dataclasses import asdict
    
    # Simple experiment
    algorithm = GeneticAlgorithm(population_size=20, max_generations=50)
    problem = get_function_by_name('sphere', 2)
    
    results = []
    for run in range(3):
        np.random.seed(42 + run)
        result = algorithm.optimize(
            objective_function=problem,
            dimension=2,
            bounds=problem.bounds,
            max_evaluations=200,
            target_fitness=1e-6
        )
        
        result_dict = asdict(result)
        result_dict.update({
            'algorithm_name': 'genetic_algorithm',
            'problem_name': 'sphere',
            'dimension': 2,
            'run_id': run
        })
        results.append(result_dict)
    
    # Save mini dataset
    df = pd.DataFrame(results)
    timestamp = int(time.time())
    filename = f"data/raw/comprehensive_performance_data_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"  Saved mini dataset to {filename}")


def test_feature_integration():
    """Test integration between problem and algorithm features."""
    print("\nTESTING FEATURE INTEGRATION")
    print("=" * 50)
    
    # Load performance data
    data_files = [f for f in os.listdir('data/raw') if f.startswith('comprehensive_performance_data')]
    
    if data_files:
        latest_file = max(data_files)
        performance_df = pd.read_csv(f'data/raw/{latest_file}')
        
        # Extract algorithm features
        algo_extractor = AlgorithmFeatureExtractor()
        algo_features = algo_extractor.extract_from_dataframe(
            performance_df,
            algorithm_column='algorithm_name',
            group_columns=['problem_name', 'dimension']
        )
        
        # Extract problem features for each problem-dimension combination
        problem_extractor = ProblemFeatureExtractor(n_samples=200)
        problem_features_list = []
        
        unique_problems = performance_df[['problem_name', 'dimension']].drop_duplicates()
        
        for _, row in unique_problems.iterrows():
            problem_name, dimension = row['problem_name'], row['dimension']
            problem = get_function_by_name(problem_name, dimension)
            
            prob_features = problem_extractor.extract_features(problem, dimension)
            prob_features['problem_name'] = problem_name
            prob_features['dimension'] = dimension
            problem_features_list.append(prob_features)
        
        problem_features_df = pd.DataFrame(problem_features_list)
        
        # Combine algorithm and problem features
        combined_features = algo_features.merge(
            problem_features_df, 
            on=['problem_name', 'dimension'], 
            how='left'
        )
        
        print(f"  Algorithm features: {algo_features.shape}")
        print(f"  Problem features: {problem_features_df.shape}")
        print(f"  Combined features: {combined_features.shape}")
        
        # Check for successful merge
        assert combined_features.shape[0] == algo_features.shape[0], "Merge failed"
        assert combined_features.shape[1] > algo_features.shape[1], "No problem features added"
        
        print(f"  ✓ Feature integration test PASSED")
        
        # Save combined features for future use
        timestamp = int(time.time())
        combined_file = f"data/processed/combined_features_{timestamp}.csv"
        os.makedirs('data/processed', exist_ok=True)
        combined_features.to_csv(combined_file, index=False)
        print(f"  Saved combined features to {combined_file}")
        
        return combined_features
    
    else:
        print("  ✗ No performance data available for integration test")
        return None


def main():
    """Main test function."""
    print("TRANSFORMER METAHEURISTIC SELECTION - FEATURE EXTRACTION TESTS")
    print("=" * 75)
    print()
    
    try:
        # Test problem feature extraction
        test_problem_features()
        
        # Test algorithm feature extraction
        algo_features = test_algorithm_features()
        
        # Test feature integration
        combined_features = test_feature_integration()
        
        print("\n" + "=" * 75)
        print("ALL FEATURE EXTRACTION TESTS PASSED!")
        print("=" * 75)
        
        if combined_features is not None:
            print(f"Total feature columns: {combined_features.shape[1]}")
            print(f"Total data points: {combined_features.shape[0]}")
            print("\nNext steps:")
            print("1. Develop machine learning models for algorithm selection")
            print("2. Implement Transformer architecture")
            print("3. Add cross-validation and evaluation framework")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 