#!/usr/bin/env python3
"""
Basic functionality test for the Transformer Metaheuristic Selection project.

This script tests the core components to ensure they work correctly
before running the full data collection pipeline.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

from benchmarks import sphere, rastrigin, rosenbrock, get_function_by_name
from metaheuristics import GeneticAlgorithm


def test_benchmark_functions():
    """Test benchmark function implementations."""
    print("Testing benchmark functions...")
    
    # Test sphere function
    sphere_2d = sphere(2)
    test_point = np.array([1.0, 1.0])
    result = sphere_2d(test_point)
    expected = 2.0  # 1^2 + 1^2
    
    print(f"Sphere(2D) at [1,1]: {result} (expected: {expected})")
    assert abs(result - expected) < 1e-10, f"Sphere function test failed: {result} != {expected}"
    
    # Test rastrigin function
    rastrigin_2d = rastrigin(2)
    zero_point = np.array([0.0, 0.0])
    result = rastrigin_2d(zero_point)
    expected = 0.0  # Global minimum
    
    print(f"Rastrigin(2D) at [0,0]: {result} (expected: {expected})")
    assert abs(result - expected) < 1e-10, f"Rastrigin function test failed: {result} != {expected}"
    
    # Test function metadata
    metadata = sphere_2d.get_metadata()
    print(f"Sphere metadata: {metadata}")
    
    print("✓ Benchmark functions test passed!\n")


def test_genetic_algorithm():
    """Test Genetic Algorithm implementation."""
    print("Testing Genetic Algorithm...")
    
    # Create a simple 2D sphere problem
    problem = sphere(2)
    
    # Initialize GA with small parameters for quick test
    ga = GeneticAlgorithm(
        population_size=20,
        crossover_rate=0.8,
        mutation_rate=0.1,
        random_state=42
    )
    
    # Run optimization
    result = ga.optimize(
        objective_function=problem,
        dimension=2,
        bounds=problem.bounds,
        max_evaluations=200,
        target_fitness=1e-6
    )
    
    print(f"GA Result:")
    print(f"  Best fitness: {result.best_fitness}")
    print(f"  Best solution: {result.best_solution}")
    print(f"  Total evaluations: {result.total_evaluations}")
    print(f"  Execution time: {result.execution_time:.4f}s")
    print(f"  Success: {result.success}")
    print(f"  Convergence history length: {len(result.convergence_history)}")
    
    # Basic checks
    assert result.best_fitness >= 0, "Fitness should be non-negative for sphere function"
    assert len(result.best_solution) == 2, "Solution should be 2D"
    assert result.total_evaluations <= 200, "Should not exceed max evaluations"
    assert len(result.convergence_history) > 0, "Should have convergence history"
    
    print("✓ Genetic Algorithm test passed!\n")


def test_function_registry():
    """Test function registry and factory functions."""
    print("Testing function registry...")
    
    # Test getting function by name
    functions_to_test = ['sphere', 'rastrigin', 'rosenbrock', 'ackley', 'griewank']
    
    for func_name in functions_to_test:
        try:
            func = get_function_by_name(func_name, 5)
            print(f"✓ {func_name}: {func.name}, dim={func.dimension}, bounds={func.bounds}")
            
            # Test evaluation
            test_point = np.zeros(5)
            result = func(test_point)
            print(f"  f(0) = {result}")
            
        except Exception as e:
            print(f"✗ Error with {func_name}: {e}")
            raise
    
    print("✓ Function registry test passed!\n")


def test_small_experiment():
    """Test a small experiment similar to what the data collector would do."""
    print("Testing small experiment...")
    
    # Test parameters
    problems = [('sphere', 2), ('rastrigin', 2)]
    algorithms = ['genetic_algorithm']
    
    results = []
    
    for problem_name, dimension in problems:
        for algorithm_name in algorithms:
            print(f"Running {algorithm_name} on {problem_name}({dimension}D)...")
            
            # Get problem
            problem = get_function_by_name(problem_name, dimension)
            
            # Get algorithm
            if algorithm_name == 'genetic_algorithm':
                algorithm = GeneticAlgorithm(
                    population_size=15,
                    random_state=42
                )
            
            # Run optimization
            result = algorithm.optimize(
                objective_function=problem,
                dimension=dimension,
                bounds=problem.bounds,
                max_evaluations=150,
                target_fitness=1e-6
            )
            
            results.append({
                'problem': problem_name,
                'dimension': dimension,
                'algorithm': algorithm_name,
                'best_fitness': result.best_fitness,
                'evaluations': result.total_evaluations,
                'time': result.execution_time,
                'success': result.success
            })
            
            print(f"  Result: fitness={result.best_fitness:.6f}, evals={result.total_evaluations}")
    
    print(f"\nExperiment Results Summary:")
    for result in results:
        print(f"  {result['algorithm']} on {result['problem']}({result['dimension']}D): "
              f"fitness={result['best_fitness']:.6f}, success={result['success']}")
    
    print("✓ Small experiment test passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TRANSFORMER METAHEURISTIC SELECTION - BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    print()
    
    try:
        test_benchmark_functions()
        test_genetic_algorithm()
        test_function_registry()
        test_small_experiment()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED! The basic functionality is working correctly.")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Run the data collection pipeline")
        print("2. Implement additional metaheuristics (PSO, DE, SA)")
        print("3. Add feature extraction capabilities")
        print("4. Begin Transformer model development")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 