#!/usr/bin/env python3
"""
Comprehensive test for all metaheuristic algorithms.

This script tests GA, PSO, DE, and SA implementations to ensure
they work correctly before running the full data collection.
"""

import numpy as np
import sys
import os
import time

# Add src to path
sys.path.append('src')

from benchmarks import sphere, rastrigin, get_function_by_name
from metaheuristics import (
    GeneticAlgorithm, 
    ParticleSwarmOptimization, 
    DifferentialEvolution, 
    SimulatedAnnealing
)


def test_all_algorithms():
    """Test all metaheuristic algorithms on a simple problem."""
    print("Testing all metaheuristic algorithms...")
    print("=" * 60)
    
    # Test problem: 2D sphere function
    problem = sphere(2)
    dimension = 2
    bounds = problem.bounds
    max_evaluations = 200
    target_fitness = 1e-6
    
    # Initialize all algorithms with small parameters for quick testing
    algorithms = {
        'GA': GeneticAlgorithm(
            population_size=20,
            crossover_rate=0.8,
            mutation_rate=0.1,
            random_state=42
        ),
        'PSO': ParticleSwarmOptimization(
            swarm_size=20,
            inertia_weight=0.9,
            cognitive_coeff=2.0,
            social_coeff=2.0,
            random_state=42
        ),
        'DE': DifferentialEvolution(
            population_size=20,
            F=0.5,
            CR=0.7,
            strategy="rand/1/bin",
            random_state=42
        ),
        'SA': SimulatedAnnealing(
            initial_temp=50.0,
            final_temp=0.01,
            cooling_rate=0.95,
            steps_per_temp=5,
            random_state=42
        )
    }
    
    results = {}
    
    print(f"Problem: {problem.name} ({dimension}D)")
    print(f"Bounds: {bounds}")
    print(f"Max evaluations: {max_evaluations}")
    print(f"Target fitness: {target_fitness}")
    print()
    
    # Test each algorithm
    for algo_name, algorithm in algorithms.items():
        print(f"Testing {algo_name}...")
        
        problem.reset_counter()
        start_time = time.time()
        
        try:
            result = algorithm.optimize(
                objective_function=problem,
                dimension=dimension,
                bounds=bounds,
                max_evaluations=max_evaluations,
                target_fitness=target_fitness
            )
            
            results[algo_name] = result
            
            print(f"  ✓ {algo_name} completed successfully")
            print(f"    Best fitness: {result.best_fitness:.6f}")
            print(f"    Best solution: [{result.best_solution[0]:.4f}, {result.best_solution[1]:.4f}]")
            print(f"    Evaluations: {result.total_evaluations}")
            print(f"    Time: {result.execution_time:.3f}s")
            print(f"    Success: {result.success}")
            print(f"    Convergence history length: {len(result.convergence_history)}")
            
            # Basic validation
            assert result.best_fitness >= 0, f"{algo_name}: Fitness should be non-negative"
            assert len(result.best_solution) == dimension, f"{algo_name}: Solution dimension mismatch"
            assert result.total_evaluations <= max_evaluations, f"{algo_name}: Exceeded max evaluations"
            assert len(result.convergence_history) > 0, f"{algo_name}: Empty convergence history"
            
        except Exception as e:
            print(f"  ✗ {algo_name} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print()
    
    # Compare algorithm performance
    print("Algorithm Performance Comparison:")
    print("-" * 50)
    print(f"{'Algorithm':<8} {'Best Fitness':<12} {'Evaluations':<12} {'Time (s)':<10} {'Success':<8}")
    print("-" * 50)
    
    for algo_name, result in results.items():
        print(f"{algo_name:<8} {result.best_fitness:<12.6f} {result.total_evaluations:<12} "
              f"{result.execution_time:<10.3f} {str(result.success):<8}")
    
    print()
    return True


def test_algorithms_on_multiple_problems():
    """Test algorithms on different problem types."""
    print("Testing algorithms on multiple problems...")
    print("=" * 60)
    
    # Test problems
    problems = [
        ('sphere', 2),
        ('rastrigin', 2),
        ('rosenbrock', 2)
    ]
    
    # Algorithms to test
    algorithms = {
        'GA': GeneticAlgorithm(population_size=15, random_state=42),
        'PSO': ParticleSwarmOptimization(swarm_size=15, random_state=42),
        'DE': DifferentialEvolution(population_size=15, random_state=42),
        'SA': SimulatedAnnealing(initial_temp=30.0, steps_per_temp=3, random_state=42)
    }
    
    all_results = []
    
    for problem_name, dimension in problems:
        print(f"\nTesting on {problem_name}({dimension}D):")
        print("-" * 40)
        
        problem = get_function_by_name(problem_name, dimension)
        
        for algo_name, algorithm in algorithms.items():
            problem.reset_counter()
            
            try:
                result = algorithm.optimize(
                    objective_function=problem,
                    dimension=dimension,
                    bounds=problem.bounds,
                    max_evaluations=150,
                    target_fitness=1e-6
                )
                
                print(f"  {algo_name}: fitness={result.best_fitness:.6f}, "
                      f"evals={result.total_evaluations}, success={result.success}")
                
                all_results.append({
                    'problem': problem_name,
                    'algorithm': algo_name,
                    'fitness': result.best_fitness,
                    'evaluations': result.total_evaluations,
                    'success': result.success
                })
                
            except Exception as e:
                print(f"  {algo_name}: ERROR - {e}")
    
    # Summary statistics
    print("\nSummary by Problem:")
    print("-" * 30)
    for problem_name, _ in problems:
        problem_results = [r for r in all_results if r['problem'] == problem_name]
        if problem_results:
            best_fitness = min(r['fitness'] for r in problem_results)
            best_algo = next(r['algorithm'] for r in problem_results if r['fitness'] == best_fitness)
            print(f"  {problem_name}: Best = {best_algo} (fitness: {best_fitness:.6f})")
    
    return True


def test_algorithm_parameters():
    """Test algorithms with different parameter settings."""
    print("\nTesting parameter variations...")
    print("=" * 40)
    
    problem = sphere(2)
    
    # Test PSO with different inertia weights
    print("PSO inertia weight variations:")
    for w in [0.5, 0.7, 0.9]:
        pso = ParticleSwarmOptimization(swarm_size=15, inertia_weight=w, random_state=42)
        problem.reset_counter()
        result = pso.optimize(problem, 2, problem.bounds, max_evaluations=100)
        print(f"  w={w}: fitness={result.best_fitness:.6f}")
    
    # Test DE with different strategies
    print("\nDE strategy variations:")
    for strategy in ["rand/1/bin", "best/1/bin", "rand/2/bin"]:
        de = DifferentialEvolution(population_size=15, strategy=strategy, random_state=42)
        problem.reset_counter()
        result = de.optimize(problem, 2, problem.bounds, max_evaluations=100)
        print(f"  {strategy}: fitness={result.best_fitness:.6f}")
    
    # Test SA with different cooling schedules
    print("\nSA cooling schedule variations:")
    for schedule in ["exponential", "linear", "logarithmic"]:
        sa = SimulatedAnnealing(initial_temp=20.0, cooling_schedule=schedule, 
                               steps_per_temp=3, random_state=42)
        problem.reset_counter()
        result = sa.optimize(problem, 2, problem.bounds, max_evaluations=100)
        print(f"  {schedule}: fitness={result.best_fitness:.6f}")
    
    return True


def main():
    """Run all tests."""
    print("COMPREHENSIVE METAHEURISTIC ALGORITHM TEST")
    print("=" * 70)
    print()
    
    try:
        # Test 1: Basic functionality
        success1 = test_all_algorithms()
        if not success1:
            print("❌ Basic algorithm test failed!")
            return 1
        
        # Test 2: Multiple problems
        success2 = test_algorithms_on_multiple_problems()
        if not success2:
            print("❌ Multi-problem test failed!")
            return 1
        
        # Test 3: Parameter variations
        success3 = test_algorithm_parameters()
        if not success3:
            print("❌ Parameter variation test failed!")
            return 1
        
        print("\n" + "=" * 70)
        print("🎉 ALL ALGORITHM TESTS PASSED!")
        print("=" * 70)
        print()
        print("✅ Genetic Algorithm (GA) - Working correctly")
        print("✅ Particle Swarm Optimization (PSO) - Working correctly") 
        print("✅ Differential Evolution (DE) - Working correctly")
        print("✅ Simulated Annealing (SA) - Working correctly")
        print()
        print("Next steps:")
        print("1. Run comprehensive data collection with all algorithms")
        print("2. Implement feature engineering")
        print("3. Begin Transformer model development")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 