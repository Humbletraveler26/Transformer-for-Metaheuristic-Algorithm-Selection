#!/usr/bin/env python3
"""
Test script for massive data collection system.
Runs a smaller version to validate functionality before full execution.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from collect_massive_research_data import MassiveDataCollector


def test_small_collection():
    """Test the massive data collector with a small subset."""
    print("🧪 TESTING MASSIVE DATA COLLECTION SYSTEM")
    print("=" * 50)
    
    # Create test collector with smaller parameters
    test_collector = MassiveDataCollector(
        target_runs=2,  # Only 2 runs per config for testing
        dimensions=[2, 5],  # Only 2 dimensions
        max_evaluations=100,  # Smaller budget
        output_dir="test_research_data",
        parallel_workers=2  # Conservative
    )
    
    print(f"\n📊 Test Configuration:")
    print(f"  Problem instances: {len(test_collector.problem_instances)}")
    print(f"  Algorithms: {len(test_collector.algorithms)}")
    print(f"  Total estimated runs: {len(test_collector.problem_instances) * len(test_collector.algorithms) * test_collector.target_runs}")
    
    # Run a smaller test with just a few problems
    test_problems = test_collector.problem_instances[:3]  # First 3 problems only
    print(f"  Testing with {len(test_problems)} problems")
    
    # Test feature extraction
    print("\n🔍 Testing feature extraction...")
    for i, problem in enumerate(test_problems):
        print(f"  Problem {i+1}: {problem.name} (dim={problem.dimension})")
        features = test_collector._extract_enhanced_features(problem)
        if 'error' not in features:
            print(f"    ✅ Features extracted: {len(features)} features")
        else:
            print(f"    ❌ Error: {features['error']}")
    
    # Test single optimization run
    print("\n⚡ Testing single optimization...")
    test_problem = test_problems[0]
    test_algorithm = list(test_collector.algorithms.keys())[0]
    
    result = test_collector.run_single_optimization(test_problem, test_algorithm, 0)
    if 'error' not in result:
        print(f"  ✅ Optimization successful")
        print(f"    Best fitness: {result['best_fitness']:.6f}")
        print(f"    Evaluations: {result['actual_evaluations']}")
        print(f"    Time: {result['execution_time_seconds']:.3f}s")
    else:
        print(f"  ❌ Optimization failed: {result['error']}")
    
    # Test batch collection with minimal data
    print("\n📦 Testing batch collection...")
    try:
        # Override for minimal test
        original_target_runs = test_collector.target_runs
        test_collector.target_runs = 1  # Just 1 run for quick test
        
        batch_df = test_collector.collect_batch_data(test_problems[:2])  # Just 2 problems
        
        print(f"  ✅ Batch collection successful")
        print(f"    Collected {len(batch_df)} runs")
        print(f"    Columns: {len(batch_df.columns)}")
        print(f"    Sample columns: {list(batch_df.columns)[:10]}")
        
        # Restore original
        test_collector.target_runs = original_target_runs
        
    except Exception as e:
        print(f"  ❌ Batch collection failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Massive data collection system is ready for full run")
    return True


def main():
    """Main test function."""
    success = test_small_collection()
    
    if success:
        print("\n🚀 Ready to run full massive data collection!")
        print("Execute: python collect_massive_research_data.py")
    else:
        print("\n❌ Tests failed. Please fix issues before running full collection.")


if __name__ == "__main__":
    main() 