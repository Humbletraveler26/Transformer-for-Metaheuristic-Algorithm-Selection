#!/usr/bin/env python3
"""
Test script for new benchmark functions.
Validates implementations and shows function characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from benchmarks.continuous_functions import (
    get_all_functions, get_function_by_name, 
    get_function_characteristics, generate_problem_instances
)


def test_function_basic(func_name: str, dimension: int = 2) -> Dict:
    """Test basic functionality of a benchmark function."""
    print(f"\n🧪 Testing {func_name} (dim={dimension})")
    
    try:
        # Create function instance
        func = get_function_by_name(func_name, dimension)
        
        # Test evaluation at origin
        x_origin = np.zeros(dimension)
        f_origin = func(x_origin)
        
        # Test evaluation at global optimum location (if known)
        if func_name == 'schwefel':
            x_optimum = np.full(dimension, 420.9687)
        elif func_name == 'levy':
            x_optimum = np.ones(dimension)
        elif func_name == 'zakharov':
            x_optimum = np.zeros(dimension)
        elif func_name == 'dixon_price':
            # Global optimum has complex form
            x_optimum = np.array([2**(-((2**i - 2) / 2**i)) for i in range(1, dimension + 1)])
        elif func_name == 'michalewicz':
            x_optimum = None  # Complex optimum
        elif func_name == 'powell':
            x_optimum = np.zeros(dimension)
        elif func_name == 'styblinski':
            x_optimum = np.full(dimension, -2.903534)
        else:
            x_optimum = None
        
        f_optimum = None
        if x_optimum is not None:
            f_optimum = func(x_optimum)
        
        # Test random evaluations
        np.random.seed(42)
        x_random = np.random.uniform(func.bounds[0], func.bounds[1], dimension)
        f_random = func(x_random)
        
        # Test bounds
        bounds = func.bounds
        
        # Test metadata
        metadata = func.get_metadata()
        
        result = {
            'success': True,
            'f_origin': f_origin,
            'f_optimum': f_optimum,
            'f_random': f_random,
            'bounds': bounds,
            'global_optimum': func.global_optimum,
            'separable': func.separable,
            'evaluations': func.evaluation_count,
            'metadata': metadata
        }
        
        print(f"  ✅ Success!")
        print(f"  📊 f(origin) = {f_origin:.6f}")
        if f_optimum is not None:
            print(f"  🎯 f(optimum) = {f_optimum:.6f}")
        print(f"  🎲 f(random) = {f_random:.6f}")
        print(f"  📏 Bounds: {bounds}")
        print(f"  🔢 Global minimum: {func.global_optimum}")
        print(f"  🧮 Separable: {func.separable}")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {'success': False, 'error': str(e)}


def test_all_new_functions():
    """Test all newly implemented benchmark functions."""
    print("🚀 Testing New Benchmark Functions")
    print("=" * 50)
    
    new_functions = [
        'schwefel', 'levy', 'zakharov', 'dixon_price', 
        'michalewicz', 'powell', 'styblinski'
    ]
    
    results = {}
    
    for func_name in new_functions:
        # Test with dimension 2
        if func_name == 'powell':
            # Powell requires dimension to be multiple of 4
            results[func_name] = test_function_basic(func_name, 4)
        else:
            results[func_name] = test_function_basic(func_name, 2)
    
    return results


def test_function_scaling():
    """Test functions with different dimensions."""
    print("\n🔬 Testing Function Scaling")
    print("=" * 30)
    
    dimensions = [2, 5, 10]
    test_functions = ['schwefel', 'levy', 'zakharov']
    
    for func_name in test_functions:
        print(f"\n📈 {func_name.title()} scaling:")
        for dim in dimensions:
            try:
                func = get_function_by_name(func_name, dim)
                x_origin = np.zeros(dim)
                f_origin = func(x_origin)
                print(f"  Dim {dim:2d}: f(origin) = {f_origin:10.6f}")
            except Exception as e:
                print(f"  Dim {dim:2d}: Error - {e}")


def visualize_2d_functions():
    """Create 2D visualizations of the new functions."""
    print("\n🎨 Creating 2D Visualizations")
    print("=" * 30)
    
    functions_to_plot = ['schwefel', 'levy', 'zakharov', 'styblinski']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, func_name in enumerate(functions_to_plot):
        try:
            func = get_function_by_name(func_name, 2)
            
            # Create grid for plotting
            x_range = np.linspace(func.bounds[0], func.bounds[1], 100)
            y_range = np.linspace(func.bounds[0], func.bounds[1], 100)
            X, Y = np.meshgrid(x_range, y_range)
            
            # Evaluate function on grid
            Z = np.zeros_like(X)
            for j in range(X.shape[0]):
                for k in range(X.shape[1]):
                    Z[j, k] = func(np.array([X[j, k], Y[j, k]]))
            
            # Plot contour
            ax = axes[i]
            contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
            ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
            ax.set_title(f'{func_name.title()} Function')
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            
            print(f"  ✅ {func_name} plotted")
            
        except Exception as e:
            print(f"  ❌ Error plotting {func_name}: {e}")
            axes[i].text(0.5, 0.5, f'Error: {func_name}', 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    plt.savefig('new_benchmark_functions_2d.png', dpi=300, bbox_inches='tight')
    print(f"  💾 Saved visualization as 'new_benchmark_functions_2d.png'")
    

def show_function_characteristics():
    """Display characteristics of all functions."""
    print("\n📋 Function Characteristics")
    print("=" * 40)
    
    characteristics = get_function_characteristics()
    
    print(f"{'Function':<15} {'Separable':<10} {'Bounds':<20} {'Global Min':<12} {'Properties'}")
    print("-" * 80)
    
    for name, char in characteristics.items():
        bounds_str = f"[{char['bounds'][0]}, {char['bounds'][1]}]"
        properties_str = ", ".join(char['properties'][:2])  # Show first 2 properties
        
        print(f"{name:<15} {str(char['separable']):<10} {bounds_str:<20} "
              f"{char['global_optimum']:<12.3f} {properties_str}")


def test_comprehensive_suite():
    """Test comprehensive problem generation."""
    print("\n🏗️  Testing Comprehensive Suite Generation")
    print("=" * 45)
    
    dimensions = [2, 5, 10]
    all_functions = list(get_all_functions().keys())
    
    print(f"Available functions: {len(all_functions)}")
    print(f"Functions: {', '.join(all_functions)}")
    
    # Generate problem instances
    try:
        instances = generate_problem_instances(dimensions, all_functions)
        print(f"\n✅ Generated {len(instances)} problem instances")
        
        # Count by dimension
        dim_counts = {}
        for instance in instances:
            dim = instance.dimension
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
        
        print("📊 Distribution by dimension:")
        for dim, count in sorted(dim_counts.items()):
            print(f"  Dimension {dim}: {count} problems")
            
        return instances
        
    except Exception as e:
        print(f"❌ Error generating instances: {e}")
        return []


def main():
    """Main test function."""
    print("🧬 NEW BENCHMARK FUNCTIONS TEST SUITE")
    print("=" * 50)
    
    # Test basic functionality
    test_results = test_all_new_functions()
    
    # Test scaling
    test_function_scaling()
    
    # Show characteristics
    show_function_characteristics()
    
    # Test comprehensive suite
    instances = test_comprehensive_suite()
    
    # Create visualizations
    try:
        visualize_2d_functions()
    except Exception as e:
        print(f"❌ Visualization error: {e}")
    
    # Summary
    print("\n📈 SUMMARY")
    print("=" * 20)
    
    successful_tests = sum(1 for result in test_results.values() if result.get('success', False))
    total_tests = len(test_results)
    
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"📊 Total functions available: {len(get_all_functions())}")
    print(f"🎯 Ready for massive data collection!")
    
    if successful_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! New benchmarks are ready for Option A.")
    else:
        print(f"\n⚠️  {total_tests - successful_tests} tests failed. Check errors above.")


if __name__ == "__main__":
    main() 