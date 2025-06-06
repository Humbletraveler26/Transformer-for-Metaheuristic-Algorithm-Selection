"""
Continuous optimization benchmark functions for metaheuristic algorithm evaluation.

This module implements standard test functions commonly used in the optimization
literature to evaluate and compare metaheuristic algorithms.
"""

import numpy as np
from typing import Callable, Dict, List, Tuple, Any


class OptimizationFunction:
    """Base class for optimization functions with metadata."""
    
    def __init__(self, name: str, dimension: int, bounds: Tuple[float, float], 
                 global_optimum: float = 0.0, separable: bool = False):
        self.name = name
        self.dimension = dimension
        self.bounds = bounds
        self.global_optimum = global_optimum
        self.separable = separable
        self.evaluation_count = 0
    
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the function at point x."""
        self.evaluation_count += 1
        return self._evaluate(x)
    
    def _evaluate(self, x: np.ndarray) -> float:
        """Override this method in subclasses."""
        raise NotImplementedError
    
    def reset_counter(self):
        """Reset the evaluation counter."""
        self.evaluation_count = 0
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return function metadata for feature extraction."""
        return {
            'name': self.name,
            'dimension': self.dimension,
            'bounds': self.bounds,
            'global_optimum': self.global_optimum,
            'separable': self.separable,
            'evaluations': self.evaluation_count
        }


class Sphere(OptimizationFunction):
    """Sphere function: f(x) = sum(x_i^2)
    
    Global minimum: f(0,...,0) = 0
    Domain: [-5.12, 5.12]^n
    Properties: Unimodal, separable, convex
    """
    
    def __init__(self, dimension: int):
        super().__init__("sphere", dimension, (-5.12, 5.12), 0.0, True)
    
    def _evaluate(self, x: np.ndarray) -> float:
        return np.sum(x**2)


class Rastrigin(OptimizationFunction):
    """Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
    
    Global minimum: f(0,...,0) = 0
    Domain: [-5.12, 5.12]^n
    Properties: Multimodal, separable, many local minima
    """
    
    def __init__(self, dimension: int, A: float = 10.0):
        super().__init__("rastrigin", dimension, (-5.12, 5.12), 0.0, True)
        self.A = A
    
    def _evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        return self.A * n + np.sum(x**2 - self.A * np.cos(2 * np.pi * x))


class Rosenbrock(OptimizationFunction):
    """Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    
    Global minimum: f(1,...,1) = 0
    Domain: [-2.048, 2.048]^n
    Properties: Unimodal, non-separable, valley-shaped
    """
    
    def __init__(self, dimension: int):
        super().__init__("rosenbrock", dimension, (-2.048, 2.048), 0.0, False)
    
    def _evaluate(self, x: np.ndarray) -> float:
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class Ackley(OptimizationFunction):
    """Ackley function: f(x) = -a*exp(-b*sqrt(1/n*sum(x_i^2))) - exp(1/n*sum(cos(c*x_i))) + a + e
    
    Global minimum: f(0,...,0) = 0
    Domain: [-32.768, 32.768]^n
    Properties: Multimodal, non-separable, many local minima
    """
    
    def __init__(self, dimension: int, a: float = 20.0, b: float = 0.2, c: float = 2*np.pi):
        super().__init__("ackley", dimension, (-32.768, 32.768), 0.0, False)
        self.a = a
        self.b = b
        self.c = c
    
    def _evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(self.c * x))
        
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / n))
        term2 = -np.exp(sum_cos / n)
        
        return term1 + term2 + self.a + np.e


class Griewank(OptimizationFunction):
    """Griewank function: f(x) = 1/4000*sum(x_i^2) - prod(cos(x_i/sqrt(i))) + 1
    
    Global minimum: f(0,...,0) = 0
    Domain: [-600, 600]^n
    Properties: Multimodal, non-separable
    """
    
    def __init__(self, dimension: int):
        super().__init__("griewank", dimension, (-600, 600), 0.0, False)
    
    def _evaluate(self, x: np.ndarray) -> float:
        sum_sq = np.sum(x**2)
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        
        return sum_sq / 4000 - prod_cos + 1


class Schwefel(OptimizationFunction):
    """Schwefel function: f(x) = 418.9829*n - sum(x_i * sin(sqrt(|x_i|)))
    
    Global minimum: f(420.9687,...,420.9687) = 0
    Domain: [-500, 500]^n
    Properties: Multimodal, separable, deceptive (global minimum far from origin)
    """
    
    def __init__(self, dimension: int):
        super().__init__("schwefel", dimension, (-500, 500), 0.0, True)
    
    def _evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


class Levy(OptimizationFunction):
    """Levy function: Complex multimodal function with many local minima
    
    f(x) = sin^2(π*w_1) + sum((w_i-1)^2 * (1 + 10*sin^2(π*w_i + 1))) + (w_n-1)^2 * (1 + sin^2(2*π*w_n))
    where w_i = 1 + (x_i - 1)/4
    
    Global minimum: f(1,...,1) = 0
    Domain: [-10, 10]^n
    Properties: Multimodal, non-separable, many local minima
    """
    
    def __init__(self, dimension: int):
        super().__init__("levy", dimension, (-10, 10), 0.0, False)
    
    def _evaluate(self, x: np.ndarray) -> float:
        w = 1 + (x - 1) / 4
        
        term1 = np.sin(np.pi * w[0])**2
        
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        
        return term1 + term2 + term3


class Zakharov(OptimizationFunction):
    """Zakharov function: f(x) = sum(x_i^2) + (sum(0.5*i*x_i))^2 + (sum(0.5*i*x_i))^4
    
    Global minimum: f(0,...,0) = 0
    Domain: [-5, 10]^n
    Properties: Unimodal, non-separable
    """
    
    def __init__(self, dimension: int):
        super().__init__("zakharov", dimension, (-5, 10), 0.0, False)
    
    def _evaluate(self, x: np.ndarray) -> float:
        term1 = np.sum(x**2)
        
        indices = np.arange(1, len(x) + 1)
        sum_term = np.sum(0.5 * indices * x)
        
        term2 = sum_term**2
        term3 = sum_term**4
        
        return term1 + term2 + term3


class DixonPrice(OptimizationFunction):
    """Dixon-Price function: f(x) = (x_1 - 1)^2 + sum(i*(2*x_i^2 - x_{i-1})^2)
    
    Global minimum: f(x*) = 0, where x*_i = 2^(-(2^i - 2)/2^i)
    Domain: [-10, 10]^n
    Properties: Unimodal, non-separable
    """
    
    def __init__(self, dimension: int):
        super().__init__("dixon_price", dimension, (-10, 10), 0.0, False)
    
    def _evaluate(self, x: np.ndarray) -> float:
        term1 = (x[0] - 1)**2
        
        indices = np.arange(2, len(x) + 1)
        term2 = np.sum(indices * (2 * x[1:]**2 - x[:-1])**2)
        
        return term1 + term2


class Michalewicz(OptimizationFunction):
    """Michalewicz function: f(x) = -sum(sin(x_i) * (sin(i*x_i^2/π))^(2*m))
    
    Global minimum: depends on dimension and m parameter
    Domain: [0, π]^n
    Properties: Multimodal, separable, many local minima
    """
    
    def __init__(self, dimension: int, m: int = 10):
        # Global minimum values for different dimensions (approximate)
        global_mins = {2: -1.8013, 5: -4.687, 10: -9.66}
        global_min = global_mins.get(dimension, -dimension)
        
        super().__init__("michalewicz", dimension, (0, np.pi), global_min, True)
        self.m = m
    
    def _evaluate(self, x: np.ndarray) -> float:
        indices = np.arange(1, len(x) + 1)
        return -np.sum(np.sin(x) * (np.sin(indices * x**2 / np.pi))**(2 * self.m))


class Powell(OptimizationFunction):
    """Powell function: f(x) = sum((x_{4i-3} + 10*x_{4i-2})^2 + 5*(x_{4i-1} - x_{4i})^2 + 
                                  (x_{4i-2} - 2*x_{4i-1})^4 + 10*(x_{4i-3} - x_{4i})^4)
    
    Global minimum: f(0,...,0) = 0
    Domain: [-4, 5]^n
    Properties: Unimodal, non-separable, requires n to be multiple of 4
    """
    
    def __init__(self, dimension: int):
        if dimension % 4 != 0:
            raise ValueError("Powell function requires dimension to be multiple of 4")
        super().__init__("powell", dimension, (-4, 5), 0.0, False)
    
    def _evaluate(self, x: np.ndarray) -> float:
        result = 0
        for i in range(0, len(x), 4):
            if i + 3 < len(x):
                term1 = (x[i] + 10 * x[i+1])**2
                term2 = 5 * (x[i+2] - x[i+3])**2
                term3 = (x[i+1] - 2 * x[i+2])**4
                term4 = 10 * (x[i] - x[i+3])**4
                result += term1 + term2 + term3 + term4
        return result


class Styblinski(OptimizationFunction):
    """Styblinski-Tang function: f(x) = 0.5 * sum(x_i^4 - 16*x_i^2 + 5*x_i)
    
    Global minimum: f(-2.903534,...,-2.903534) ≈ -39.16599*n
    Domain: [-5, 5]^n
    Properties: Multimodal, separable
    """
    
    def __init__(self, dimension: int):
        global_min = -39.16599 * dimension  # Approximate
        super().__init__("styblinski", dimension, (-5, 5), global_min, True)
    
    def _evaluate(self, x: np.ndarray) -> float:
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)


# Factory functions
def sphere(dimension: int) -> Sphere:
    """Create a Sphere function instance."""
    return Sphere(dimension)

def rastrigin(dimension: int) -> Rastrigin:
    """Create a Rastrigin function instance."""
    return Rastrigin(dimension)

def rosenbrock(dimension: int) -> Rosenbrock:
    """Create a Rosenbrock function instance."""
    return Rosenbrock(dimension)

def ackley(dimension: int) -> Ackley:
    """Create an Ackley function instance."""
    return Ackley(dimension)

def griewank(dimension: int) -> Griewank:
    """Create a Griewank function instance."""
    return Griewank(dimension)

def schwefel(dimension: int) -> Schwefel:
    """Create a Schwefel function instance."""
    return Schwefel(dimension)

def levy(dimension: int) -> Levy:
    """Create a Levy function instance."""
    return Levy(dimension)

def zakharov(dimension: int) -> Zakharov:
    """Create a Zakharov function instance."""
    return Zakharov(dimension)

def dixon_price(dimension: int) -> DixonPrice:
    """Create a Dixon-Price function instance."""
    return DixonPrice(dimension)

def michalewicz(dimension: int) -> Michalewicz:
    """Create a Michalewicz function instance."""
    return Michalewicz(dimension)

def powell(dimension: int) -> Powell:
    """Create a Powell function instance."""
    return Powell(dimension)

def styblinski(dimension: int) -> Styblinski:
    """Create a Styblinski-Tang function instance."""
    return Styblinski(dimension)


# Registry of available functions
FUNCTION_REGISTRY = {
    'sphere': sphere,
    'rastrigin': rastrigin,
    'rosenbrock': rosenbrock,
    'ackley': ackley,
    'griewank': griewank,
    'schwefel': schwefel,
    'levy': levy,
    'zakharov': zakharov,
    'dixon_price': dixon_price,
    'michalewicz': michalewicz,
    'powell': powell,
    'styblinski': styblinski
}


def get_all_functions() -> Dict[str, Callable]:
    """Get all available benchmark functions."""
    return FUNCTION_REGISTRY.copy()


def get_function_by_name(name: str, dimension: int) -> OptimizationFunction:
    """Get a function instance by name and dimension."""
    if name not in FUNCTION_REGISTRY:
        raise ValueError(f"Unknown function: {name}. Available: {list(FUNCTION_REGISTRY.keys())}")
    
    return FUNCTION_REGISTRY[name](dimension)


def generate_problem_instances(dimensions: List[int], 
                             function_names: List[str] = None) -> List[OptimizationFunction]:
    """Generate a list of problem instances for experimentation."""
    if function_names is None:
        function_names = list(FUNCTION_REGISTRY.keys())
    
    instances = []
    for dim in dimensions:
        for func_name in function_names:
            try:
                instances.append(get_function_by_name(func_name, dim))
            except ValueError as e:
                # Skip functions that don't support certain dimensions
                print(f"Skipping {func_name} with dimension {dim}: {e}")
                continue
    
    return instances


def get_function_characteristics() -> Dict[str, Dict[str, Any]]:
    """Get characteristics of all benchmark functions for analysis."""
    characteristics = {}
    
    for name, factory in FUNCTION_REGISTRY.items():
        try:
            # Create instance with dimension 2 for characteristics
            func = factory(2)
            characteristics[name] = {
                'separable': func.separable,
                'bounds': func.bounds,
                'global_optimum': func.global_optimum,
                'properties': _get_function_properties(name)
            }
        except Exception as e:
            print(f"Error getting characteristics for {name}: {e}")
    
    return characteristics


def _get_function_properties(func_name: str) -> List[str]:
    """Get descriptive properties of each function."""
    properties_map = {
        'sphere': ['unimodal', 'convex', 'smooth'],
        'rastrigin': ['multimodal', 'many_local_minima', 'separable'],
        'rosenbrock': ['unimodal', 'valley_shaped', 'ill_conditioned'],
        'ackley': ['multimodal', 'plate_shaped', 'many_local_minima'],
        'griewank': ['multimodal', 'product_term', 'scaling_dependent'],
        'schwefel': ['multimodal', 'deceptive', 'global_min_far_from_origin'],
        'levy': ['multimodal', 'many_local_minima', 'complex_landscape'],
        'zakharov': ['unimodal', 'ill_conditioned', 'higher_order_terms'],
        'dixon_price': ['unimodal', 'ridge_like', 'asymmetric'],
        'michalewicz': ['multimodal', 'many_local_minima', 'steep_ridges'],
        'powell': ['unimodal', 'non_separable', 'high_condition_number'],
        'styblinski': ['multimodal', 'separable', 'many_global_minima']
    }
    
    return properties_map.get(func_name, ['unknown']) 