"""
Base classes for metaheuristic optimization algorithms.

This module provides the common interface and data structures that all
metaheuristic algorithms should implement.
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable


@dataclass
class OptimizationResult:
    """Container for optimization results and metadata."""
    
    # Core results
    best_solution: np.ndarray
    best_fitness: float
    convergence_history: List[float]
    
    # Performance metrics
    total_evaluations: int
    execution_time: float
    success: bool  # Whether target fitness was reached
    
    # Algorithm metadata
    algorithm_name: str
    parameters: Dict[str, Any]
    problem_name: str
    problem_dimension: int
    
    # Additional statistics
    final_population: Optional[np.ndarray] = None
    diversity_history: Optional[List[float]] = None
    stagnation_generations: int = 0


class MetaheuristicAlgorithm(ABC):
    """Abstract base class for all metaheuristic optimization algorithms."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.parameters = kwargs
        self.random_state = kwargs.get('random_state', None)
        
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
    
    @abstractmethod
    def optimize(self, 
                 objective_function: Callable[[np.ndarray], float],
                 dimension: int,
                 bounds: tuple,
                 max_evaluations: int = 1000,
                 target_fitness: float = 1e-8,
                 **kwargs) -> OptimizationResult:
        """
        Optimize the given objective function.
        
        Args:
            objective_function: Function to minimize
            dimension: Problem dimension
            bounds: Tuple of (lower_bound, upper_bound)
            max_evaluations: Maximum number of function evaluations
            target_fitness: Target fitness value for early stopping
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            OptimizationResult containing the optimization results
        """
        pass
    
    def _initialize_population(self, 
                             population_size: int, 
                             dimension: int, 
                             bounds: tuple) -> np.ndarray:
        """Initialize a random population within bounds."""
        lower, upper = bounds
        return np.random.uniform(lower, upper, (population_size, dimension))
    
    def _clip_to_bounds(self, solution: np.ndarray, bounds: tuple) -> np.ndarray:
        """Clip solution to stay within bounds."""
        lower, upper = bounds
        return np.clip(solution, lower, upper)
    
    def _evaluate_population(self, 
                           population: np.ndarray, 
                           objective_function: Callable) -> np.ndarray:
        """Evaluate fitness for entire population."""
        return np.array([objective_function(individual) for individual in population])
    
    def _check_convergence(self, 
                         best_fitness: float, 
                         target_fitness: float,
                         evaluations: int,
                         max_evaluations: int) -> bool:
        """Check if optimization should stop."""
        return (best_fitness <= target_fitness or 
                evaluations >= max_evaluations)
    
    def _calculate_diversity(self, population: np.ndarray) -> float:
        """Calculate population diversity (average pairwise distance)."""
        if len(population) < 2:
            return 0.0
        
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distances.append(np.linalg.norm(population[i] - population[j]))
        
        return np.mean(distances) if distances else 0.0
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        return self.parameters.copy()
    
    def set_parameters(self, **kwargs):
        """Update algorithm parameters."""
        self.parameters.update(kwargs)


class PopulationBasedAlgorithm(MetaheuristicAlgorithm):
    """Base class for population-based metaheuristics."""
    
    def __init__(self, name: str, population_size: int = 30, **kwargs):
        super().__init__(name, population_size=population_size, **kwargs)
        self.population_size = population_size
    
    def _track_convergence(self, 
                         generation: int,
                         best_fitness: float,
                         population: np.ndarray,
                         convergence_history: List[float],
                         diversity_history: List[float]) -> None:
        """Track convergence metrics."""
        convergence_history.append(best_fitness)
        
        if diversity_history is not None:
            diversity = self._calculate_diversity(population)
            diversity_history.append(diversity)


class SingleSolutionAlgorithm(MetaheuristicAlgorithm):
    """Base class for single-solution metaheuristics."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
    
    def _generate_neighbor(self, 
                         current_solution: np.ndarray, 
                         bounds: tuple,
                         step_size: float = 0.1) -> np.ndarray:
        """Generate a neighbor solution."""
        dimension = len(current_solution)
        perturbation = np.random.normal(0, step_size, dimension)
        neighbor = current_solution + perturbation
        return self._clip_to_bounds(neighbor, bounds) 