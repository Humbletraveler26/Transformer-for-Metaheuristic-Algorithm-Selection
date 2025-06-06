"""
Differential Evolution implementation for continuous optimization.

This module implements the Differential Evolution algorithm with various
mutation strategies, crossover operators, and selection mechanisms.
"""

import time
import numpy as np
from typing import Callable, List, Tuple
from .base import PopulationBasedAlgorithm, OptimizationResult


class DifferentialEvolution(PopulationBasedAlgorithm):
    """
    Differential Evolution for continuous optimization problems.
    
    Features:
    - Multiple mutation strategies (rand/1/bin, best/1/bin, rand/2/bin, etc.)
    - Binomial and exponential crossover
    - Self-adaptive parameter control
    - Boundary constraint handling
    """
    
    def __init__(self, 
                 population_size: int = 30,
                 F: float = 0.5,  # Scaling factor
                 CR: float = 0.7,  # Crossover rate
                 strategy: str = "rand/1/bin",
                 adaptive: bool = False,
                 **kwargs):
        
        super().__init__("differential_evolution", population_size=population_size, **kwargs)
        
        self.F = F
        self.CR = CR
        self.strategy = strategy
        self.adaptive = adaptive
        
        # Validate strategy
        valid_strategies = [
            "rand/1/bin", "rand/1/exp", "best/1/bin", "best/1/exp",
            "rand/2/bin", "rand/2/exp", "best/2/bin", "best/2/exp"
        ]
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Valid strategies: {valid_strategies}")
        
        # Update parameters dict
        self.parameters.update({
            'F': F,
            'CR': CR,
            'strategy': strategy,
            'adaptive': adaptive
        })
    
    def optimize(self, 
                 objective_function: Callable[[np.ndarray], float],
                 dimension: int,
                 bounds: tuple,
                 max_evaluations: int = 1000,
                 target_fitness: float = 1e-8,
                 **kwargs) -> OptimizationResult:
        """
        Run the Differential Evolution algorithm.
        """
        start_time = time.time()
        
        # Initialize population
        population = self._initialize_population(self.population_size, dimension, bounds)
        fitness = self._evaluate_population(population, objective_function)
        evaluations = self.population_size
        
        # Track best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # Convergence tracking
        convergence_history = [best_fitness]
        diversity_history = [self._calculate_diversity(population)]
        generation = 0
        stagnation_count = 0
        
        # Adaptive parameters
        current_F = self.F
        current_CR = self.CR
        
        # Main evolution loop
        while not self._check_convergence(best_fitness, target_fitness, evaluations, max_evaluations):
            generation += 1
            new_population = population.copy()
            
            # Generate trial vectors for each individual
            for i in range(self.population_size):
                # Generate mutant vector
                mutant = self._generate_mutant(population, i, best_idx, current_F, bounds)
                
                # Apply crossover
                trial = self._crossover(population[i], mutant, current_CR, dimension)
                
                # Ensure bounds
                trial = self._clip_to_bounds(trial, bounds)
                
                # Evaluate trial vector
                trial_fitness = objective_function(trial)
                evaluations += 1
                
                # Selection: keep trial if better
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    
                    # Update best solution
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial.copy()
                        stagnation_count = 0
                
                # Check if we've exceeded evaluation budget
                if evaluations >= max_evaluations:
                    break
            
            # Update population
            population = new_population
            
            # Update best index
            best_idx = np.argmin(fitness)
            if fitness[best_idx] == best_fitness:
                stagnation_count += 1
            
            # Adaptive parameter control
            if self.adaptive and generation % 10 == 0:
                current_F, current_CR = self._adapt_parameters(current_F, current_CR, stagnation_count)
            
            # Track convergence
            self._track_convergence(generation, best_fitness, population, 
                                  convergence_history, diversity_history)
        
        execution_time = time.time() - start_time
        success = best_fitness <= target_fitness
        
        return OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            convergence_history=convergence_history,
            total_evaluations=evaluations,
            execution_time=execution_time,
            success=success,
            algorithm_name=self.name,
            parameters=self.get_parameters(),
            problem_name=getattr(objective_function, 'name', 'unknown'),
            problem_dimension=dimension,
            final_population=population,
            diversity_history=diversity_history,
            stagnation_generations=stagnation_count
        )
    
    def _generate_mutant(self, population: np.ndarray, target_idx: int, best_idx: int, 
                        F: float, bounds: tuple) -> np.ndarray:
        """Generate mutant vector according to the selected strategy."""
        pop_size, dimension = population.shape
        
        # Select random indices (different from target)
        candidates = list(range(pop_size))
        candidates.remove(target_idx)
        
        if "rand" in self.strategy:
            # Random-based strategies
            if "/1/" in self.strategy:
                # rand/1 strategy: v = x_r1 + F * (x_r2 - x_r3)
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutant = population[r1] + F * (population[r2] - population[r3])
            elif "/2/" in self.strategy:
                # rand/2 strategy: v = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
                r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
                mutant = (population[r1] + 
                         F * (population[r2] - population[r3]) + 
                         F * (population[r4] - population[r5]))
        
        elif "best" in self.strategy:
            # Best-based strategies
            if "/1/" in self.strategy:
                # best/1 strategy: v = x_best + F * (x_r1 - x_r2)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutant = population[best_idx] + F * (population[r1] - population[r2])
            elif "/2/" in self.strategy:
                # best/2 strategy: v = x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)
                r1, r2, r3, r4 = np.random.choice(candidates, 4, replace=False)
                mutant = (population[best_idx] + 
                         F * (population[r1] - population[r2]) + 
                         F * (population[r3] - population[r4]))
        
        # Apply boundary constraints
        return self._clip_to_bounds(mutant, bounds)
    
    def _crossover(self, target: np.ndarray, mutant: np.ndarray, 
                  CR: float, dimension: int) -> np.ndarray:
        """Apply crossover operation to generate trial vector."""
        
        if "/bin" in self.strategy:
            # Binomial crossover
            trial = target.copy()
            j_rand = np.random.randint(0, dimension)  # Ensure at least one parameter from mutant
            
            for j in range(dimension):
                if np.random.random() <= CR or j == j_rand:
                    trial[j] = mutant[j]
        
        elif "/exp" in self.strategy:
            # Exponential crossover
            trial = target.copy()
            j = np.random.randint(0, dimension)
            L = 0
            
            # Consecutive copying starting from random position
            while np.random.random() <= CR and L < dimension:
                trial[j] = mutant[j]
                j = (j + 1) % dimension
                L += 1
            
            # Ensure at least one parameter is copied
            if L == 0:
                j_rand = np.random.randint(0, dimension)
                trial[j_rand] = mutant[j_rand]
        
        return trial
    
    def _adapt_parameters(self, F: float, CR: float, stagnation_count: int) -> Tuple[float, float]:
        """Adaptive parameter control based on search progress."""
        
        # Simple adaptive scheme: increase exploration if stagnating
        if stagnation_count > 5:
            # Increase mutation strength and crossover rate for more exploration
            new_F = min(1.0, F * 1.1)
            new_CR = min(1.0, CR * 1.05)
        elif stagnation_count == 0:
            # Decrease for more exploitation
            new_F = max(0.1, F * 0.95)
            new_CR = max(0.1, CR * 0.95)
        else:
            new_F = F
            new_CR = CR
        
        return new_F, new_CR 