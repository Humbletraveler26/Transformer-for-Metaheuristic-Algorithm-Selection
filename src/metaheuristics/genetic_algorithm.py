"""
Genetic Algorithm implementation for continuous optimization.

This module implements a standard Genetic Algorithm with real-valued encoding,
tournament selection, simulated binary crossover, and polynomial mutation.
"""

import time
import numpy as np
from typing import Callable, List, Tuple
from .base import PopulationBasedAlgorithm, OptimizationResult


class GeneticAlgorithm(PopulationBasedAlgorithm):
    """
    Genetic Algorithm for continuous optimization problems.
    
    Features:
    - Real-valued encoding
    - Tournament selection
    - Simulated Binary Crossover (SBX)
    - Polynomial mutation
    - Elitism
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 3,
                 eta_c: float = 20.0,  # Crossover distribution index
                 eta_m: float = 20.0,  # Mutation distribution index
                 elitism: bool = True,
                 **kwargs):
        
        super().__init__("genetic_algorithm", population_size=population_size, **kwargs)
        
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.elitism = elitism
        
        # Update parameters dict
        self.parameters.update({
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'tournament_size': tournament_size,
            'eta_c': eta_c,
            'eta_m': eta_m,
            'elitism': elitism
        })
    
    def optimize(self, 
                 objective_function: Callable[[np.ndarray], float],
                 dimension: int,
                 bounds: tuple,
                 max_evaluations: int = 1000,
                 target_fitness: float = 1e-8,
                 **kwargs) -> OptimizationResult:
        """
        Run the Genetic Algorithm optimization.
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
        
        # Main evolution loop
        while not self._check_convergence(best_fitness, target_fitness, evaluations, max_evaluations):
            generation += 1
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individual
            if self.elitism:
                new_population.append(best_solution.copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._sbx_crossover(parent1, parent2, bounds)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self._polynomial_mutation(child1, bounds)
                child2 = self._polynomial_mutation(child2, bounds)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            new_population = new_population[:self.population_size]
            population = np.array(new_population)
            
            # Evaluate new population
            fitness = self._evaluate_population(population, objective_function)
            evaluations += self.population_size
            
            # Update best solution
            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]
            
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = population[current_best_idx].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
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
    
    def _tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Tournament selection operator."""
        tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _sbx_crossover(self, parent1: np.ndarray, parent2: np.ndarray, bounds: tuple) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)."""
        lower, upper = bounds
        dimension = len(parent1)
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(dimension):
            if np.random.random() <= 0.5:  # Crossover probability per variable
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    # Calculate beta
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]
                    
                    # Calculate beta bounds
                    beta_l = 1.0 + (2.0 * (y1 - lower) / (y2 - y1))
                    beta_u = 1.0 + (2.0 * (upper - y2) / (y2 - y1))
                    
                    # Generate random number
                    rand = np.random.random()
                    
                    if rand <= 1.0 / (2.0 * beta_l):
                        beta = (2.0 * rand * beta_l) ** (1.0 / (self.eta_c + 1.0))
                    else:
                        beta = (1.0 / (2.0 - 2.0 * rand * beta_l)) ** (1.0 / (self.eta_c + 1.0))
                    
                    if rand > 1.0 / (2.0 * beta_u):
                        beta = (2.0 * rand * beta_u) ** (1.0 / (self.eta_c + 1.0))
                    else:
                        beta = (1.0 / (2.0 - 2.0 * rand * beta_u)) ** (1.0 / (self.eta_c + 1.0))
                    
                    # Create offspring
                    child1[i] = 0.5 * ((y1 + y2) - beta * abs(y2 - y1))
                    child2[i] = 0.5 * ((y1 + y2) + beta * abs(y2 - y1))
        
        # Ensure bounds
        child1 = self._clip_to_bounds(child1, bounds)
        child2 = self._clip_to_bounds(child2, bounds)
        
        return child1, child2
    
    def _polynomial_mutation(self, individual: np.ndarray, bounds: tuple) -> np.ndarray:
        """Polynomial mutation operator."""
        lower, upper = bounds
        dimension = len(individual)
        mutated = individual.copy()
        
        for i in range(dimension):
            if np.random.random() < self.mutation_rate:
                y = mutated[i]
                delta_l = (y - lower) / (upper - lower)
                delta_u = (upper - y) / (upper - lower)
                
                rand = np.random.random()
                mut_pow = 1.0 / (self.eta_m + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta_l
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (self.eta_m + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta_u
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (self.eta_m + 1.0))
                    delta_q = 1.0 - val ** mut_pow
                
                mutated[i] = y + delta_q * (upper - lower)
        
        return self._clip_to_bounds(mutated, bounds) 