"""
Simulated Annealing implementation for continuous optimization.

This module implements the Simulated Annealing algorithm with various
cooling schedules, neighborhood generation strategies, and acceptance criteria.
"""

import time
import numpy as np
import math
from typing import Callable, List, Tuple
from .base import SingleSolutionAlgorithm, OptimizationResult


class SimulatedAnnealing(SingleSolutionAlgorithm):
    """
    Simulated Annealing for continuous optimization problems.
    
    Features:
    - Multiple cooling schedules (linear, exponential, logarithmic)
    - Adaptive step size control
    - Gaussian neighborhood generation
    - Metropolis acceptance criterion
    """
    
    def __init__(self, 
                 initial_temp: float = 100.0,
                 final_temp: float = 0.01,
                 cooling_schedule: str = "exponential",
                 cooling_rate: float = 0.95,
                 steps_per_temp: int = 10,
                 step_size: float = 0.1,
                 adaptive_step: bool = True,
                 **kwargs):
        
        super().__init__("simulated_annealing", **kwargs)
        
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_schedule = cooling_schedule
        self.cooling_rate = cooling_rate
        self.steps_per_temp = steps_per_temp
        self.step_size = step_size
        self.adaptive_step = adaptive_step
        
        # Validate cooling schedule
        valid_schedules = ["linear", "exponential", "logarithmic", "inverse"]
        if cooling_schedule not in valid_schedules:
            raise ValueError(f"Unknown cooling schedule: {cooling_schedule}. Valid schedules: {valid_schedules}")
        
        # Update parameters dict
        self.parameters.update({
            'initial_temp': initial_temp,
            'final_temp': final_temp,
            'cooling_schedule': cooling_schedule,
            'cooling_rate': cooling_rate,
            'steps_per_temp': steps_per_temp,
            'step_size': step_size,
            'adaptive_step': adaptive_step
        })
    
    def optimize(self, 
                 objective_function: Callable[[np.ndarray], float],
                 dimension: int,
                 bounds: tuple,
                 max_evaluations: int = 1000,
                 target_fitness: float = 1e-8,
                 **kwargs) -> OptimizationResult:
        """
        Run the Simulated Annealing algorithm.
        """
        start_time = time.time()
        
        # Initialize solution
        lower, upper = bounds
        current_solution = np.random.uniform(lower, upper, dimension)
        current_fitness = objective_function(current_solution)
        evaluations = 1
        
        # Best solution tracking
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        # Temperature and step size tracking
        current_temp = self.initial_temp
        current_step_size = self.step_size
        
        # Convergence tracking
        convergence_history = [best_fitness]
        temperature_history = [current_temp]
        acceptance_history = []
        step_count = 0
        temp_iterations = 0
        stagnation_count = 0
        
        # Calculate total temperature steps for linear cooling
        if self.cooling_schedule == "linear":
            max_temp_steps = max_evaluations // self.steps_per_temp
        
        # Main annealing loop
        while (current_temp > self.final_temp and 
               evaluations < max_evaluations and 
               not self._check_convergence(best_fitness, target_fitness, evaluations, max_evaluations)):
            
            temp_iterations += 1
            accepted_at_temp = 0
            
            # Perform steps at current temperature
            for _ in range(self.steps_per_temp):
                if evaluations >= max_evaluations:
                    break
                
                step_count += 1
                
                # Generate neighbor solution
                neighbor = self._generate_neighbor(current_solution, bounds, current_step_size)
                neighbor_fitness = objective_function(neighbor)
                evaluations += 1
                
                # Calculate acceptance probability
                delta = neighbor_fitness - current_fitness
                
                if delta < 0:
                    # Better solution - always accept
                    accept = True
                else:
                    # Worse solution - accept with probability
                    prob = math.exp(-delta / current_temp) if current_temp > 0 else 0
                    accept = np.random.random() < prob
                
                # Update current solution if accepted
                if accept:
                    current_solution = neighbor
                    current_fitness = neighbor_fitness
                    accepted_at_temp += 1
                    
                    # Update best solution if improved
                    if neighbor_fitness < best_fitness:
                        best_fitness = neighbor_fitness
                        best_solution = neighbor.copy()
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                
                acceptance_history.append(accept)
                
                # Track convergence every few steps
                if step_count % 5 == 0:
                    convergence_history.append(best_fitness)
            
            # Adaptive step size control
            if self.adaptive_step:
                acceptance_rate = accepted_at_temp / self.steps_per_temp
                current_step_size = self._adapt_step_size(current_step_size, acceptance_rate)
            
            # Cool down temperature
            current_temp = self._update_temperature(current_temp, temp_iterations, max_temp_steps if self.cooling_schedule == "linear" else None)
            temperature_history.append(current_temp)
        
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
            final_population=np.array([best_solution]),  # Single solution algorithm
            diversity_history=None,  # Not applicable for single solution
            stagnation_generations=stagnation_count
        )
    
    def _generate_neighbor(self, current_solution: np.ndarray, bounds: tuple, step_size: float) -> np.ndarray:
        """Generate a neighbor solution using Gaussian perturbation."""
        dimension = len(current_solution)
        
        # Generate Gaussian perturbation
        perturbation = np.random.normal(0, step_size, dimension)
        neighbor = current_solution + perturbation
        
        # Apply boundary constraints
        return self._clip_to_bounds(neighbor, bounds)
    
    def _update_temperature(self, current_temp: float, iteration: int, max_iterations: int = None) -> float:
        """Update temperature according to cooling schedule."""
        
        if self.cooling_schedule == "exponential":
            # T(k) = T0 * α^k
            return current_temp * self.cooling_rate
        
        elif self.cooling_schedule == "linear":
            # T(k) = T0 - k * (T0 - Tf) / max_iter
            if max_iterations is None:
                max_iterations = 100  # Default fallback
            return max(self.final_temp, 
                      self.initial_temp - iteration * (self.initial_temp - self.final_temp) / max_iterations)
        
        elif self.cooling_schedule == "logarithmic":
            # T(k) = T0 / log(1 + k)
            return self.initial_temp / math.log(1 + iteration)
        
        elif self.cooling_schedule == "inverse":
            # T(k) = T0 / (1 + α * k)
            return self.initial_temp / (1 + self.cooling_rate * iteration)
        
        else:
            return current_temp * self.cooling_rate  # Default to exponential
    
    def _adapt_step_size(self, current_step: float, acceptance_rate: float) -> float:
        """Adapt step size based on acceptance rate."""
        
        # Target acceptance rate around 0.4-0.6 (commonly used in SA)
        target_rate = 0.5
        
        if acceptance_rate > 0.6:
            # Too many acceptances - increase step size for more exploration
            new_step = min(1.0, current_step * 1.1)
        elif acceptance_rate < 0.3:
            # Too few acceptances - decrease step size for more local search
            new_step = max(0.001, current_step * 0.9)
        else:
            # Good acceptance rate - keep current step size
            new_step = current_step
        
        return new_step 