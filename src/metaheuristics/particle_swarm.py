"""
Particle Swarm Optimization implementation for continuous optimization.

This module implements the standard Particle Swarm Optimization algorithm
with inertia weight, cognitive and social components, and velocity clamping.
"""

import time
import numpy as np
from typing import Callable, List, Tuple
from .base import PopulationBasedAlgorithm, OptimizationResult


class ParticleSwarmOptimization(PopulationBasedAlgorithm):
    """
    Particle Swarm Optimization for continuous optimization problems.
    
    Features:
    - Standard PSO velocity-position update equations
    - Inertia weight for exploration-exploitation balance
    - Cognitive and social acceleration coefficients
    - Velocity clamping for boundary handling
    - Global best tracking
    """
    
    def __init__(self, 
                 swarm_size: int = 30,
                 inertia_weight: float = 0.9,
                 inertia_decay: float = 0.99,
                 cognitive_coeff: float = 2.0,
                 social_coeff: float = 2.0,
                 velocity_clamp: bool = True,
                 max_velocity_factor: float = 0.2,
                 **kwargs):
        
        super().__init__("particle_swarm_optimization", population_size=swarm_size, **kwargs)
        
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.inertia_decay = inertia_decay
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.velocity_clamp = velocity_clamp
        self.max_velocity_factor = max_velocity_factor
        
        # Update parameters dict
        self.parameters.update({
            'swarm_size': swarm_size,
            'inertia_weight': inertia_weight,
            'inertia_decay': inertia_decay,
            'cognitive_coeff': cognitive_coeff,
            'social_coeff': social_coeff,
            'velocity_clamp': velocity_clamp,
            'max_velocity_factor': max_velocity_factor
        })
    
    def optimize(self, 
                 objective_function: Callable[[np.ndarray], float],
                 dimension: int,
                 bounds: tuple,
                 max_evaluations: int = 1000,
                 target_fitness: float = 1e-8,
                 **kwargs) -> OptimizationResult:
        """
        Run the Particle Swarm Optimization algorithm.
        """
        start_time = time.time()
        
        # Initialize swarm positions and velocities
        lower, upper = bounds
        positions = self._initialize_population(self.swarm_size, dimension, bounds)
        
        # Calculate velocity bounds
        velocity_range = upper - lower
        max_velocity = self.max_velocity_factor * velocity_range if self.velocity_clamp else np.inf
        velocities = np.random.uniform(-max_velocity, max_velocity, (self.swarm_size, dimension))
        
        # Evaluate initial swarm
        fitness = self._evaluate_population(positions, objective_function)
        evaluations = self.swarm_size
        
        # Initialize personal bests
        personal_best_positions = positions.copy()
        personal_best_fitness = fitness.copy()
        
        # Initialize global best
        global_best_idx = np.argmin(fitness)
        global_best_position = positions[global_best_idx].copy()
        global_best_fitness = fitness[global_best_idx]
        
        # Convergence tracking
        convergence_history = [global_best_fitness]
        diversity_history = [self._calculate_diversity(positions)]
        iteration = 0
        stagnation_count = 0
        current_inertia = self.inertia_weight
        
        # Main PSO loop
        while not self._check_convergence(global_best_fitness, target_fitness, evaluations, max_evaluations):
            iteration += 1
            
            # Update velocities and positions for each particle
            for i in range(self.swarm_size):
                # Random coefficients for cognitive and social components
                r1 = np.random.random(dimension)
                r2 = np.random.random(dimension)
                
                # Velocity update equation
                cognitive_component = self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i])
                social_component = self.social_coeff * r2 * (global_best_position - positions[i])
                
                velocities[i] = (current_inertia * velocities[i] + 
                               cognitive_component + social_component)
                
                # Apply velocity clamping
                if self.velocity_clamp:
                    velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)
                
                # Position update
                positions[i] = positions[i] + velocities[i]
                
                # Apply boundary constraints
                positions[i] = self._clip_to_bounds(positions[i], bounds)
            
            # Evaluate new positions
            fitness = self._evaluate_population(positions, objective_function)
            evaluations += self.swarm_size
            
            # Update personal bests
            better_mask = fitness < personal_best_fitness
            personal_best_positions[better_mask] = positions[better_mask].copy()
            personal_best_fitness[better_mask] = fitness[better_mask]
            
            # Update global best
            current_best_idx = np.argmin(personal_best_fitness)
            current_best_fitness = personal_best_fitness[current_best_idx]
            
            if current_best_fitness < global_best_fitness:
                global_best_fitness = current_best_fitness
                global_best_position = personal_best_positions[current_best_idx].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Update inertia weight (linear decay)
            current_inertia *= self.inertia_decay
            
            # Track convergence
            self._track_convergence(iteration, global_best_fitness, positions, 
                                  convergence_history, diversity_history)
        
        execution_time = time.time() - start_time
        success = global_best_fitness <= target_fitness
        
        return OptimizationResult(
            best_solution=global_best_position,
            best_fitness=global_best_fitness,
            convergence_history=convergence_history,
            total_evaluations=evaluations,
            execution_time=execution_time,
            success=success,
            algorithm_name=self.name,
            parameters=self.get_parameters(),
            problem_name=getattr(objective_function, 'name', 'unknown'),
            problem_dimension=dimension,
            final_population=positions,
            diversity_history=diversity_history,
            stagnation_generations=stagnation_count
        )
    
    def _calculate_diversity(self, positions: np.ndarray) -> float:
        """Calculate swarm diversity (average distance from swarm center)."""
        if len(positions) < 2:
            return 0.0
        
        # Calculate swarm center
        center = np.mean(positions, axis=0)
        
        # Calculate average distance from center
        distances = [np.linalg.norm(particle - center) for particle in positions]
        return np.mean(distances) 