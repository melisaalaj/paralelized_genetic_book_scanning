import random
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List

from models.selection_strategies import SelectionStrategies
from models.tweaks import Tweaks
from models.solution import Solution
from models.instance_data import InstanceData


class GeneticSolver:
    def __init__(self,
                 initial_solution: Solution,
                 instance: InstanceData,
                 population_size=100,
                 generations=500,
                 mutation_prob=0.39,
                 crossover_rate=0.33,
                 immigrant_frac=0.06,
                 steady_state_ratio=0.25,
                 time_limit_sec=10 * 60,
                 tweak_steps=5,
                 num_parallel_populations=4
                 ):
        self.initial_solution = initial_solution
        self.instance = instance
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.mutation_prob = mutation_prob
        self.crossover_rate = crossover_rate
        self.immigrant_frac = immigrant_frac
        self.tweak_steps = tweak_steps
        self.time_limit_sec = time_limit_sec
        self.steady_state_ratio = steady_state_ratio
        self.steady_gen_start = int(self.generations * (1 - steady_state_ratio))
        self.steady_time_start = self.time_limit_sec * (1 - steady_state_ratio)
        self.num_parallel_populations = num_parallel_populations

    def solve(self):
        if self.num_parallel_populations <= 1:
            return self._solve_sequential()
        else:
            return self._solve_parallel()

    def _solve_sequential(self):
        # Initialize population with slight variations of initial solution
        population = self.initialize_population(self.initial_solution)

        start_time = time.time()

        best_fitness = None
        plateau_counter = 0
        base_immigrant_frac = self.immigrant_frac

        for generation in range(self.generations):
            elapsed = time.time() - start_time
            if elapsed >= self.time_limit_sec:
                print(f"Stopping at gen {generation} due to time limit ({elapsed:.1f}s)")
                break

            # Evaluate population
            population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
            best_solution = population[0]
            print(f"Gen {generation}: Best fitness = {best_solution.fitness_score}")

            # Plateau tracking
            if best_fitness is None or best_solution.fitness_score > best_fitness:
                best_fitness = best_solution.fitness_score
                plateau_counter = 0
                self.immigrant_frac = base_immigrant_frac  # Reset if improvement
            else:
                plateau_counter += 1
                # Increase immigrant_frac after N stagnant generations (e.g., 10)
                if plateau_counter > 5:
                    self.immigrant_frac = min(1.0, self.immigrant_frac * 1.5)

            # decide whether to use generational or steady-state:
            use_steady = (
                    generation >= self.steady_gen_start
                    or elapsed >= self.steady_time_start
            )
            if not use_steady:
                new_population = self.create_offspring_generative(population)
            else:
                new_population = self.create_offspring_steady_state(population)

            num_immigrants = int(self.immigrant_frac * self.population_size)
            if num_immigrants > 0:
                immigrants = self.initialize_population(self.initial_solution)[:num_immigrants]

                # Remove worst individuals to make room for immigrants
                new_population = sorted(new_population, key=lambda x: x.fitness_score, reverse=True)
                new_population = new_population[:-num_immigrants] + immigrants

            # Ensure best solution is not lost
            if best_solution.fitness_score > min(new_population, key=lambda x: x.fitness_score).fitness_score:
                new_population[-1] = best_solution

            # Update population
            population = new_population[:self.population_size]

        return max(population, key=lambda x: x.fitness_score)

    def _solve_parallel(self):
        """Evolve multiple populations in parallel and combine results"""
        # Number of processes is based on available cores
        num_processes = min(self.num_parallel_populations, mp.cpu_count())
        
        print(f"Running {num_processes} parallel populations on {mp.cpu_count()} available CPU cores")
        print(f"Each population will run for up to {self.time_limit_sec/60:.1f} minutes")
        
        # Increase generations to ensure we can run for the full time limit
        increased_generations = self.generations * 2
        
        # Prepare arguments for each process
        process_args = []
        for i in range(num_processes):
            # Vary some parameters slightly to ensure diversity
            mutation_prob = self.mutation_prob * random.uniform(0.9, 1.1)
            crossover_rate = self.crossover_rate * random.uniform(0.9, 1.1)
            
            process_args.append((
                self.initial_solution,
                self.instance,
                self.population_size,
                increased_generations,  # Use increased generations
                mutation_prob,
                crossover_rate,
                self.immigrant_frac,
                self.steady_state_ratio,
                self.time_limit_sec,  # Use full time limit for each population
                self.tweak_steps,
                i  # process id for logging
            ))
        
        # Run parallel evolution
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(self._evolve_population, process_args))
        
        # Find the best solution from all populations
        best_solution = max(results, key=lambda x: x.fitness_score)
        print(f"Best overall solution from parallel populations: {best_solution.fitness_score}")
        
        return best_solution
    
    @staticmethod
    def _evolve_population(args):
        """Static method to evolve a single population (used by parallel solver)"""
        (initial_solution, instance, population_size, generations, 
         mutation_prob, crossover_rate, immigrant_frac, steady_state_ratio, 
         time_limit_sec, tweak_steps, process_id) = args
        
        # Create a new solver with these parameters
        solver = GeneticSolver(
            initial_solution=initial_solution,
            instance=instance,
            population_size=population_size,
            generations=generations,
            mutation_prob=mutation_prob,
            crossover_rate=crossover_rate,
            immigrant_frac=immigrant_frac,
            steady_state_ratio=steady_state_ratio,
            time_limit_sec=time_limit_sec,
            tweak_steps=tweak_steps,
            num_parallel_populations=1  # No further parallelization
        )
        
        # Initialize population
        population = solver.initialize_population(initial_solution)
        
        start_time = time.time()
        best_fitness = None
        plateau_counter = 0
        base_immigrant_frac = immigrant_frac
        current_best_solution = None
        
        print(f"Population {process_id} starting evolution, time limit: {time_limit_sec/60:.1f} minutes")
        
        for generation in range(generations):
            elapsed = time.time() - start_time
            if elapsed >= time_limit_sec:
                print(f"Population {process_id}: Stopping at gen {generation} due to time limit ({elapsed:.1f}s)")
                break
                
            # Evaluate population
            population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
            best_solution = population[0]
            current_best_solution = best_solution
            
            if generation % 10 == 0:  # Reduce logging frequency
                print(f"Population {process_id}, Gen {generation}: Best fitness = {best_solution.fitness_score}, Time: {elapsed:.1f}s")
                
            # Plateau tracking
            if best_fitness is None or best_solution.fitness_score > best_fitness:
                best_fitness = best_solution.fitness_score
                plateau_counter = 0
                solver.immigrant_frac = base_immigrant_frac
            else:
                plateau_counter += 1
                if plateau_counter > 5:
                    solver.immigrant_frac = min(1.0, solver.immigrant_frac * 1.5)
                    
            # Decide evolution strategy
            use_steady = (
                generation >= solver.steady_gen_start or 
                elapsed >= solver.steady_time_start
            )
            
            if not use_steady:
                new_population = solver.create_offspring_generative(population)
            else:
                new_population = solver.create_offspring_steady_state(population)
                
            # Add immigrants
            num_immigrants = int(solver.immigrant_frac * population_size)
            if num_immigrants > 0:
                immigrants = solver.initialize_population(initial_solution)[:num_immigrants]
                new_population = sorted(new_population, key=lambda x: x.fitness_score, reverse=True)
                new_population = new_population[:-num_immigrants] + immigrants
                
            # Elitism
            if best_solution.fitness_score > min(new_population, key=lambda x: x.fitness_score).fitness_score:
                new_population[-1] = best_solution
                
            # Update population
            population = new_population[:population_size]
        
        elapsed = time.time() - start_time
        print(f"Population {process_id} finished: Best fitness = {current_best_solution.fitness_score}, Actual runtime: {elapsed:.1f}s")
            
        return current_best_solution or max(population, key=lambda x: x.fitness_score)

    def create_offspring_generative(self, population):
        new_population = []
        while len(new_population) < self.population_size:
            selection_method = SelectionStrategies.choose_selection_method()
            parent1 = selection_method(population)
            parent2 = selection_method(population)

            offspring1, offspring2 = self.crossover(parent1, parent2)

            if random.random() < self.mutation_prob:
                offspring1 = Tweaks.tweak_with_iterations(offspring1, self.instance, iterations=self.tweak_steps)
            if random.random() < self.mutation_prob:
                offspring2 = Tweaks.tweak_with_iterations(offspring2, self.instance, iterations=self.tweak_steps)

            new_population.extend([offspring1.shallow_copy(), offspring2.shallow_copy()])

        return new_population

    def create_offspring_steady_state(self, population):
        new_population = [population[0]]

        while len(new_population) < self.population_size:
            # Selection
            selection_method = SelectionStrategies.choose_selection_method()
            parent1 = selection_method(population)
            parent2 = selection_method(population)

            offspring1, offspring2 = self.crossover(parent1, parent2)

            if random.random() < self.mutation_prob:
                offspring1 = Tweaks.tweak_with_iterations(offspring1, self.instance, iterations=self.tweak_steps)
            if random.random() < self.mutation_prob:
                offspring2 = Tweaks.tweak_with_iterations(offspring2, self.instance, iterations=self.tweak_steps)

            # Combine the population with offspring and select the best ones
            combined = population + [offspring1.shallow_copy(), offspring2.shallow_copy()]
            new_population = sorted(combined, key=lambda x: x.fitness_score, reverse=True)[:self.population_size]

        return new_population

    def initialize_population(self, initial_solution, tweak_ratio: float = 0.5):
        population = [initial_solution.shallow_copy()]

        num_tweaked = int(self.population_size * tweak_ratio)
        num_clones = self.population_size - num_tweaked - 1

        # Add tweaked solutions
        for _ in range(num_tweaked):
            tweaked = Tweaks.tweak_with_iterations(
                initial_solution,
                self.instance,
                iterations=random.randint(1, self.tweak_steps)
            )
            population.append(tweaked.shallow_copy())

        # Add direct shallow clones
        for _ in range(num_clones):
            population.append(initial_solution.shallow_copy())

        return population


    def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:

        def create_offspring(p1_signed, p2_signed):
            size = len(p1_signed)

            # Convert to sets for O(1) lookups
            p1_set = set(p1_signed)
            p2_set = set(p2_signed)

            # Create quick lookup for used libraries
            used_libs = set()

            # Initialize offspring with None
            offspring_signed = [None] * size

            # 1. Select random positions from parent1 (faster sampling)
            copy_indices = random.sample(range(size), k=size // 2)
            for idx in copy_indices:
                lib = p1_signed[idx]
                offspring_signed[idx] = lib
                used_libs.add(lib)

            # 2. Prepare parent2 libraries not yet used
            available_p2 = [lib for lib in p2_signed if lib not in used_libs]
            p2_ptr = 0

            # 3. Fill remaining positions
            remaining_indices = [i for i, lib in enumerate(offspring_signed) if lib is None]

            for idx in remaining_indices:
                if p2_ptr < len(available_p2):
                    offspring_signed[idx] = available_p2[p2_ptr]
                    p2_ptr += 1
                else:
                    # Fallback to unused libraries from parent1
                    remaining_p1 = [lib for lib in p1_signed if lib not in used_libs]
                    if remaining_p1:
                        offspring_signed[idx] = random.choice(remaining_p1)
                        used_libs.add(offspring_signed[idx])
                    else:
                        # If all libraries are used (shouldn't happen with valid parents)
                        unused = list(p1_set - used_libs)
                        if unused:
                            offspring_signed[idx] = random.choice(unused)
                        else:
                            raise ValueError("Crossover failed - no available libraries")

            return offspring_signed

        # Create base signed orders
        try:
            offspring1_signed = create_offspring(parent1.signed_libraries, parent2.signed_libraries)
            offspring2_signed = create_offspring(parent2.signed_libraries, parent1.signed_libraries)

            # Create complete solutions
            def build_solution(signed_libs):
                scanned_books = set()
                scanned_per_lib = {}
                used_libs = []

                current_day = 0
                for lib in signed_libs:
                    lib_data = self.instance.libs[lib]
                    if current_day + lib_data.signup_days > self.instance.num_days:
                        continue

                    current_day += lib_data.signup_days
                    remaining_days = self.instance.num_days - current_day
                    max_books = remaining_days * lib_data.books_per_day

                    available_books = [b.id for b in lib_data.books if b.id not in scanned_books]
                    available_books.sort(key=lambda x: self.instance.scores[x], reverse=True)
                    selected = available_books[:max_books]

                    if selected:
                        scanned_books.update(selected)
                        scanned_per_lib[lib] = selected
                        used_libs.append(lib)

                built = Solution(
                    signed_libs=used_libs,
                    unsigned_libs=list(set(range(self.instance.num_libs)) - set(used_libs)),
                    scanned_books_per_library=scanned_per_lib,
                    scanned_books=scanned_books
                )

                built.calculate_fitness_score(self.instance.scores)
                return built

            return (build_solution(offspring1_signed),
                    build_solution(offspring2_signed))


        except ValueError as e:
            # Fallback to parents if crossover fails
            print(f"Crossover failed: {e}, returning parents")
            return parent1, parent2