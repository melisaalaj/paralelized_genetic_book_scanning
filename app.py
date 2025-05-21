import sys

from models import Parser
from models.initial_solution import InitialSolution
import os
import multiprocessing as mp
from models.genetic_solver import GeneticSolver
from models.meta_genetic_optimizer import MetaGeneticOptimizer
from validator.multiple_validator import validate_all_solutions


# directory = os.listdir('input')
#
# # files = [
# #     # 'b_read_on.txt',
# #     # 'c_incunabula.txt',
# #     #  'd_tough_choices.txt',
# #      'e_so_many_books.txt',
# #      # 'f_libraries_of_the_world.txt',
# #      # 'B5000_L90_D21.txt',
# #      # 'B50000_L400_D28.txt',
# #      # 'B90000_L850_D21.txt',
# #      # 'B95000_L2000_D28.txt'
# # ]
#
# files = ['c_incunabula.txt']
#
#
# for file in directory:
#     if files.__contains__(file):
#         print(f'Computing ./input/{file}')
#         parser = Parser(f'./input/{file}')
#         instance = parser.parse()
#         initial_solution = InitialSolution.generate_initial_solution(instance)
#         # geneticSolver = GeneticAlgorithmSolver(instance, initial_solution)
#         geneticSolver = GeneticSolver(
#             initial_solution=initial_solution,  # your initial Solution instance
#             instance=instance
#         )
#         solution = geneticSolver.solve()
#
#         # meta_opt = MetaGeneticOptimizer(GeneticSolver, instance, initial_solution)
#         # best_hyper = meta_opt.optimize()
#         # print("Best hyperparameters:", best_hyper)
#         # # run full solver with best hyper
#         # solver = GeneticSolver(initial_solution, instance)
#         # solver.mutation_prob = best_hyper['mutation_prob']
#         # solver.crossover_rate = best_hyper['crossover_rate']
#         # solver.immigrant_frac = best_hyper['immigrant_frac']
#         # solution = solver.solve()
#
#         solution.export(f'./output/{file}')
#         print(f"{solution.fitness_score:,}", file)


def run_instances(output_dir='output', n_processes=None):
    print(output_dir)
    directory = os.listdir('input')
    results = []
    os.makedirs(output_dir, exist_ok=True)

    # Determine number of processes for parallel generation
    if n_processes is None:
        # Use CPU count by default
        n_processes = mp.cpu_count()
    
    for file in directory:
        if file.endswith('.txt'):
            print(f'Computing ./input/{file}')
            parser = Parser(f'./input/{file}')
            instance = parser.parse()
            initial_solution = InitialSolution.generate_initial_solution(instance)
            
            # Use parallel genetic solver with specified number of processes
            genetic_solver = GeneticSolver(
                initial_solution=initial_solution, 
                instance=instance,
                n_processes=n_processes
            )
            
            solution = genetic_solver.solve()
            score = solution.fitness_score
            results.append((file, score))
            print(f"Final score for {file}: {score:,}")
            output_file = os.path.join(output_dir, file)
            solution.export(output_file)
            print("----------------------")

    print("\nValidating all solutions...")
    validate_all_solutions(input_dir='input', output_dir=output_dir)

    # Print summary of all instances
    print("\nSummary of all instances:")
    print("-" * 50)
    print(f"{'Instance':<20} {'Score':>15}")
    print("-" * 50)
    for file, score in results:
        print(f"{file:<20} {score:>15,}")
    print("-" * 50)

    # Write summary to a text file
    summary_file = os.path.join(output_dir, 'summary_results.txt')
    with open(summary_file, 'w') as f:
        f.write("Summary of all instances:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Instance':<20} {'Score':>15}\n")
        f.write("-" * 50 + "\n")
        for file, score in results:
            f.write(f"{file:<20} {score:>15,}\n")
        f.write("-" * 50 + "\n")
    print(f"\nSummary has been written to: {summary_file}")


def main():
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        # Use the second argument as the number of processes for parallel generation
        n_processes = int(sys.argv[2])
        print(f"Using {n_processes} processes for parallel generation")
    else:
        # Use all CPU cores by default
        n_processes = None
    
    if len(sys.argv) > 1:
        subdir = sys.argv[1]
        run_instances(f"./output/{subdir}", n_processes)
    else:
        print("No argument provided. Saving outputs to ./output")
        run_instances(n_processes=n_processes)


if __name__ == "__main__":
    main()