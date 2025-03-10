import numpy as np
import pandas as pd
from deap import base, creator, tools
import signal
from utils_parallel import ParallelGeneticAlgorithm  # Import custom parallel module
from utils import individual_generator, decode_chromosome, setup_logger, RRMSE
import logging

setup_logger()
logger = logging.getLogger(__name__)

# Configuration parameters
TARGET_WAVELENGTHS = np.array([1404, 1436, 1422])   # rearrange the order
# TARGET_WAVELENGTHS = np.array([90, 360, 25])

CACHE_SIZE = 5000

def main():
    # Add proper signal handling
    def handle_interrupt(signum, frame):
        logger.info("\nReceived interrupt signal, initiating graceful shutdown...")
        ga.boss.shutdown()
        exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    # Problem setup
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    # Initialize GA with custom parallelism
    ga = ParallelGeneticAlgorithm(
        evaluate_func=None,  # Handled by _parallel_evaluator
        individual_generator=individual_generator,
        crossover_op=tools.cxOnePoint,
        mutation_op=tools.mutFlipBit,
        selection_op=tools.selBest,
        target=TARGET_WAVELENGTHS,
        population_size=55,
        crossover_prob=0.85,
        mutation_prob=0.2,
        random_seed=None
    )

    # Run Generic Algorithm
    try:
        population, logbook = ga.run(generations=10, clear_cache=False, verbose=True)

        # extract DEAP logbook
        logbook_data = []
        for record in logbook:
            logbook_data.append(record)
        logbook_df = pd.DataFrame(logbook_data)
        logbook_df.to_csv("GA_logbook.csv", index=False)

        # Choose the Top k Individuals
        k = 10  # Number of the best individuals to select
        best_individuals = tools.selBest(population, k)
        # Output Results
        logger.info(f"Top {k} Individuals:")
        for idx, ind in enumerate(best_individuals):
            h, T, d = decode_chromosome(ind)
            logger.info(f"(h={h}, T={T}, d={d}): Fitness = {ind.fitness.values[0]}, Chromosome = {ind}")

    finally:
        # Retrieve cache data
        data = []
        for hind, result in ga.boss.cache.items():
            # Extract d, T, h (3 genes) from a chromosome (ind)
            h, T, d = decode_chromosome(hind)
            # Fitness value
            if result is not None:
                fitness = RRMSE(result, TARGET_WAVELENGTHS)
                data.append({
                    "d": d,
                    "T": T,
                    "h": h,
                    "ind": hind,  # chromosome
                    "fitness": fitness
                })

        # Save cache data
        df = pd.DataFrame(data)
        df.to_csv('cached_data.csv', index=False)

        ga.boss.shutdown()  # shutting down


if __name__ == "__main__":
    main()
