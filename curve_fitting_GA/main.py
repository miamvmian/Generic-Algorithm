import pandas as pd
from deap import base, creator, tools
import signal
from utils_parallel import ParallelGeneticAlgorithm  # Import custom parallel module
from utils import individual_generator, decode_chromosome, setup_logger, RMSE
import logging
import random

from model import TARGET, BOUNDS, PHYS_BOUNDS
from config import path
from utils import int2phys


setup_logger()
logger = logging.getLogger(__name__)


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
        eval_func=RMSE,  # evaluation function for calculating fitness
        individual_generator=individual_generator,
        crossover_op=tools.cxTwoPoint,
        mutation_op=tools.mutFlipBit,
        selection_op=tools.selTournament,
        # selection_op=tools.selBest,
        target=TARGET,
        population_size=10000,
        crossover_prob=0.9,
        mutation_prob=0.2,
        random_seed=random.random()
    )

    # Run Generic Algorithm
    try:
        population, logbook = ga.run(generations=50, clear_cache=False, verbose=False)

        # extract DEAP logbook
        logbook_data = []
        for record in logbook:
            logbook_data.append(record)
        logbook_df = pd.DataFrame(logbook_data)
        logbook_df.to_csv(path/"GA_logbook.csv", index=False)

        # Choose the Top k Individuals
        k = 10  # Number of the best individuals to select
        best_individuals = tools.selBest(population, k)
        # Output Results
        logger.info(f"Top {k} Individuals:")
        for idx, ind in enumerate(best_individuals):
            int_params = decode_chromosome(ind)
            gamma, omega0, coef = int2phys(int_params, BOUNDS, PHYS_BOUNDS)
            logger.info(f"(gamma ={gamma}, omega0={omega0}, coef={coef}: Fitness = {ind.fitness.values[0]}, Chromosome = {ind}")

    finally:
        # Retrieve cache data
        data = []
        for hind, result in ga.boss.cache.items():
            # Extract physic parameters from a chromosome (ind)
            int_params = decode_chromosome(hind)
            gamma, omega0, coef = int2phys(int_params, BOUNDS, PHYS_BOUNDS)
            # Fitness value
            if result is not None:
                fitness = RMSE(result, TARGET)
                data.append({
                    "gama": gamma,
                    "omega0": omega0,
                    "coef": coef,
                    "ind": hind,  # chromosome
                    "fitness_RMSE": fitness
                })

        # Save cache data
        df = pd.DataFrame(data)
        df.to_csv(path/'cached_data.csv', index=False)

        ga.boss.shutdown()  # shutting down


if __name__ == "__main__":
    main()
