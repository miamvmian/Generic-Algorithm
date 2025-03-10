import logging
import logging.handlers
import time
import random
from queue import Empty

import numpy as np
from multiprocessing import Process, Queue, Value, Lock
from collections import defaultdict
import os
import signal
from utils import decode_chromosome, param_constraints, mask_func, fem_simulation, setup_logger, RRMSE
from deap import base, creator, tools, algorithms

setup_logger()
logger = logging.getLogger(__name__)

MAX_WORKERS = 32

class SimulationWorker(Process):
    def __init__(self, task_queue, result_queue, terminate_flag):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.terminate_flag = terminate_flag  # 共享的终止标志
        self.worker_log = logging.getLogger(f"Worker.{self.name}")

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.worker_log.info(f"Worker started (PID: {os.getpid()})")

        while not self.terminate_flag.value:  # 检查共享的终止标志
            task = self.task_queue.get()
            if task is None:
                self.worker_log.info(f"Get a NONE task, Worker {self.name} terminated.")
                break
            ind, hind, task_id = task

            try:
                self.worker_log.info(f"Task-{task_id}: processing {ind}")

                # decode chromosome into parameters of real physic model
                decode_params = decode_chromosome(ind)

                # To check if the params meets the constraint requirement
                start_time = time.time()
                if not param_constraints(decode_params):
                    # if the parameters out of the constraints range
                    self.worker_log.info(f"The individual {ind} doesn't meet the constraints.")
                    result = None
                else:
                    # calculate Tamm plasmon resonance wavelength
                    result = fem_simulation(decode_params)
                    if result is None:
                        self.worker_log.info(
                            f"Individual {ind} results None.")
                    # result = mask_func(decode_params)
                duration = time.time() - start_time

                self.result_queue.put((task_id, hind, result))
                self.worker_log.info(
                    f"Task-{task_id}: Completed in {duration:.2f}s "
                    f"| Params: {ind} | Result: {result}"
                )

            except Exception as e:
                self.worker_log.error(
                    f"Task-{task_id}: failed: {str(e)}",
                    exc_info=True
                )
                self.result_queue.put((task_id, hind, None))

        self.worker_log.info(f"Worker {self.name} shutting down.")


class OptimizationBoss:
    def __init__(self):
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.workers = []
        self.cache = defaultdict(lambda: None)
        self.pending_tasks = {}
        self.task_counter = 0
        self.log = logger

        # 创建共享的终止标志和锁
        self.terminate_flag = Value('b', False)  # 'b' 表示布尔类型
        self.flag_lock = Lock()
        self._shutdown_initiated = False

        self.log.info("Initializing optimization boss")
        self.log.info(f"Starting {MAX_WORKERS} worker processes")

        for i in range(MAX_WORKERS):
            worker = SimulationWorker(
                self.task_queue,
                self.result_queue,
                self.terminate_flag  # 传递共享标志
            )
            worker.start()
            self.workers.append(worker)

    def submit_task(self, params):
        ind, hind = params

        self.task_counter += 1
        self.task_queue.put((ind, hind, self.task_counter))
        self.pending_tasks[self.task_counter] = hind
        self.log.debug(f"Task-{self.task_counter} submitted {params}")
        return self.task_counter


    def get_results(self, timeout=None):
        start_time = time.time()
        results = {}
        self.log.info(f"Collecting up to {len(self.pending_tasks)} results")

        while len(self.pending_tasks)>0:
            if timeout and (time.time() - start_time) > timeout:
                self.log.warning(f"Timeout reached ({timeout}s)")
                break

            try:
                # wait maximum 1s. If get no result, raise Empty exception
                task_id, hind, result = self.result_queue.get(timeout=1)
                del self.pending_tasks[task_id]
                self.log.info(f"{len(self.pending_tasks)} pending tasks remaining")

                results[task_id] = result

            except Empty:
                if self._shutdown_initiated:
                    self.log.warning("Shutdown initiated, breaking result collection")
                    break
                self.log.debug("Result queue empty, retrying...")

            except Exception as e:
                self.log.error(f"Unexpected error during result collection: {str(e)}", exc_info=True)
                break

        return results

    def shutdown(self):
        if self._shutdown_initiated:
            return
        self._shutdown_initiated = True
        self.log.info("Initiating shutdown sequence")

        # 安全更新终止标志
        with self.flag_lock:
            self.terminate_flag.value = True
        # 发送终止信号给所有worker
        for _ in self.workers:
            self.task_queue.put(None)
        # 等待worker终止
        for worker in self.workers:
            worker.join()

        self.log.info("All workers terminated")

    def clear_cache(self):
        self.cache.clear()


# Genetic Algorithm parallelism
class ParallelGeneticAlgorithm:
    def __init__(self, evaluate_func, individual_generator, crossover_op, mutation_op, selection_op, target=None,
                 crossover_prob=0.85, mutation_prob=0.2, population_size=50, random_seed=None):
        """
        Initialize GA with custom parallel evaluation using OptimizationBoss.
        """
        self.evaluate = evaluate_func
        self.individual_generator = individual_generator
        self.crossover_op = crossover_op
        self.mutation_op = mutation_op
        self.selection_op = selection_op
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population_size = population_size

        self.random_seed = random_seed
        self._TARGET = target

        # Initialize DEAP toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.individual_generator)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self.crossover_op)
        self.toolbox.register("mutate", self.mutation_op, indpb=0.2)
        self.toolbox.register("select", self.selection_op)

        # Register parallel evaluator
        self.toolbox.register("evaluate", self._parallel_evaluator)

        if self.random_seed is not None:
            random.seed(self.random_seed)

        self.boss = OptimizationBoss()  # Single boss instance for lifetime
        self._hashable_individual = lambda ind: tuple(ind)  # Convert to hashable


    def _parallel_evaluator(self, individuals, timeout=False):
        """Evaluation with persistent cache across generations"""
        tasks = []
        hashable_inds = [self._hashable_individual(ind) for ind in individuals]

        # Submit only uncached individuals
        for ind, hind in zip(individuals, hashable_inds):
            if hind not in self.boss.cache:
                task_id = self.boss.submit_task((ind,hind))
                tasks.append((ind, hind, task_id))

        # Get results for submitted tasks
        if tasks:
            results = self.boss.get_results(timeout=timeout)
            for ind, hind, task_id in tasks:
                if results and task_id in results:
                    self.boss.cache[hind] = results[task_id]

        # Retrieve fitness from cache
        return [self._get_fitness(hind) for hind in hashable_inds]

    def _get_fitness(self, hashable_ind, penalty=10):
        """Convert cached value to DEAP-compatible fitness"""
        result = self.boss.cache[hashable_ind]
        if result is None:
            return (penalty,)  # Penalize invalid solutions
        # fitness: relative root mean square error.
        else:
            fitness = RRMSE(self._TARGET, result)
            return (fitness,)

    def run(self, generations=50, clear_cache=False, stats=None, verbose=False):
        """Run GA with optional cache clearing，execute GA with custom parallelism"""
        if verbose:
            logger.info("=============================================================")
            logger.setLevel(logging.DEBUG)
            logger.info("=============================================================")

        try:
            if clear_cache:
                self.boss.clear_cache()

            pop = self.toolbox.population(n=self.population_size)

            # Configure statistics
            if stats is None:
                stats = tools.Statistics(lambda ind: ind.fitness.values[0])
                stats.register("avg", np.mean)
                stats.register("std", np.std)
                stats.register("min", np.min)
                stats.register("max", np.max)

            logbook = tools.Logbook()
            logbook.header = ["gen", "nevals"] + stats.fields # Define column order

            # Evolutionary algorithm loop
            for gen in range(generations):
                # Variation
                offspring = algorithms.varAnd(pop, self.toolbox,
                                              self.crossover_prob, self.mutation_prob)

                # Evaluate new individuals
                fitnesses = self.toolbox.evaluate(offspring)
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit

                # Selection
                pop = self.toolbox.select(pop + offspring, self.population_size)

                # Compile statistics and record them in the logbook
                record = stats.compile(pop)
                logbook.record(
                    gen=gen,  # Current generation
                    nevals=len(pop),  # Number of evaluations (adjust as needed)
                    **record  # Unpack stats (avg, min, max, etc.)
                )

                avg_fitness = logbook.select("avg")
                std_fitness = logbook.select("std")
                min_fitness = logbook.select("min")
                max_fitness = logbook.select("max")

                logger.info("=============================================================")
                logger.info(f"{'gen':<5} {'nevals':<5} {'avg':<10} {'std':<10} {'min':<10} {'max':<10}")  # Header
                logger.info(f"{gen:<5} {len(pop):<5} {avg_fitness:<10.2e} {std_fitness:<10.2e} {min_fitness:<10.2e} {max_fitness:<10.2e}")
                logger.info("=============================================================")

                if (avg_fitness-min_fitness)<min_fitness/10:
                    logger.info(f"The evolution terminated at {gen} generation.")
                    break

            return pop, logbook

        except KeyboardInterrupt:
            self.boss.log.warning("Keyboard interrupt received, initiating shutdown")
            self.boss.shutdown()
            raise
        finally:
            self.boss.shutdown()

    def __del__(self):
        if hasattr(self, 'boss') and not self.boss._shutdown_initiated:
            self.boss.shutdown()
