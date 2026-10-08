"""Genetic algorithm individuals, operators, and population initialization."""

import random
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class GAConfig:
    """Genetic algorithm parameters."""

    population_size: int = 100
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    max_num_mutations: int = 15
    elite_size: int = 1
    tournament_size: int = 2
    penalty_factor: float = 1000.0
    n_jobs: int = 1


class Individual:
    """Represents an individual in the genetic algorithm population."""

    def __init__(self, chromosome: List[int], fitness: float = 0.0):
        self.chromosome = chromosome
        self.fitness = fitness
        self.is_evaluated = False
        self.is_feasible = True
        self.constraint_violations = 0
        self.age = 0
        self.optim_results = {}

    def __len__(self):
        return len(self.chromosome)

    def __str__(self):
        return (
            f"Individual(size={len(self.chromosome)}, fitness={self.fitness:.3f}, is_feasible={self.is_feasible}, "
            f"is_evaluated={self.is_evaluated})"
        )

    def __repr__(self):
        return self.__str__()

    def copy(self, is_evaluated: bool = False):
        """Copy the chromosome and feasibility state, optionally retaining evaluation."""
        new_individual = Individual(self.chromosome.copy(), self.fitness)
        new_individual.is_feasible = self.is_feasible
        new_individual.constraint_violations = self.constraint_violations
        if is_evaluated:
            new_individual.is_evaluated = True
            new_individual.age = self.age
            new_individual.optim_results = self.optim_results

        return new_individual


def repair_chromosome(
    chromosome: List[int], n_categories: int, min_per_category: dict
) -> List[int]:
    """Reassign surplus samples to satisfy minimum cluster sizes."""
    chromosome = chromosome.copy()
    category_counts = Counter(chromosome)

    deficits = {}
    surpluses = {}
    for category in range(n_categories):
        count = category_counts.get(category, 0)
        min_required = min_per_category[category]

        if count < min_required:
            deficits[category] = min_required - count
        elif count > min_required:
            surpluses[category] = count - min_required

    individual_indices = list(range(len(chromosome)))
    random.shuffle(individual_indices)

    for deficit_cat, deficit_count in deficits.items():
        moved = 0
        for idx in individual_indices:
            if moved >= deficit_count:
                break

            current_cat = chromosome[idx]
            if current_cat in surpluses and surpluses[current_cat] > 0:
                chromosome[idx] = deficit_cat
                surpluses[current_cat] -= 1
                moved += 1

                if surpluses[current_cat] == 0:
                    del surpluses[current_cat]

    return chromosome


def check_constraints(
    chromosome: List[int], n_categories: int, min_per_category: dict
) -> Tuple[bool, int]:
    """Return feasibility and the total minimum-size deficit."""
    category_counts = Counter(chromosome)
    violations = 0

    for category in range(n_categories):
        count = category_counts.get(category, 0)
        if count < min_per_category[category]:
            violations += min_per_category[category] - count

    return violations == 0, violations


def evaluate_individual(
    individual: Individual,
    fitness_function: callable,
    penalty_factor: float,
    n_categories: int,
    min_per_category: dict,
) -> Individual:
    """Evaluate an individual's fitness and constraint satisfaction."""
    individual = individual.copy()

    is_feasible, violations = check_constraints(
        individual.chromosome, n_categories, min_per_category
    )

    individual.is_feasible = is_feasible
    individual.constraint_violations = violations

    optim_results = fitness_function(individual.chromosome)
    base_fitness = max([v["aco_score"] for v in optim_results.values()])
    penalty = penalty_factor * violations
    individual.fitness = base_fitness - penalty
    individual.is_evaluated = True
    individual.optim_results = optim_results

    return individual


def random_individual_init(
    n_categories: int, min_per_category: dict, chromosome_size: int
) -> Individual:
    """Create a random feasible individual."""
    chromosome = []
    for category in range(n_categories):
        chromosome.extend([category] * min_per_category[category])

    remaining = chromosome_size - len(chromosome)
    for _ in range(remaining):
        chromosome.append(random.randint(0, n_categories - 1))

    random.shuffle(chromosome)

    individual = Individual(chromosome)

    return individual


def tournament_selection(
    population: List[Individual], tournament_size: int
) -> Individual:
    """Select an individual using tournament selection."""
    tournament_size = min(len(population), tournament_size)
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x.fitness)


def crossover(
    parent1: Individual,
    parent2: Individual,
    crossover_rate: float,
    n_categories: int,
    min_per_category: dict,
) -> Tuple[Individual, Individual]:
    """Apply uniform crossover and repair offspring when crossover occurs."""
    if random.random() > crossover_rate:
        return parent1.copy(is_evaluated=True), parent2.copy(is_evaluated=True)

    child1_chromosome = []
    child2_chromosome = []

    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1_chromosome.append(parent1.chromosome[i])
            child2_chromosome.append(parent2.chromosome[i])
        else:
            child1_chromosome.append(parent2.chromosome[i])
            child2_chromosome.append(parent1.chromosome[i])

    child1 = Individual(child1_chromosome)
    child2 = Individual(child2_chromosome)

    child1 = check_and_repair_individual(child1, n_categories, min_per_category)
    child2 = check_and_repair_individual(child2, n_categories, min_per_category)

    return child1, child2


def check_and_repair_individual(
    individual: Individual, n_categories: int, min_per_category: dict
) -> Individual:
    is_feasible, _ = check_constraints(
        individual.chromosome, n_categories, min_per_category
    )
    if not is_feasible:
        new_chromosome = repair_chromosome(
            individual.chromosome, n_categories, min_per_category
        )
        new_individual = individual.copy()
        new_individual.chromosome = new_chromosome

        return new_individual

    return individual


def mutate(
    individual: Individual,
    mutation_rate: float,
    n_categories: int,
    min_per_category: dict,
    max_num_mutations: int,
) -> Individual:
    """Apply random-resetting mutation while preserving minimum cluster sizes."""
    if random.random() > mutation_rate:
        return individual

    mutated = individual.copy()
    chromosome = mutated.chromosome
    n = len(chromosome)

    counts = Counter(chromosome)

    num_mutations = random.randint(1, max_num_mutations)

    for _ in range(num_mutations):
        for _attempt in range(10):
            idx = random.randrange(n)
            old_cat = chromosome[idx]

            if counts[old_cat] <= min_per_category[old_cat]:
                continue

            new_cat = random.randrange(n_categories)
            if new_cat == old_cat:
                continue

            chromosome[idx] = new_cat
            counts[old_cat] -= 1
            counts[new_cat] += 1
            break

    mutated.chromosome = chromosome
    mutated = check_and_repair_individual(mutated, n_categories, min_per_category)

    return mutated


def random_population_init(
    population_size: int,
    n_categories: int,
    min_per_category: dict,
    chromosome_size: int,
) -> List[Individual]:
    """Initialize a population of feasible individuals."""
    population = []
    for _ in range(population_size):
        individual = random_individual_init(
            n_categories, min_per_category, chromosome_size
        )
        population.append(individual)

    return population


def get_elite(population: List[Individual], elite_size: int) -> List[Individual]:
    """Get elite individuals from current population."""
    sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
    return sorted_population[:elite_size]


def hamming_distance(a: list, b: list) -> int:
    return np.sum(np.array(a) != np.array(b))


def average_hamming_diversity(population: List[Individual]) -> float:
    pairs = list(combinations(population, 2))
    if not pairs:
        return 0
    distances = [
        hamming_distance(ind1.chromosome, ind2.chromosome) for ind1, ind2 in pairs
    ]
    return float(np.mean(distances))


def increase_population_age(population: List[Individual]):
    """Increment each population member's age in place."""
    for ind in population:
        ind.age += 1


def apply_cached_solutions(
    population: List[Individual], cache: Dict[Tuple[int], float]
):
    """Restore cached fitness values to population members in place."""
    for ind in population:
        if not ind.is_evaluated:
            chromosome_tuple = tuple(ind.chromosome)
            if chromosome_tuple in cache:
                ind.is_evaluated = True
                ind.fitness = cache[chromosome_tuple]


def cache_solutions(population: List[Individual], cache: Dict[Tuple[int], float]):
    """Add evaluated chromosomes and their fitness values to the cache."""
    for ind in population:
        if ind.is_evaluated:
            chromosome_tuple = tuple(ind.chromosome)
            if not chromosome_tuple in cache:
                cache[chromosome_tuple] = ind.fitness


def kmeans_population_init(
    clust_data: np.ndarray,
    pca_n_components: int,
    population_size: int,
    n_categories: int,
    min_per_category: dict,
    random_deviation_perc: float = 0.1,
    keep_initial_perc: float = 0.15,
    random_initial_perc: float = 0.25,
    seed: int = None,
) -> List[Individual]:
    """Initialize a population from PCA-based K-Means and random assignments."""
    if seed is not None:
        np.random.seed(seed)

    logger.info("Applying a K-means based initialization of the population")

    principal_components = PCA(n_components=pca_n_components).fit_transform(clust_data)

    kmeans = KMeans(
        n_clusters=n_categories, init="k-means++", n_init=100, random_state=seed
    )
    kmeans.fit(principal_components)

    initial_silhouette = float(silhouette_score(principal_components, kmeans.labels_))

    logger.info(f"Initial silhouette score: {initial_silhouette:.3f}")

    init_clust_assignment = kmeans.labels_

    n_preserved = int(np.ceil(keep_initial_perc * population_size))
    n_random = int(np.ceil(random_initial_perc * population_size))
    n_positions = len(init_clust_assignment)
    n_swap_positions = int(np.ceil(random_deviation_perc * n_positions))

    logger.info(f"Number of retained initial clustering solutions: {n_preserved}")
    logger.info(
        f"Number of perturbed positions per initial solution: {n_swap_positions}"
    )
    logger.info(f"Number of random solutions added: {n_random}")

    init_chromosomes = [
        deepcopy(init_clust_assignment.tolist()) for _ in range(n_preserved)
    ]

    init_chromosomes = init_chromosomes + [
        np.random.randint(0, n_categories, size=n_positions).tolist()
        for _ in range(n_random)
    ]

    for _ in range(len(init_chromosomes), population_size):
        indices = np.arange(n_positions)
        np.random.shuffle(indices)
        swap_indices = indices[:n_swap_positions]
        init_clust_assignment_ = deepcopy(init_clust_assignment)
        for idx in swap_indices:
            init_clust_assignment_[idx] = int(
                np.random.choice(
                    [i for i in range(n_categories) if i != init_clust_assignment[idx]]
                )
            )
        init_chromosomes.extend([init_clust_assignment_.tolist()])

    population = [
        check_and_repair_individual(
            Individual(chromosome), n_categories, min_per_category
        )
        for chromosome in init_chromosomes
    ]

    return population
