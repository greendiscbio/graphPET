"""Shared ant colony system parameters."""

import antco_css as antco

iterations: int = 300
n_ants: int = 40
informed_start_ratio: float = 0.5
pheromone_init: float = 5.0
pheromone_update_exp: float = 1.0
heuristic_update_exp: float = 1.0
evaporation: float = 0.02
local_evaporation: float = 0.1
prob_deterministic: float = 0.1
prob_random: float = 0.05
n_best_current: int = 10
n_best_so_far: int = 3
coef_strong_reinforcement: float = 1.0
coef_weak_reinforcement: float = 0.5
local_pheromone_base: float = 0.1
random_seed: int = 42


def build_config(n_nodes: int) -> antco.config.AcoConfig:
    """Build the ACS configuration for a target subnetwork size."""
    return antco.config.AcoConfig(
        iterations=iterations,
        n_ants=n_ants,
        informed_start_ratio=informed_start_ratio,
        k=n_nodes,
        pheromone_init=pheromone_init,
        pheromone_update_exp=pheromone_update_exp,
        heuristic_update_exp=heuristic_update_exp,
        evaporation=evaporation,
        local_evaporation=local_evaporation,
        prob_deterministic=prob_deterministic,
        coef_strong_reinforcement=coef_strong_reinforcement,
        coef_weak_reinforcement=coef_weak_reinforcement,
        local_pheromone_base=local_pheromone_base,
        random_seed=random_seed,
        n_best_current=n_best_current,
        n_best_so_far=n_best_so_far,
        prob_random=prob_random,
    )
