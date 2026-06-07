"""
File: ga_selector.py
Deskripsi: Implementasi Genetic Algorithm dengan Tournament Selection dan Order Crossover.
"""

import random
from typing import List, Tuple
from data_generator import Player, TEAM_SIZE, RANDOM_SEED

class GeneticAlgorithm:
    def __init__(
        self, queue: List[Player], k: int = TEAM_SIZE, pop_size: int = 100,
        generations: int = 300, mutation_rate: float = 0.02,
        tournament_k: int = 3, seed: int = RANDOM_SEED,
    ):
        self.queue         = queue
        self.n             = len(queue)
        self.k             = k
        self.pop_size      = pop_size
        self.generations   = generations
        self.mutation_rate = mutation_rate
        self.tournament_k  = tournament_k
        self.rng           = random.Random(seed)
        self.fitness_history: List[float] = []

    def _make_chromosome(self) -> List[int]:
        return self.rng.sample(range(self.n), self.k)

    def _fitness(self, chromosome: List[int]) -> float:
        if len(set(chromosome)) != self.k:
            return 0.0
        return sum(self.queue[i].composite_score() for i in chromosome)

    def _tournament_select(self, population: List[List[int]]) -> List[int]:
        candidates = self.rng.choices(population, k=self.tournament_k)
        return max(candidates, key=self._fitness)

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        k = self.k
        pt = self.rng.randint(1, k - 1)

        seg1 = parent1[:pt]
        rem1 = [x for x in parent2 if x not in seg1]
        child1 = seg1 + rem1[:k - pt]

        seg2 = parent2[:pt]
        rem2 = [x for x in parent1 if x not in seg2]
        child2 = seg2 + rem2[:k - pt]

        return child1, child2

    def _mutate(self, chromosome: List[int]) -> List[int]:
        if self.rng.random() < self.mutation_rate:
            chrom = chromosome.copy()
            mut_pos = self.rng.randint(0, self.k - 1)
            all_indices = set(range(self.n))
            available = list(all_indices - set(chrom))
            if available:
                chrom[mut_pos] = self.rng.choice(available)
            return chrom
        return chromosome

    def solve(self) -> Tuple[List[Player], float]:
        population = [self._make_chromosome() for _ in range(self.pop_size)]
        best_chromosome = max(population, key=self._fitness)
        best_fitness    = self._fitness(best_chromosome)

        for gen in range(self.generations):
            new_population = [best_chromosome.copy()]
            
            while len(new_population) < self.pop_size:
                p1 = self._tournament_select(population)
                p2 = self._tournament_select(population)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)

            population = new_population

            gen_best = max(population, key=self._fitness)
            gen_best_fitness = self._fitness(gen_best)
            if gen_best_fitness > best_fitness:
                best_fitness    = gen_best_fitness
                best_chromosome = gen_best.copy()

            self.fitness_history.append(best_fitness)

        selected_players = [self.queue[i] for i in best_chromosome]
        return selected_players, best_fitness

def ga_select(
    queue: List[Player], k: int = TEAM_SIZE, pop_size: int = 100,
    generations: int = 300, mutation_rate: float = 0.02, tournament_k: int = 3,
) -> Tuple[List[Player], float]:
    ga = GeneticAlgorithm(queue, k, pop_size, generations, mutation_rate, tournament_k)
    return ga.solve()