"""
File: bnb_selector.py
Deskripsi: Implementasi Branch and Bound dengan Bounding Function berbasis pengurutan CS.
"""

import math
from typing import List, Tuple
from data_generator import Player, TEAM_SIZE

class BranchAndBound:
    def __init__(self, queue: List[Player], k: int = TEAM_SIZE):
        self.queue   = sorted(queue, key=lambda p: p.composite_score(), reverse=True)
        self.n       = len(queue)
        self.k       = k
        self.best_cs = -math.inf
        self.best_selection: List[int] = [] 
        self.nodes_explored = 0
        self.nodes_pruned   = 0

    def _upper_bound(self, idx: int, selected_count: int, current_cs: float) -> float:
        remaining_needed = self.k - selected_count
        if remaining_needed <= 0:
            return current_cs
        
        ub = current_cs
        count = 0
        for j in range(idx, self.n):
            if count >= remaining_needed:
                break
            ub += self.queue[j].composite_score()
            count += 1
            
        if count < remaining_needed:
            return -math.inf
        return ub

    def _dfs(self, idx: int, selected: List[int], current_cs: float) -> None:
        self.nodes_explored += 1

        if len(selected) == self.k:
            if current_cs > self.best_cs:
                self.best_cs = current_cs
                self.best_selection = selected.copy()
            return

        remaining_slots = self.k - len(selected)
        remaining_players = self.n - idx
        if remaining_players < remaining_slots:
            return

        ub = self._upper_bound(idx, len(selected), current_cs)
        if ub <= self.best_cs:
            self.nodes_pruned += 1
            return

        selected.append(idx)
        player_cs = self.queue[idx].composite_score()
        self._dfs(idx + 1, selected, current_cs + player_cs)
        selected.pop()

        self._dfs(idx + 1, selected, current_cs)

    def solve(self) -> Tuple[List[Player], float]:
        self._dfs(0, [], 0.0)
        selected_players = [self.queue[i] for i in self.best_selection]
        return selected_players, self.best_cs

    def get_stats(self) -> dict:
        total_possible = math.comb(self.n, self.k)
        explored_pct = (self.nodes_explored / max(total_possible, 1)) * 100
        return {
            "nodes_explored": self.nodes_explored,
            "nodes_pruned": self.nodes_pruned,
            "total_possible": total_possible,
            "explored_pct": explored_pct,
        }

def bnb_select(queue: List[Player], k: int = TEAM_SIZE) -> Tuple[List[Player], float, dict]:
    solver = BranchAndBound(queue, k)
    selected, score = solver.solve()
    return selected, score, solver.get_stats()