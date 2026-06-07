"""
File: greedy_selector.py
Deskripsi: Implementasi Algoritma Greedy berbasis pengurutan Composite Score.
"""

from typing import List, Tuple
from data_generator import Player, total_cs, TEAM_SIZE

def greedy_select(queue: List[Player], k: int = TEAM_SIZE) -> Tuple[List[Player], float]:
    """
    Algoritma Greedy: memilih k pemain dengan Composite Score tertinggi.
    Kompleksitas: O(N log N)
    """
    # Urutkan antrian secara menurun berdasarkan CS
    sorted_queue = sorted(queue, key=lambda p: p.composite_score(), reverse=True)
    # Ambil k pemain pertama
    selected = sorted_queue[:k]
    
    return selected, total_cs(selected)