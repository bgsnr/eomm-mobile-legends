"""
File: data_generator.py
Deskripsi: Modul pembangkitan data antrian pemain fiktif yang reproducible.
"""

import random
from dataclasses import dataclass
from typing import List

# Konfigurasi Global
RANDOM_SEED = 42
TEAM_SIZE = 4
W_SR = 0.7   # bobot Skill Rating
W_WR = 0.3   # bobot Win Rate

SR_MIN, SR_MAX = 100, 1000
WR_MIN, WR_MAX = 0.30, 0.70

# Pemain lose streak (subjek utama)
LOSE_STREAK_PLAYER = {"id": "USER", "sr": 450, "wr": 0.38}

@dataclass
class Player:
    """Representasi seorang pemain dalam antrian matchmaking."""
    pid:   str
    sr:    int     
    wr:    float   

    def composite_score(self) -> float:
        """Fungsi objektif komposit: CS = W_SR * SR + W_WR * (WR * 1000)"""
        return W_SR * self.sr + W_WR * (self.wr * 1000)

    def __repr__(self) -> str:
        return (f"Player({self.pid}, SR={self.sr}, "
                f"WR={self.wr:.2f}, CS={self.composite_score():.1f})")

def generate_queue(n: int, seed: int = RANDOM_SEED) -> List[Player]:
    """Membangkitkan antrian N pemain fiktif secara pseudo-random."""
    rng = random.Random(seed)
    players = []
    for i in range(n):
        pid = f"P{i+1:04d}"
        sr  = rng.randint(SR_MIN, SR_MAX)
        wr  = round(rng.uniform(WR_MIN, WR_MAX), 2)
        players.append(Player(pid, sr, wr))
    return players

def total_cs(players: List[Player]) -> float:
    """Menghitung total Composite Score dari sekumpulan pemain."""
    return sum(p.composite_score() for p in players)