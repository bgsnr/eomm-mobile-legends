"""
File: benchmark.py
Deskripsi: Skrip integrasi untuk menjalankan seluruh skenario eksperimen EOMM.
"""

import time
import math
import csv
import os
import sys
from dataclasses import dataclass
from typing import List

# Import dari modul custom yang telah dipecah
from data_generator import (Player, generate_queue, total_cs, LOSE_STREAK_PLAYER, 
                            TEAM_SIZE, W_SR, W_WR, RANDOM_SEED)
from greedy_selector import greedy_select
from bnb_selector import bnb_select
from ga_selector import ga_select

# ── Matplotlib (opsional) ────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARN] matplotlib/numpy tidak ditemukan. Grafik tidak akan dibuat.")


# ═══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI EKSPERIMEN (Disamakan dengan isi makalah)
# ═══════════════════════════════════════════════════════════════════════════════
QUEUE_SIZES = [50, 100, 500, 1000]

# DISESUAIKAN DENGAN MAKALAH: N=50,100 -> 10 kali; N=500,1000 -> 5 kali
REPEATS = {50: 10, 100: 10, 500: 5, 1000: 5}

GA_PARAMS = {
    50:   {"pop_size": 50,  "generations": 200, "mutation_rate": 0.02, "tournament_k": 3},
    100:  {"pop_size": 100, "generations": 300, "mutation_rate": 0.02, "tournament_k": 3},
    500:  {"pop_size": 200, "generations": 500, "mutation_rate": 0.03, "tournament_k": 4},
    1000: {"pop_size": 300, "generations": 800, "mutation_rate": 0.03, "tournament_k": 4},
}

C = {
    "reset":  "\033[0m", "bold":   "\033[1m", "red":    "\033[91m",
    "green":  "\033[92m", "yellow": "\033[93m", "blue":   "\033[94m",
    "cyan":   "\033[96m", "white":  "\033[97m", "gray":   "\033[90m",
}

def cprint(text: str, color: str = "white", bold: bool = False) -> None:
    prefix = C.get("bold", "") if bold else ""
    print(f"{prefix}{C.get(color, '')}{text}{C['reset']}")


@dataclass
class ExperimentResult:
    n: int
    greedy_time_ms: float
    greedy_cs: float
    bnb_time_ms: float
    bnb_cs: float
    bnb_nodes_exp: int
    bnb_nodes_prun: int
    bnb_exp_pct: float
    ga_time_ms: float
    ga_cs: float
    greedy_opt_pct: float = 0.0
    ga_opt_pct: float = 0.0

def run_benchmark(n: int, repeats: int, verbose: bool = True) -> ExperimentResult:
    if verbose:
        cprint(f"\n{'═'*60}", "cyan")
        cprint(f"  Menjalankan benchmark N = {n} ({repeats} ulangan)...", "cyan", bold=True)
    
    queue = generate_queue(n)

    # GREEDY
    greedy_times, greedy_results = [], []
    for rep in range(repeats):
        t_start = time.perf_counter()
        sel, score = greedy_select(queue)
        t_end = time.perf_counter()
        greedy_times.append((t_end - t_start) * 1000)
        greedy_results.append(score)
        if verbose:
            sys.stdout.write(f"\r  Greedy : ulangan {rep+1}/{repeats} ")
            sys.stdout.flush()
    if verbose: print()

    # B&B (Khusus N besar, dibatasi karena O(2^N) jika pruning tidak optimal, 
    # namun B&B kita efektif sehingga akan jalan sesuai REPEATS makalah)
    bnb_times, bnb_results, bnb_stats_list = [], [], []
    for rep in range(repeats):
        t_start = time.perf_counter()
        sel, score, stats = bnb_select(queue)
        t_end = time.perf_counter()
        bnb_times.append((t_end - t_start) * 1000)
        bnb_results.append(score)
        bnb_stats_list.append(stats)
        if verbose:
            sys.stdout.write(f"\r  B&B    : ulangan {rep+1}/{repeats} pruned={stats['nodes_pruned']:,} ")
            sys.stdout.flush()
    if verbose: print()

    # GENETIC ALGORITHM
    ga_params = GA_PARAMS[n]
    ga_times, ga_results = [], []
    for rep in range(repeats):
        t_start = time.perf_counter()
        sel, score = ga_select(queue, TEAM_SIZE, **ga_params)
        t_end = time.perf_counter()
        ga_times.append((t_end - t_start) * 1000)
        ga_results.append(score)
        if verbose:
            sys.stdout.write(f"\r  GA     : ulangan {rep+1}/{repeats} ")
            sys.stdout.flush()
    if verbose: print()

    # AVERAGE & OPTIMALITY
    g_time = sum(greedy_times) / len(greedy_times)
    g_cs   = sum(greedy_results) / len(greedy_results)
    
    b_time = sum(bnb_times) / len(bnb_times)
    b_cs   = sum(bnb_results) / len(bnb_results)
    b_exp  = sum(s["nodes_explored"] for s in bnb_stats_list) // repeats
    b_prun = sum(s["nodes_pruned"] for s in bnb_stats_list) // repeats
    b_pct  = sum(s["explored_pct"] for s in bnb_stats_list) / repeats

    ga_time = sum(ga_times) / len(ga_times)
    ga_cs   = sum(ga_results) / len(ga_results)

    optimal_cs = b_cs
    g_opt = (g_cs / optimal_cs) * 100 if optimal_cs > 0 else 0
    ga_opt = (ga_cs / optimal_cs) * 100 if optimal_cs > 0 else 0

    return ExperimentResult(n, g_time, g_cs, b_time, b_cs, b_exp, b_prun, b_pct, ga_time, ga_cs, g_opt, ga_opt)


def print_banner() -> None:
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║   Simulasi EOMM Mobile Legends — Kasus Lose Streak                   ║
║   Makalah Proyek ASA 2026 — S1 Informatika UNDIP                     ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    cprint(banner, "cyan", bold=True)

def print_running_time_table(results: List[ExperimentResult]) -> None:
    cprint("\n  TABEL 4.1 — PERBANDINGAN RUNNING TIME (ms)", "yellow", bold=True)
    cprint(f"  {'─'*68}", "gray")
    header = f"  {'N':>8} │ {'Greedy (ms)':>14} │ {'B&B (ms)':>16} │ {'GA (ms)':>14}"
    cprint(header, "white", bold=True)
    cprint(f"  {'─'*68}", "gray")
    for r in results:
        print(f"  {'N='+str(r.n):>8} │ {r.greedy_time_ms:>14.3f} │ {r.bnb_time_ms:>16.1f} │ {r.ga_time_ms:>14.1f}")
    cprint(f"  {'─'*68}", "gray")

def print_quality_table(results: List[ExperimentResult]) -> None:
    cprint("\n  TABEL 4.2 — PERBANDINGAN KUALITAS SOLUSI (CS & OPT%)", "yellow", bold=True)
    cprint(f"  {'─'*80}", "gray")
    header = f"  {'N':>8} │ {'Greedy CS':>12} {'OPT%':>7} │ {'B&B CS':>12} {'OPT%':>7} │ {'GA CS':>12} {'OPT%':>7}"
    cprint(header, "white", bold=True)
    cprint(f"  {'─'*80}", "gray")
    for r in results:
        print(f"  {'N='+str(r.n):>8} │ {r.greedy_cs:>12.2f} {r.greedy_opt_pct:>6.2f}% │ "
              f"{r.bnb_cs:>12.2f} {'100.00%':>7} │ {r.ga_cs:>12.2f} {r.ga_opt_pct:>6.2f}%")
    cprint(f"  {'─'*80}", "gray")

def export_csv(results: List[ExperimentResult], output_file: str) -> None:
    fieldnames = [
        "N", "greedy_time_ms", "greedy_cs", "greedy_opt_pct",
        "bnb_time_ms", "bnb_cs", "bnb_opt_pct",
        "bnb_nodes_explored", "bnb_nodes_pruned", "bnb_explored_pct",
        "ga_time_ms", "ga_cs", "ga_opt_pct"
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "N": r.n, "greedy_time_ms": f"{r.greedy_time_ms:.4f}",
                "greedy_cs": f"{r.greedy_cs:.4f}", "greedy_opt_pct": f"{r.greedy_opt_pct:.4f}",
                "bnb_time_ms": f"{r.bnb_time_ms:.4f}", "bnb_cs": f"{r.bnb_cs:.4f}",
                "bnb_opt_pct": "100.0000", "bnb_nodes_explored": r.bnb_nodes_exp,
                "bnb_nodes_pruned": r.bnb_nodes_prun, "bnb_explored_pct": f"{r.bnb_exp_pct:.6f}",
                "ga_time_ms": f"{r.ga_time_ms:.4f}", "ga_cs": f"{r.ga_cs:.4f}",
                "ga_opt_pct": f"{r.ga_opt_pct:.4f}",
            })
    cprint(f"  ✓ Hasil CSV disimpan: {output_file}", "green")

def main() -> None:
    print_banner()
    cprint("  Konfigurasi Eksperimen:", "white", bold=True)
    print(f"    Random Seed      : {RANDOM_SEED}")
    print(f"    Ulangan / Repeats: {REPEATS}")

    results = []
    for n in QUEUE_SIZES:
        rep = REPEATS.get(n, 3)
        res = run_benchmark(n, repeats=rep, verbose=True)
        results.append(res)

    print_running_time_table(results)
    print_quality_table(results)
    export_csv(results, "eomm_results.csv")
    cprint("\n  EKSPERIMEN SELESAI!", "green", bold=True)

if __name__ == "__main__":
    main()