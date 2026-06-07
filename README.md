# EOMM Mobile Legends — Lose Streak Experiment

This repository contains an experimental framework for evaluating team-selection algorithms
(Greedy, Branch & Bound, Genetic Algorithm) for an "EOMM" matchmaking scenario focused on
a lose-streak player.

## Project structure

- `benchmark.py` — integration script to run experiments and compare algorithms.
- `data_generator.py` — reproducible fake queue generator and `Player` model.
- `greedy_selector.py` — greedy baseline that picks top composite scores.
- `bnb_selector.py` — Branch-and-Bound exact solver with pruning statistics.
- `ga_selector.py` — Genetic Algorithm solver (tournament selection + crossover).
- `eomm_results.csv` — (example) CSV output produced by `benchmark.py`.

## Requirements

- Python 3.10+ recommended.
- No external packages required for core functionality.
- Optional: `matplotlib` and `numpy` to enable plot generation in `benchmark.py`.

Install optional deps with:

```bash
python -m pip install matplotlib numpy
```

## Running the experiments

Run the benchmark script from the project root:

```bash
python benchmark.py
```

`benchmark.py` will:
- Run experiments for queue sizes configured in `QUEUE_SIZES`.
- Repeat each experiment according to `REPEATS`.
- Compare running time and solution quality between Greedy, B&B, and GA.
- Export results to `eomm_results.csv` in the same folder.

## Configuration

- Adjust `QUEUE_SIZES`, `REPEATS`, and `GA_PARAMS` inside `benchmark.py`.
- Set the random seed in `data_generator.py` via `RANDOM_SEED` for reproducibility.
- Team size is defined by `TEAM_SIZE` in `data_generator.py` (default 4).

## Output

- The CSV `eomm_results.csv` contains per-experiment metrics (times, CS, optimality %,
  B&B node statistics, etc.).
- If `matplotlib` is installed, `benchmark.py` attempts to use a non-interactive backend
  and can be extended to save figures.

## Notes

- The Branch-and-Bound implementation sorts players by composite score to improve
  pruning effectiveness — worst-case complexity remains combinatorial for large `n`.
- The GA implementation uses integer index-based chromosomes referencing the queue.

## License & Contact

This project is for academic/prototyping use. For questions, open an issue or contact
the repository owner.
