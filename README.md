# Smart Order Router Backtest — Based on Cont & Kukanov Model

This project implements and back-tests a Smart Order Router (SOR) that follows the static cost minimization framework proposed by Cont & Kukanov (2014)for optimal order placement across fragmented limit order markets.

The system aims to minimize total expected execution cost of a 5,000-share buy order by distributing it across multiple venues using a static allocator and tuning over three risk parameters: `lambda_over`, `lambda_under`, and `theta_queue`.

---

## Repository Contents

- `backtest.py` – Complete implementation of the static allocator and backtest loop. Outputs performance JSON and generates `results.png`.
- `allocator_psuedocode.txt` – Provided logic for the snapshot-based allocator.
- `l1_day.csv` – Level-1 historical market data (mocked), containing ~60,000 messages from August 1, 2024.
- `Optimal Order Placement in Limit Order Markets.pdf` – Reference paper by Cont & Kukanov.
- `Trial Task Description.pdf` – Detailed requirements for the trial task.
- `results.png` – Cumulative cost plot using best-tuned parameters.

---

## Problem Setup

The router must decide how to allocate shares at each market snapshot, considering:
- Displayed ask price and size for each venue,
- Fees and rebates per venue (fixed in code),
- Execution penalties:
  - `lambda_under`: cost of underfilling the order,
  - `lambda_over`: cost of overfilling it,
  - `theta_queue`: penalty for non-execution risk.

The router runs a grid search over these parameters and selects the configuration that minimizes total cost.

---

# How It Works

# Step-by-step:
1. Preprocess: Sort the data and extract one quote per venue per `ts_event`.
2. Allocator: Implements the snapshot-based dynamic programming logic from the pseudocode file.
3. Backtest: Simulates execution from the first snapshot to the last, rolling unfilled shares forward.
4. Benchmarks:
   - Best Ask: Always chooses lowest current ask.
   - TWAP: Executes evenly across all time slices.
   - VWAP: Allocates based on volume-weighted price.
5. Evaluation: Prints a JSON report and generates a cumulative cost plot.


## Parameter Grid

```python
lambda_over ∈ [0.1, 0.5, 1.0]
lambda_under ∈ [0.1, 0.5, 1.0]
theta_queue ∈ [0.0, 0.1, 0.2]
