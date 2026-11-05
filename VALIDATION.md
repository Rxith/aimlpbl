# Development validation

Executed on 2026-11-05 using Python 3.12.14 on Linux. Exact package versions are in `requirements-lock.txt`. Windows commands are documented but were not run on Windows.

- `python -m pytest -q`: **21 passed**. Coverage includes baseline hits, misses, deterministic eviction, LFU reinsertion semantics, repeated evictions for byte capacity, oversize bypass, workload reproducibility, model reproducibility, single-class labels, causality under future-trace changes, complete label horizons, size-based ML ranking, paired experiment output, and a Streamlit interaction that trains and renders results.
- `python -m pip check`: no broken requirements.
- `python -m adaptive_cache train --requests 600 --objects 20 --horizon 10 --seeds 11 --workloads zipf --output artifacts/train-smoke`: completed and saved a model and JSON metrics. Test precision 0.6596, recall 0.5082, F1 0.5741 and ROC-AUC 0.7452 on 165 sampled examples. These are small synthetic diagnostic results, not generalization claims.
- The experiment below completed **48 simulations** (four workloads × two seeds × two capacities × three policies), each with **180 held-out requests**. It wrote raw/aggregated CSV, prediction metrics, configuration, winner tables and three PNG plots.
- Streamlit started on loopback and its health endpoint returned HTTP 200 (`ok`). AppTest separately exercised the actual application code and result rendering, rather than relying on the health endpoint alone.

```bash
python -m adaptive_cache experiment --requests 600 --objects 20 --horizon 10 --seeds 11 22 --capacities 8192 16384 --output artifacts/smoke
```

The actual per-seed records, prediction metrics and run configuration are retained in `examples/smoke_*.csv` and `examples/smoke_configuration.json`. Plots and model files are generated into ignored `artifacts/` and can be regenerated with the commands above. No measured values below were invented or selected from additional tuning runs.

Mean object hit rate (%), two seeds, synthetic variable-size catalog:

| Workload | Capacity (bytes) | LRU | LFU | ML |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 8192 | 8.06 | 8.06 | 10.28 |
| Uniform | 16384 | 15.28 | 14.17 | 31.39 |
| Zipf | 8192 | 24.44 | 28.89 | 30.00 |
| Zipf | 16384 | 44.44 | **55.56** | 53.61 |
| Bursty | 8192 | 71.94 | 73.89 | 74.44 |
| Bursty | 16384 | 83.33 | 85.28 | 85.83 |
| Shifting | 8192 | 23.61 | 24.17 | 25.83 |
| Shifting | 16384 | 42.50 | 43.06 | 51.67 |

**A baseline wins:** at Zipf / 16384 bytes LFU beats ML on mean hit rate and mean simulated response (15.65 versus 17.11 ms). LFU wins that hit-rate comparison in both seeds. It also beats ML in seed 11's bursty runs at both capacities. LRU did not win object hit rate in this smoke run; this does not imply it cannot win other settings.

The forest has much higher measured computation cost: roughly 177–1373 microseconds per request across these aggregate settings, versus roughly 1–2 microseconds for baselines on this machine. These times are excluded from simulated latency. Concurrent development checks were running, so timings are descriptive and not a controlled microbenchmark. The uniform workload's ML gains partly reflect favoring small objects, not learned future knowledge; equal-size and size-heuristic-only ablations would be needed to isolate the model's contribution.

Only two seeds and short test windows were used for the smoke run. It establishes execution, not statistical significance or production superiority. The documented default CLI runs three seeds and larger traces; extend the seed count for a report. No full default-scale benchmark or native Windows test is claimed.
