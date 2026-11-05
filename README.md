# Machine Learning-Based Adaptive Cache Management for Web Systems

A reproducible college mini-project comparing **LRU**, **LFU**, and **random-forest-based cache replacement** in a trace-driven web cache. Everything runs locally using synthetic data; no APIs, credentials, datasets, or external services are needed after dependency installation. Results are simulations, not measurements of a production web server. ML is not assumed to win.

## Problem and architecture

A web cache has limited byte capacity. Which objects should it retain to reduce backend fetches and response latency as popularity changes?

```text
Synthetic workload → chronological 50/20/30 split
                       ├─ train → causal feature snapshots + future labels → forest
                       ├─ validation → diagnostic predictive metrics
                       └─ held-out test → independent empty LRU / LFU / ML caches
                                          ↓
                                per-seed metrics → aggregation, CSV, figures, dashboard
```

The simulator handles sequential requests `(timestamp, object ID, size in bytes, backend retrieval ms)` without real HTTP requests or sleeps. Each object has stable metadata. Every lookup pays configurable cache latency; a miss additionally pays backend retrieval latency and counts one backend access. Retrieval latency includes fixed server cost, object transfer cost and a seeded per-object component. Arrival timestamps provide temporal features; there is no queueing or concurrent service model.

Capacity is enforced after every request. Objects larger than the cache bypass it without evicting residents. Every cacheable miss is inserted, under **all** policies, after as many evictions as necessary. There is no selective ML admission. Occupancy assertions catch byte accounting errors.

## Policies and prediction

- **LRU:** evict the resident with oldest last access.
- **LFU:** count accesses **since the current insertion**, starting at one and incrementing on hits. Evict the smallest count, breaking ties by last access, then integer object ID. Counts reset after eviction/reinsertion. This is resident LFU, not lifetime LFU.
- **ML:** rank residents by `P(reuse in the next H requests) / size_bytes` and evict the lowest value first. Ties use last access, then object ID. Rank once at the miss time and remove residents until the new object fits. Features do not depend on eviction order, so reranking after each removal would be redundant. This heuristic values expected reuse per byte; it is not an optimal packing algorithm or a prediction of exact time to next use. It does not explicitly optimize latency savings.

The `RandomForestClassifier` uses 60 trees, maximum depth 10, minimum leaf size 4, seeded randomness, one worker, and balanced subsample class weights. Parameters are fixed before evaluation. Features are recency in seconds, counts in the last 10 and 50 requests, total observed access count, object size, mean and standard deviation of the last ten observed inter-arrival gaps, and a cold-start indicator. Recent counts include the current observed event. Times and request-index windows are distinct. History stores bounded windows and cumulative counts; one record per known object is retained. Cache residency is omitted because it depends on the policy being evaluated.

Unknown objects have explicit zero-history features and a cold-start flag. Objects with one access have zero gap statistics. Resident candidates necessarily have observed history; the flag is defensive and is not learned from artificial unseen catalog entries. No catalog metadata for future unseen objects enters training.

### Labels and leakage prevention

At decision index `i`, history observes request `i`, then candidate features are constructed. The label is one if that candidate occurs in requests `i+1` through `i+H`, inclusive; current access is excluded. Only the label builder reads future events. Every fifth request samples up to five **already known** objects uniformly, independent of future reuse. This includes idle objects rather than training only on objects at their own accesses.

Split the original trace chronologically into 50% train, 20% validation, 30% test. Construct examples independently within each partition. Exclude its final H decision times because their complete label horizon is unavailable. Thus no training label reaches validation and no validation label reaches test. History resets at partition boundaries and test simulation begins with empty history and empty cache for all policies. Training and validation requests never warm test caches. The fitted forest is frozen; adaptation during test comes from changing causal features, not online retraining.

Validation reports diagnostic scores without fitting or tuning on it. Precision, recall and F1 use a 0.5 threshold; eviction uses continuous scores. ROC-AUC is null when only one label class exists. Single-class forests are supported. Class-weighted forest scores are not calibrated probabilities, and correlated overlapping horizons limit interpretation of classification scores. Resident candidates can differ from uniformly sampled known training objects (distribution shift).

## Installation (Python 3.11+)

From this repository directory, on Windows PowerShell / VS Code:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```

Use any installed Python 3.11+ version if `py -3.11` is unavailable. If activation is restricted, call `.\.venv\Scripts\python.exe` directly instead of changing PowerShell security settings.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```

`requirements.txt` defines supported version ranges. `requirements-lock.txt` records exact versions used in the validated Python 3.12 Linux environment; optionally install it for closer reproduction. The lock is not a guarantee of compatibility on every future platform. No GPU is needed.

## Train and run experiments

```bash
# Save a local model and predictive metrics for the first selected workload/seed.
python -m adaptive_cache train --workloads zipf --seeds 11 --requests 3000

# Full paired experiment: four workloads × three seeds × three byte capacities × three policies.
python -m adaptive_cache experiment --requests 3000 --seeds 11 22 33 --capacities 16384 32768 65536

# Smaller end-to-end demonstration.
python -m adaptive_cache experiment --requests 600 --objects 20 --horizon 10 --seeds 11 22 --capacities 8192 16384
```

The request count is the **whole** trace count, not the number of test requests. Each partition must contain more requests than H. Training artifacts default to ignored `artifacts/`; choose `--output` to keep separate runs. Experiment models are fitted independently per workload and seed and shared across capacities; the CLI train artifact is for inspection/reuse and is not silently loaded by experiments. Do not load untrusted joblib files.

Experiment output includes:

- `results.csv`: one row per seed, workload, capacity, policy.
- `summary.csv`: means and sample standard deviations across seeds (std is undefined for a single seed).
- `predictive_metrics.csv`: validation/test classification metrics, sample counts, prevalence and training time (includes validation diagnostics).
- `winners.csv`: every policy tied for highest mean object hit rate, explicitly including LRU and LFU winners.
- `configuration.json`: input parameters; plot PNGs for hit rate, mean response and backend fetches versus capacity, faceted by workload, with standard deviation bars.

Workloads: uniform; Zipf popularity (default exponent 1.2); bursty (85% requests target a hot subset refreshed every 250 requests); and shifting (Zipf ranks rotate every 250 requests). Sizes default to seeded integers from 512–8192 bytes, or fixed 4096 bytes. The Python generator additionally exposes Zipf exponent, phase length and mean arrival interval. The dashboard exposes fixed/variable sizes, request count, seed count, horizon, hit latency and capacity. It runs three capacities around the selected value and offers raw CSV export.

## Experimental methodology and metrics

All policies process exactly the same held-out requests with the same metadata, cache capacity, and hit latency. Every simulation creates fresh policy/cache/history state. Multiple seeds vary both workload generation and forest randomness; pair comparisons by seed. Cache sizes are bytes, not object counts. No test data is used in fitting or hyperparameter selection. Seeds and workloads describe a synthetic benchmark, not independent samples from real production traffic.

- **Object hit/miss rates:** hits/misses divided by number of requests × 100.
- **Backend accesses:** number of misses, including oversize bypasses.
- **Average and P95 response (ms):** mean and NumPy's default linearly interpolated 95th percentile of simulated request latencies.
- **Byte hit rate:** bytes served on cache hits divided by all requested bytes × 100.
- **Utilization:** mean post-request occupancy divided by capacity × 100 (request-weighted, not time-weighted).
- **Policy overhead:** separately timed wall-clock bookkeeping, history extraction, admission and eviction, including RF inference when needed, summed over requests and reported per request. It excludes model training, simulator validation assertions, metrics aggregation, and simulated backend work. It is machine-dependent and **not** added to simulated response latency. Baselines share history maintenance to keep bookkeeping consistent.

Small synthetic runs can favor either baseline; the winner report retains ties and does not hide losses. Hit-rate winners need not win byte hit rate or latency. Standard deviations describe variation, not statistical significance; use more seeds and paired confidence intervals before drawing strong claims. See [VALIDATION.md](VALIDATION.md) for the measured development smoke run.

## Project structure

```text
adaptive_cache/
  workload.py       seeded catalogs, four workloads, chronological splitting
  simulator.py      byte accounting, retrieval semantics, metrics
  policies.py       LRU, LFU, ML ranking
  features.py       observable history and within-partition labels
  model.py          forest fitting, inference, predictive evaluation
  experiments.py    paired runs, aggregation and winners
  plots.py          Matplotlib/Seaborn exported figures
  __main__.py       executable train / experiment CLI
app.py              Streamlit dashboard
tests/              baseline, byte capacity, causal ML and dashboard tests
```

## Limitations and future work

Synthetic popularity is simplified. Objects are immutable; there is no TTL, invalidation, write traffic, prefetching, multi-tier storage, network contention, concurrency, or backend queueing. Time gaps do not simulate real serving delays. Test is a cold-start measurement and includes compulsory misses. A fixed RF is not an online learning algorithm; drift may favor LRU or LFU. Per-eviction inference scales with resident count, so this implementation is for small educational experiments rather than production throughput. Training candidate sampling, class weighting and probability-per-byte ranking introduce modeling choices, not guarantees. Metadata and model memory do not count against object byte capacity; unlimited historical object IDs are another scaling limitation.

Future work: real anonymized traces, calibrated reuse scores, temporal cross-validation, paired uncertainty intervals, online retraining with delayed labels, explicit admission control, latency-weighted value, TTLs, and inference batching/caching. Such variants require separate evaluation rather than claiming improvements without experiments.
