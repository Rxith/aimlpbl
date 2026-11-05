"""Interactive, reproducible synthetic cache experiment dashboard."""
import pandas as pd
import streamlit as st
from adaptive_cache.workload import WORKLOADS, generate_trace, chronological_split
from adaptive_cache.experiments import run_experiments, aggregate, winners

st.set_page_config(page_title="Adaptive Web Cache Lab", page_icon="🗃️", layout="wide")
st.title("Adaptive Web Cache Lab")
st.caption("Machine Learning-Based Adaptive Cache Management for Web Systems")
st.info("Synthetic trace-driven simulation — no live web traffic. Response latency is simulated; computation overhead is measured separately.")
with st.expander("Project overview and policy definitions"):
    st.markdown("""A byte-limited cache reduces backend fetches by retaining reusable web objects.
Compare traditional policies with a random forest that predicts reuse in the next H requests.

- **LRU:** evicts the least recently accessed resident object.
- **LFU:** evicts the resident with the fewest accesses since insertion; ties use LRU, then object ID.
- **ML:** evicts the lowest predicted reuse probability per byte; ties use LRU, then object ID. This size heuristic is distinct from the predictor and need not outperform either baseline.

Every run uses a 50% training / 20% validation / 30% test chronological split. Complete label horizons stay within each partition. Test caches and feature histories start empty. The model is frozen during testing; it responds to current history without online retraining.""")
with st.sidebar:
    st.header("Experiment controls")
    policies = st.multiselect("Algorithms", ["LRU", "LFU", "ML"], default=["LRU", "LFU", "ML"])
    workload = st.selectbox("Workload", WORKLOADS, index=1)
    capacity_kib = st.slider("Cache capacity (KiB)", 4, 256, 32, 4)
    requests = st.slider("Total requests (before split)", 600, 10000, 3000, 200)
    n_objects = st.slider("Catalog objects", 20, 300, 100, 10)
    seed = st.number_input("Random seed", 0, 100000, 42)
    seed_count = st.slider("Independent seeds", 1, 5, 1)
    horizon = st.slider("Future reuse horizon (requests)", 5, 100, 30, 5)
    hit_latency = st.number_input("Cache lookup / hit latency (ms)", 0.0, 20.0, 1.0)
    variable_sizes = st.checkbox("Vary object sizes (512–8192 bytes)", True)
    run = st.button("Run simulation", type="primary", use_container_width=True)

@st.cache_data(show_spinner=False)
def execute(kind, seeds, capacities, count, objects, h, latency, sizes, algorithms):
    return run_experiments((kind,), seeds, capacities, count, objects, h, latency, sizes, algorithms)

if run:
    if not policies:
        st.warning("Select at least one algorithm.")
    else:
        capacities = tuple(sorted({max(1024, capacity_kib * 512), capacity_kib * 1024, capacity_kib * 2048}))
        with st.spinner("Training on the prefix and simulating held-out requests…"):
            raw, predictive = execute(workload, tuple(range(seed, seed + seed_count)), capacities,
                                      requests, n_objects, horizon, hit_latency, variable_sizes, tuple(policies))
        st.session_state["result"] = (raw, predictive, workload, requests, n_objects, seed, variable_sizes, capacity_kib * 1024)

if "result" in st.session_state:
    raw, predictive, kind, count, objects, seed_used, sizes, capacity = st.session_state["result"]
    st.subheader(f"Results · {kind} · {count:,} total requests · {int(raw.requests.iloc[0]):,} held-out requests per run")
    st.caption("Results retain the settings from the last Run click. Charts show seed means; the table reports sample standard deviations (undefined for one seed).")
    selected = raw[raw.capacity_bytes == capacity]
    columns = st.columns(selected.policy.nunique())
    for column, (name, group) in zip(columns, selected.groupby("policy", sort=False)):
        with column:
            st.markdown(f"### {name}")
            st.metric("Hit rate", f"{group.hit_rate_pct.mean():.2f}%")
            st.metric("Average response", f"{group.average_response_ms.mean():.2f} ms")
            st.metric("Backend accesses", f"{group.backend_accesses.mean():.0f}")
    charts = st.columns(3)
    for column, metric, title in zip(charts, ["hit_rate_pct", "average_response_ms", "backend_accesses"],
                                      ["Hit rate (%)", "Average simulated response (ms)", "Backend accesses"]):
        with column:
            st.markdown(f"**{title}**")
            st.bar_chart(selected.groupby("policy")[metric].mean())
    st.subheader("Hit rate versus cache capacity")
    st.line_chart(raw.pivot_table(index="capacity_bytes", columns="policy", values="hit_rate_pct"), x_label="Capacity (bytes)", y_label="Hit rate (%)")
    st.subheader("Held-out workload popularity")
    test = chronological_split(generate_trace(kind, count, objects, seed_used, sizes))[2]
    popularity = pd.Series([r.object_id for r in test]).value_counts().sort_index()
    st.bar_chart(popularity.rename("Requests"), x_label="Object ID", y_label="Requests (first seed)")
    st.subheader("All metrics · mean and sample standard deviation")
    st.dataframe(aggregate(raw), use_container_width=True)
    st.subheader("Mean hit-rate winners (including ties)")
    st.dataframe(winners(raw), hide_index=True)
    if not predictive.empty:
        st.subheader("Reuse prediction on unseen partitions")
        st.dataframe(predictive, hide_index=True)
        st.caption("ROC-AUC is undefined for a single-class label set. Precision/recall/F1 use a 0.5 threshold; eviction uses continuous probabilities.")
    st.download_button("Download per-seed results (CSV)", raw.to_csv(index=False), "cache_results.csv", "text/csv")
else:
    st.markdown("Choose an experiment in the sidebar and select **Run simulation** to compare policies on the same held-out trace.")
