"""Run with python -m adaptive_cache {train,experiment}."""
import argparse
import json
from pathlib import Path
import joblib
from .workload import WORKLOADS, generate_trace, chronological_split
from .model import train_model, predictive_metrics
from .features import FEATURE_NAMES
from .experiments import run_experiments, aggregate, winners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "experiment"))
    parser.add_argument("--requests", type=int, default=3000, help="Total requests before 50/20/30 chronological split")
    parser.add_argument("--objects", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--capacities", type=int, nargs="+", default=[16384, 32768, 65536], help="Bytes")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        train, val, test = chronological_split(generate_trace(args.workloads[0], args.requests, args.objects, args.seeds[0]))
        model, metrics = train_model(train, val, args.horizon, args.seeds[0])
        joblib.dump({"model": model, "horizon": args.horizon, "features": FEATURE_NAMES}, args.output / "model.joblib")
        report = {"validation": metrics, "test": predictive_metrics(model, test, args.horizon, args.seeds[0])}
        (args.output / "predictive_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    else:
        raw, predictive = run_experiments(args.workloads, args.seeds, args.capacities,
                                         args.requests, args.objects, args.horizon)
        raw.to_csv(args.output / "results.csv", index=False)
        predictive.to_csv(args.output / "predictive_metrics.csv", index=False)
        summary = aggregate(raw)
        summary.to_csv(args.output / "summary.csv")
        best = winners(raw)
        best.to_csv(args.output / "winners.csv", index=False)
        (args.output / "configuration.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
        from .plots import save_plots
        save_plots(raw, args.output)
        print(summary.to_string())
        print("\nMean hit-rate winners (ties retained):\n" + best.to_string(index=False))
        print("\nAll latencies are simulated; policy overhead is separately measured wall time.")

if __name__ == "__main__":
    main()
