"""Test whether C13 separates pilot events beyond a visual radar-echo proxy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara radar RGB e radar+C13 no piloto por eventos; não é avaliação preditiva."
    )
    parser.add_argument("--station-signal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radar-low-echo-threshold", type=float, default=0.05)
    return parser.parse_args()


def out_of_fold_scores(features: np.ndarray, labels: np.ndarray, groups: np.ndarray | None) -> dict[str, float]:
    """Return event-level OOF AUC/AP using folds that never split station records."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    smallest_class = int(np.bincount(labels).min())
    folds = min(5, smallest_class)
    if folds < 2:
        raise ValueError("A análise requer pelo menos dois eventos em cada classe.")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1_000, random_state=42))
    if groups is not None and len(np.unique(groups)) >= folds:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
        split_arguments = {"groups": groups}
        split_name = "StratifiedGroupKFold(pair_id)"
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        split_arguments = {}
        split_name = "StratifiedKFold(event)"
    probabilities = cross_val_predict(
        model, features, labels,
        cv=splitter,
        method="predict_proba",
        **split_arguments,
    )[:, 1]
    return {
        "roc_auc_oof": float(roc_auc_score(labels, probabilities)),
        "average_precision_oof": float(average_precision_score(labels, probabilities)),
        "folds": folds, "splitter": split_name,
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.radar_low_echo_threshold <= 1.0:
        raise ValueError("--radar-low-echo-threshold deve estar entre zero e um.")
    source = pd.read_csv(args.station_signal)
    required = {"event_id", "kind", "complete", "c13_kelvin", "radar_echo_fraction", "m15_mm_15min"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"station_signal sem colunas requeridas: {sorted(missing)}")
    records = source.loc[source["complete"]].dropna(subset=["c13_kelvin", "radar_echo_fraction"])
    grouping = ["event_id", "pair_id", "kind"] if "pair_id" in records else ["event_id", "kind"]
    event_features = records.groupby(grouping, as_index=False, dropna=False).agg(
        c13_station_mean_kelvin=("c13_kelvin", "mean"),
        c13_station_min_kelvin=("c13_kelvin", "min"),
        radar_echo_fraction_mean=("radar_echo_fraction", "mean"),
        radar_echo_fraction_max=("radar_echo_fraction", "max"),
        target_max_m15_mm_15min=("m15_mm_15min", "max"),
        stations=("m15_mm_15min", "size"),
    )
    event_features["is_wet"] = (event_features["kind"] == "wet").astype(np.int8)
    if set(event_features["is_wet"]) != {0, 1}:
        raise ValueError("São necessários eventos wet e dry_control completos.")
    labels = event_features["is_wet"].to_numpy()
    groups = event_features["pair_id"].to_numpy() if "pair_id" in event_features and event_features["pair_id"].notna().all() else None
    radar = event_features[["radar_echo_fraction_mean", "radar_echo_fraction_max"]].to_numpy()
    joint = event_features[[
        "radar_echo_fraction_mean", "radar_echo_fraction_max",
        "c13_station_mean_kelvin", "c13_station_min_kelvin",
    ]].to_numpy()
    results = {
        "scope": "event-audit-only; not predictive evaluation; controls and wet events were selected with observed targets",
        "events": int(len(event_features)),
        "wet_events": int(labels.sum()),
        "dry_control_events": int((labels == 0).sum()),
        "unit_of_analysis": "event; station records are aggregated before cross-validation",
        "radar_features": ["radar_echo_fraction_mean", "radar_echo_fraction_max"],
        "goes_features": ["c13_station_mean_kelvin", "c13_station_min_kelvin"],
        "radar_only": out_of_fold_scores(radar, labels, groups),
        "radar_plus_c13": out_of_fold_scores(joint, labels, groups),
    }
    threshold = args.radar_low_echo_threshold
    overlap = event_features.loc[event_features["radar_echo_fraction_mean"] <= threshold]
    overlap_counts = overlap.groupby("kind").size().to_dict()
    results["conditional_radar_low_echo"] = {
        "threshold": threshold,
        "events": int(len(overlap)),
        "wet_events": int(overlap_counts.get("wet", 0)),
        "dry_control_events": int(overlap_counts.get("dry_control", 0)),
        "interpretation": (
            "C13 can only be assessed conditionally in this stratum when both classes are present; "
            "otherwise the pilot has no radar-matched support for an incremental-value claim."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_features.sort_values(["kind", "event_id"]).to_csv(args.output_dir / "event_features.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "Análise condicional GOES concluída | "
        f"eventos={results['events']} | AUC radar={results['radar_only']['roc_auc_oof']:.3f} | "
        f"AUC radar+C13={results['radar_plus_c13']['roc_auc_oof']:.3f} | saída={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
