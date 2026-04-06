"""
Aggregate commons simulation JSON exports into a researcher-friendly CSV.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from statistics import mean, stdev
from typing import Any

from metrics import summarize_run


SUMMARY_FIELDS = [
    "experiment",
    "roster",
    "prompt_mode",
    "realism",
    "demand_regime",
    "need_visibility",
    "n_runs",
    "survival_rate",
    "mean_rounds_completed",
    "sd_rounds_completed",
    "mean_final_pool_pct",
    "mean_total_harvested",
    "mean_inventory_gini",
    "mean_hypocrisy_rate",
    "mean_pledge_strength",
    "mean_pledge_sufficiency_rate",
    "mean_pledge_compliance_rate",
    "mean_first_sufficient_pledge_round",
    "mean_late_governance_index",
    "mean_empty_governance_score",
    "pledge_policy_survival_rate",
    "mean_pledge_policy_latest_saving_round",
    "failure_modes",
    "mean_broken_pledges",
    "mean_total_penalties",
    "mean_message_cooperation_score",
    "mean_request_to_sustainable_share",
    "mean_effective_request_to_sustainable_share",
    "mean_request_to_need_ratio",
    "mean_pledge_to_need_ratio",
    "mean_need_satisfaction_rate",
    "mean_over_need_request_rate",
    "mean_under_need_pledge_rate",
    "mean_scarcity_pressure",
    "demand_feasibility_labels",
    "mean_demand_feasibility_rate",
    "mean_need_to_safe_yield",
    "mean_request_to_safe_yield",
    "mean_pledge_to_safe_yield",
    "mean_need_safe_gap",
    "mean_request_safe_gap",
    "mean_pledge_safe_gap",
    "policy_pressure_modes",
    "mean_model_fallback_turns",
    "mean_provider_fallback_turns",
]


def _load_runs(
    results_dir: str,
    max_provider_fallback_rate: float | None = None,
) -> list[dict[str, Any]]:
    runs = []
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            data = json.load(f)
        if "metrics" not in data or "experiment" not in data:
            continue
        metrics = data.get("metrics", {})
        if (
            (
                "failure_mode" not in metrics
                or "pledge_sufficiency_rate" not in metrics
                or "mean_request_to_need_ratio" not in metrics
                or "mean_request_to_safe_yield" not in metrics
                or "demand_feasibility_label" not in metrics
            )
            and data.get("turns")
            and data.get("config")
        ):
            data["metrics"] = summarize_run(
                data["turns"],
                data.get("agents", {}),
                data["config"],
                data["config"].get("max_rounds", len(data["turns"])),
                data.get("metrics", {}).get("final_pool", 0),
                data["config"].get("max_capacity", 1),
            )
        if max_provider_fallback_rate is not None:
            total_agent_turns = sum(len(turn.get("requested", {})) for turn in data.get("turns", []))
            provider_fallbacks = data.get("metrics", {}).get("provider_fallback_turns", 0)
            fallback_rate = provider_fallbacks / total_agent_turns if total_agent_turns else 0
            if fallback_rate > max_provider_fallback_rate:
                continue
        runs.append({"path": path, **data})
    return runs


def _sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        experiment = run.get("experiment", {}).get("name", "legacy")
        roster = run.get("roster", "legacy")
        prompt_mode = run.get("prompt_mode", "legacy")
        realism = run.get("realism", {}).get("name", "legacy")
        demand_regime = run.get("demand_regime", {}).get("name", "legacy")
        need_visibility = run.get("need_visibility", "legacy")
        grouped[
            (
                experiment,
                roster,
                prompt_mode,
                realism,
                demand_regime,
                need_visibility,
            )
        ].append(run)

    rows = []
    for (
        experiment,
        roster,
        prompt_mode,
        realism,
        demand_regime,
        need_visibility,
    ), group in sorted(grouped.items()):
        metrics = [run["metrics"] for run in group]
        rounds = [m.get("rounds_completed", 0) for m in metrics]
        rows.append({
            "experiment": experiment,
            "roster": roster,
            "prompt_mode": prompt_mode,
            "realism": realism,
            "demand_regime": demand_regime,
            "need_visibility": need_visibility,
            "n_runs": len(group),
            "survival_rate": mean(1 if m.get("survived") else 0 for m in metrics),
            "mean_rounds_completed": mean(rounds),
            "sd_rounds_completed": _sd(rounds),
            "mean_final_pool_pct": mean(m.get("final_pool_pct", 0) for m in metrics),
            "mean_total_harvested": mean(m.get("total_harvested", 0) for m in metrics),
            "mean_inventory_gini": mean(m.get("inventory_gini", 0) for m in metrics),
            "mean_hypocrisy_rate": mean(m.get("hypocrisy_rate", 0) for m in metrics),
            "mean_pledge_strength": mean(m.get("pledge_strength", 0) for m in metrics),
            "mean_pledge_sufficiency_rate": mean(
                m.get("pledge_sufficiency_rate", 0) for m in metrics
            ),
            "mean_pledge_compliance_rate": mean(
                m.get("pledge_compliance_rate", 0) for m in metrics
            ),
            "mean_first_sufficient_pledge_round": mean(
                m.get("first_sufficient_pledge_round") or 0 for m in metrics
            ),
            "mean_late_governance_index": mean(
                m.get("late_governance_index", 0) for m in metrics
            ),
            "mean_empty_governance_score": mean(
                m.get("empty_governance_score", 0) for m in metrics
            ),
            "pledge_policy_survival_rate": mean(
                1 if m.get("pledge_policy_survives_from_start") else 0
                for m in metrics
            ),
            "mean_pledge_policy_latest_saving_round": mean(
                m.get("pledge_policy_latest_saving_round") or 0
                for m in metrics
            ),
            "failure_modes": ";".join(
                sorted({str(m.get("failure_mode", "unknown")) for m in metrics})
            ),
            "mean_broken_pledges": mean(m.get("broken_pledges", 0) for m in metrics),
            "mean_total_penalties": mean(m.get("total_penalties", 0) for m in metrics),
            "mean_message_cooperation_score": mean(
                m.get("mean_message_cooperation_score", 0) for m in metrics
            ),
            "mean_request_to_sustainable_share": mean(
                m.get("mean_request_to_sustainable_share", 0) for m in metrics
            ),
            "mean_effective_request_to_sustainable_share": mean(
                m.get(
                    "mean_effective_request_to_sustainable_share",
                    m.get("mean_request_to_sustainable_share", 0),
                )
                for m in metrics
            ),
            "mean_request_to_need_ratio": mean(
                m.get("mean_request_to_need_ratio", 0) for m in metrics
            ),
            "mean_pledge_to_need_ratio": mean(
                m.get("mean_pledge_to_need_ratio", 0) for m in metrics
            ),
            "mean_need_satisfaction_rate": mean(
                m.get("need_satisfaction_rate", 0) for m in metrics
            ),
            "mean_over_need_request_rate": mean(
                m.get("over_need_request_rate", 0) for m in metrics
            ),
            "mean_under_need_pledge_rate": mean(
                m.get("under_need_pledge_rate", 0) for m in metrics
            ),
            "mean_scarcity_pressure": mean(
                m.get("scarcity_pressure", 0) for m in metrics
            ),
            "demand_feasibility_labels": ";".join(
                sorted({str(m.get("demand_feasibility_label", "unknown")) for m in metrics})
            ),
            "mean_demand_feasibility_rate": mean(
                m.get("demand_feasibility_rate", 0) for m in metrics
            ),
            "mean_need_to_safe_yield": mean(
                m.get("mean_need_to_safe_yield", 0) for m in metrics
            ),
            "mean_request_to_safe_yield": mean(
                m.get("mean_request_to_safe_yield", 0) for m in metrics
            ),
            "mean_pledge_to_safe_yield": mean(
                m.get("mean_pledge_to_safe_yield", 0) for m in metrics
            ),
            "mean_need_safe_gap": mean(
                m.get("mean_need_safe_gap", 0) for m in metrics
            ),
            "mean_request_safe_gap": mean(
                m.get("mean_request_safe_gap", 0) for m in metrics
            ),
            "mean_pledge_safe_gap": mean(
                m.get("mean_pledge_safe_gap", 0) for m in metrics
            ),
            "policy_pressure_modes": ";".join(
                sorted(
                    {
                        label
                        for m in metrics
                        for label in m.get("policy_pressure_counts", {}).keys()
                    }
                )
            ),
            "mean_model_fallback_turns": mean(
                m.get("model_fallback_turns", 0) for m in metrics
            ),
            "mean_provider_fallback_turns": mean(
                m.get("provider_fallback_turns", 0) for m in metrics
            ),
        })
    return rows


def write_csv(rows: list[dict[str, Any]], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, Any]]):
    for row in rows:
        print(
            f"{row['experiment']:<24} {row['roster']:<20} "
            f"{row['prompt_mode']:<12} "
            f"{row['realism']:<8} "
            f"{row['demand_regime']:<7} "
            f"{row['need_visibility']:<7} "
            f"n={row['n_runs']:<3} survival={row['survival_rate']:.2f} "
            f"rounds={row['mean_rounds_completed']:.1f} "
            f"gini={row['mean_inventory_gini']:.2f} "
            f"mode={row['failure_modes']} "
            f"fallback={row['mean_provider_fallback_turns']:.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate commons simulation results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/summary.csv")
    parser.add_argument(
        "--max-provider-fallback-rate",
        type=float,
        default=None,
        help="Exclude runs whose provider fallback rate exceeds this threshold.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = aggregate_runs(_load_runs(args.results_dir, args.max_provider_fallback_rate))
    write_csv(rows, args.out)
    print_table(rows)
    print(f"\nWrote {args.out}")
