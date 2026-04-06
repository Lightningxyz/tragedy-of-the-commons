"""
Generate a human-readable Markdown report and SVG plot for one simulation run.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
from typing import Any

from metrics import summarize_run

logger = logging.getLogger(__name__)


def _latest_result(results_dir: str) -> str:
    paths = [
        os.path.join(results_dir, name)
        for name in os.listdir(results_dir)
        if name.endswith(".json")
    ]
    if not paths:
        raise FileNotFoundError(f"No JSON results found in {results_dir}")
    return max(paths, key=os.path.getmtime)


def _ensure_metrics(data: dict[str, Any]) -> dict[str, Any]:
    metrics = data.get("metrics", {})
    if (
        ("failure_mode" not in metrics or "mean_request_to_need_ratio" not in metrics)
        and data.get("turns")
    ):
        data["metrics"] = summarize_run(
            data["turns"],
            data.get("agents", {}),
            data["config"],
            data["config"].get("max_rounds", len(data["turns"])),
            data.get("metrics", {}).get("final_pool", 0),
            data["config"].get("max_capacity", 1),
        )
    return data


def _series(turns: list[dict[str, Any]], key: str) -> list[int]:
    if key == "pool":
        return [turn["pool_after"] for turn in turns]
    if key == "requested":
        return [sum(turn.get("requested", {}).values()) for turn in turns]
    if key == "effective":
        return [sum(turn.get("effective_requested", {}).values()) for turn in turns]
    if key == "pledged":
        values = []
        for turn in turns:
            pledges = [p for p in turn.get("pledges_next", {}).values() if p is not None]
            values.append(sum(pledges))
        return values
    return []


def write_svg_plot(data: dict[str, Any], out_path: str):
    turns = data.get("turns", [])
    if not turns:
        return

    width, height = 900, 420
    margin = 56
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    max_y = max(
        data.get("config", {}).get("max_capacity", 120),
        *(_series(turns, "requested") or [0]),
        *(_series(turns, "effective") or [0]),
        *(_series(turns, "pledged") or [0]),
    )
    max_y = max(max_y, 1)
    max_x = max(len(turns) - 1, 1)

    def point(idx: int, value: float) -> tuple[float, float]:
        x = margin + (idx / max_x) * plot_w
        y = height - margin - (value / max_y) * plot_h
        return x, y

    def polyline(values: list[int], color: str) -> str:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, v) for i, v in enumerate(values)))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3" />'

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420" viewBox="0 0 900 420">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222" />',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222" />',
        f'<text x="{margin}" y="30" font-family="Arial" font-size="18" font-weight="bold">Commons Run: Pool, Requests, Pledges</text>',
    ]

    # Y-axis tick labels and gridlines
    num_y_ticks = 5
    for i in range(num_y_ticks + 1):
        tick_val = int(max_y * i / num_y_ticks)
        _, y_pos = point(0, tick_val)
        lines.append(
            f'<text x="{margin-8}" y="{y_pos+4:.1f}" font-family="Arial" '
            f'font-size="10" text-anchor="end">{tick_val}</text>'
        )
        if i > 0:
            lines.append(
                f'<line x1="{margin}" y1="{y_pos:.1f}" x2="{width-margin}" '
                f'y2="{y_pos:.1f}" stroke="#ddd" stroke-dasharray="4,4" />'
            )

    lines.extend([
        polyline(_series(turns, "pool"), "#0f766e"),
        polyline(_series(turns, "requested"), "#dc2626"),
        polyline(_series(turns, "effective"), "#2563eb"),
        polyline(_series(turns, "pledged"), "#9333ea"),
    ])
    legend = [
        ("pool", "#0f766e"),
        ("requested", "#dc2626"),
        ("effective", "#2563eb"),
        ("pledged next", "#9333ea"),
    ]
    for i, (label, color) in enumerate(legend):
        x = margin + i * 170
        y = height - 15
        lines.append(f'<line x1="{x}" y1="{y}" x2="{x+24}" y2="{y}" stroke="{color}" stroke-width="3" />')
        lines.append(f'<text x="{x+30}" y="{y+5}" font-family="Arial" font-size="13">{label}</text>')

    for idx, turn in enumerate(turns):
        x, _ = point(idx, 0)
        lines.append(f'<text x="{x-4:.1f}" y="{height-margin+18}" font-family="Arial" font-size="10">{turn["round"]}</text>')
    lines.append("</svg>")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def _collapse_window(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not turns:
        return []
    if turns[-1].get("pool_after", 1) <= 0:
        return turns[max(0, len(turns) - 3):]
    return turns[-3:]


def write_markdown_report(data: dict[str, Any], source_path: str, out_path: str, svg_path: str):
    metrics = data["metrics"]
    exp = data.get("experiment", {}).get("name", "unknown")
    roster = data.get("roster", "unknown")
    prompt_mode = data.get("prompt_mode", "legacy")
    realism = data.get("realism", {}).get("name", "legacy")
    demand_regime = data.get("demand_regime", {}).get("name", "legacy")
    need_visibility = data.get("need_visibility", "legacy")

    lines = [
        f"# Commons Run Report",
        "",
        f"- Source: `{source_path}`",
        f"- Experiment: `{exp}`",
        f"- Roster: `{roster}`",
        f"- Prompt mode: `{prompt_mode}`",
        f"- Realism: `{realism}`",
        f"- Demand regime: `{demand_regime}`",
        f"- Need visibility: `{need_visibility}`",
        f"- Outcome: `{data.get('outcome', 'unknown')}`",
        f"- Failure mode: `{metrics.get('failure_mode', 'unknown')}`",
        "",
        f"![Run plot]({os.path.basename(svg_path)})",
        "",
        "## Key Metrics",
        "",
        f"- Rounds completed: {metrics.get('rounds_completed')}",
        f"- Final pool: {metrics.get('final_pool')} ({metrics.get('final_pool_pct', 0):.2%})",
        f"- Total harvested: {metrics.get('total_harvested')}",
        f"- Inventory Gini: {metrics.get('inventory_gini', 0):.3f}",
        f"- Pledge compliance rate: {metrics.get('pledge_compliance_rate', 0):.2%}",
        f"- Pledge strength: {metrics.get('pledge_strength', 0):.2f}",
        f"- Empty governance score: {metrics.get('empty_governance_score', 0):.2f}",
        f"- Request-to-need ratio: {metrics.get('mean_request_to_need_ratio', 0):.2f}",
        f"- Pledge-to-need ratio: {metrics.get('mean_pledge_to_need_ratio', 0):.2f}",
        f"- Need satisfaction rate: {metrics.get('need_satisfaction_rate', 0):.2%}",
        f"- Over-need request rate: {metrics.get('over_need_request_rate', 0):.2%}",
        f"- Under-need pledge rate: {metrics.get('under_need_pledge_rate', 0):.2%}",
        f"- Scarcity pressure: {metrics.get('scarcity_pressure', 0):.2f}",
        f"- Demand feasibility: {metrics.get('demand_feasibility_label', 'unknown')} ({metrics.get('demand_feasibility_rate', 0):.2%})",
        f"- Request-to-safe-yield ratio: {metrics.get('mean_request_to_safe_yield', 0):.2f}",
        f"- Need-to-safe-yield ratio: {metrics.get('mean_need_to_safe_yield', 0):.2f}",
        f"- Pledge-to-safe-yield ratio: {metrics.get('mean_pledge_to_safe_yield', 0):.2f}",
        f"- Pledge policy survives from start: {metrics.get('pledge_policy_survives_from_start')}",
        f"- Latest saving round under pledge policy: {metrics.get('pledge_policy_latest_saving_round')}",
        "",
        "## Demand Audit",
        "",
        "| Agent | Mean Need | Mean Request | Mean Granted | Mean Pledge | Request/Need | Grant/Need | Over-Need Requests | Under-Need Pledges |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, profile in sorted(metrics.get("agent_demand_profiles", {}).items()):
        lines.append(
            f"| {name} | "
            f"{profile.get('mean_private_need', 0):.2f} | "
            f"{profile.get('mean_request', 0):.2f} | "
            f"{profile.get('mean_granted', 0):.2f} | "
            f"{profile.get('mean_pledge', 0):.2f} | "
            f"{profile.get('request_to_need_ratio', 0):.2f} | "
            f"{profile.get('grant_to_need_ratio', 0):.2f} | "
            f"{profile.get('over_need_request_rate', 0):.2%} | "
            f"{profile.get('under_need_pledge_rate', 0):.2%} |"
        )

    lines.extend([
        "",
        "## Policy Adequacy",
        "",
        "| Round | Pool Before | Safe Yield | Need | Requested | Effective | Granted | Pledged | Need-Safe Gap | Request-Safe Gap | Pledge-Safe Gap | Classification |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])

    for row in metrics.get("policy_adequacy_rounds", []):
        pledged = row.get("total_pledged")
        pledge_gap = row.get("pledge_safe_gap")
        lines.append(
            f"| {row.get('round')} | "
            f"{row.get('pool_before')} | "
            f"{row.get('safe_yield')} | "
            f"{row.get('total_private_need')} | "
            f"{row.get('total_requested')} | "
            f"{row.get('total_effective')} | "
            f"{row.get('total_granted')} | "
            f"{'' if pledged is None else pledged} | "
            f"{row.get('need_safe_gap')} | "
            f"{row.get('request_safe_gap')} | "
            f"{'' if pledge_gap is None else pledge_gap} | "
            f"{row.get('classification')} |"
        )

    lines.extend([
        "",
        "## Per-Round Table",
        "",
        "| Round | Pool Before | Need | Requested | Effective | Granted | Pledged Next | Pool After |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for turn in data.get("turns", []):
        total_need = sum(
            obs.get("private_need", 0) or 0
            for obs in turn.get("observations", {}).values()
        )
        lines.append(
            f"| {turn['round']} | {turn['pool_before']} | "
            f"{total_need} | "
            f"{sum(turn.get('requested', {}).values())} | "
            f"{sum(turn.get('effective_requested', {}).values())} | "
            f"{sum(turn.get('granted', {}).values())} | "
            f"{sum(p for p in turn.get('pledges_next', {}).values() if p is not None)} | "
            f"{turn['pool_after']} |"
        )

    lines.extend(["", "## Collapse Window / Final Messages", ""])
    for turn in _collapse_window(data.get("turns", [])):
        lines.append(f"### Round {turn['round']}")
        for name, message in turn.get("messages", {}).items():
            if message:
                lines.append(f"- **{name}:** {html.escape(message)}")
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        _interpretation(metrics),
        "",
    ])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def _interpretation(metrics: dict[str, Any]) -> str:
    mode = metrics.get("failure_mode")
    if mode == "weak_pledge_failure":
        return (
            "The run collapsed even though pledge compliance was high. This suggests "
            "the commitments were too weak relative to the aquifer's sustainable yield. "
            "Use the demand audit and policy adequacy table to separate need pressure, "
            "strategic over-requesting, and ecologically insufficient pledges."
        )
    if mode == "deception_failure":
        return "The run collapsed with broken pledges, suggesting enforcement or credibility failure."
    if mode == "late_governance_failure":
        return "Governance appeared after the latest counterfactual saving round."
    if mode == "empty_governance_failure":
        return "The agents produced cooperative governance language, but it did not translate into sustainability."
    if mode == "sustained":
        return "The commons survived under this condition."
    return "The run requires qualitative inspection; the classifier did not assign a sharper failure mode."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a report for one commons run.")
    parser.add_argument("path", nargs="?", help="Result JSON path. Defaults to latest result.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="reports")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source_path = args.path or _latest_result(args.results_dir)
    with open(source_path) as f:
        data = _ensure_metrics(json.load(f))
    stem = os.path.splitext(os.path.basename(source_path))[0]
    svg_path = os.path.join(args.out_dir, f"{stem}.svg")
    md_path = os.path.join(args.out_dir, f"{stem}.md")
    write_svg_plot(data, svg_path)
    write_markdown_report(data, source_path, md_path, svg_path)
    logger.info("report_generated path=%s", md_path)
