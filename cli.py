"""
CLI entrypoint for running commons simulations.
"""

from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv

load_dotenv()

import ui
from config import DEMAND_REGIMES, EXPERIMENTS, MAX_ROUNDS, REALISM_PROFILES, ROSTERS
from logging_utils import configure_logging
from simulation_core import run_simulation, run_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM commons experiments.")
    parser.add_argument(
        "--experiment",
        choices=[*EXPERIMENTS.keys(), "suite"],
        default="cheap_talk",
        help="Institutional condition to run.",
    )
    parser.add_argument(
        "--roster",
        choices=ROSTERS.keys(),
        default="heterogeneous",
        help="Model ecology / stakeholder roster.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["benchmark", "naturalistic"],
        default="benchmark",
        help="Use benchmark framing or more naturalistic stakeholder framing.",
    )
    parser.add_argument(
        "--realism",
        choices=REALISM_PROFILES.keys(),
        default="perfect",
        help="Observation realism profile.",
    )
    parser.add_argument(
        "--demand-regime",
        choices=DEMAND_REGIMES.keys(),
        default="medium",
        help="Private operational demand regime.",
    )
    parser.add_argument(
        "--need-visibility",
        choices=["private", "public", "audited"],
        default="private",
        help="Whether private needs are hidden, publicly stated, or authority-audited.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for realism noise and private demand shocks.",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Skip per-round delay.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM generation temperature (e.g. 0.0 for strict reproducibility).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of repeated runs.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=MAX_ROUNDS,
        help="Maximum number of quarters to run. Lower this for cheaper pilots.",
    )
    parser.add_argument(
        "--agent-timeout-seconds",
        type=float,
        default=60.0,
        help="Per-agent decision timeout before conservative fallback is applied.",
    )
    parser.add_argument(
        "--parallel-agent-calls",
        action="store_true",
        help="Query all agents concurrently. Disabled by default to reduce provider rate-limit pressure.",
    )
    parser.add_argument(
        "--agent-call-delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay between sequential agent calls. Ignored when --parallel-agent-calls is set.",
    )
    parser.add_argument(
        "--rotate-heterogeneous-models",
        action="store_true",
        help="In suite mode with heterogeneous roster, rotate model assignments across trials.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level for structured runtime logs.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to write logs (in addition to console).",
    )
    parser.add_argument(
        "--force-log-config",
        action="store_true",
        help="Force-reset existing logging handlers before applying CLI logging config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(
        level=args.log_level,
        log_file=args.log_file,
        force=args.force_log_config,
    )
    if args.experiment == "suite":
        run_suite(
            roster_name=args.roster,
            trials=args.trials,
            prompt_mode=args.prompt_mode,
            realism_name=args.realism,
            demand_regime_name=args.demand_regime,
            need_visibility=args.need_visibility,
            seed=args.seed,
            temperature=args.temperature,
            max_rounds=args.max_rounds,
            agent_timeout_seconds=args.agent_timeout_seconds,
            rotate_heterogeneous_models=args.rotate_heterogeneous_models,
            parallel_agent_calls=args.parallel_agent_calls,
            agent_call_delay_seconds=args.agent_call_delay_seconds,
        )
        return

    for trial in range(1, args.trials + 1):
        if args.trials > 1:
            ui.console.print(
                f"\n[bold]Trial {trial}/{args.trials}[/bold] - {args.experiment}"
            )
        run_simulation(
            config=EXPERIMENTS[args.experiment],
            roster_name=args.roster,
            prompt_mode=args.prompt_mode,
            realism_name=args.realism,
            demand_regime_name=args.demand_regime,
            need_visibility=args.need_visibility,
            seed=None if args.seed is None else args.seed + trial - 1,
            temperature=args.temperature,
            max_rounds=args.max_rounds,
            sleep_seconds=0 if args.no_sleep else 1,
            agent_timeout_seconds=args.agent_timeout_seconds,
            parallel_agent_calls=args.parallel_agent_calls,
            agent_call_delay_seconds=args.agent_call_delay_seconds,
        )


if __name__ == "__main__":
    main()
