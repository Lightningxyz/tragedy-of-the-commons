"""
Compatibility module re-exporting simulation APIs.

This keeps older imports stable while implementation is split across:
- config.py
- institutions.py
- observations.py
- simulation_core.py
- cli.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from config import (
    AGENT_ROSTER,
    DEMAND_REGIMES,
    EXPERIMENTS,
    GROWTH_RATE,
    INITIAL_POOL,
    MAX_CAPACITY,
    MAX_HARVEST_PCT,
    MAX_ROUNDS,
    REALISM_PROFILES,
    ROSTERS,
    DemandRegime,
    ExperimentConfig,
    RealismConfig,
    homogeneous_roster,
    rotated_model_roster,
)
from institutions import (
    POLICY_LABELS,
    _apply_request_rules,
    _calculate_penalties,
    _compute_binding_quota,
    _institution_rules,
    _institution_state,
    _next_contract_quota,
    _weighted_sustainable_share,
)
from observations import _demand_multiplier, _need_visibility_block, _observed_env_summary
from simulation_core import (
    _build_agent_rngs,
    _build_standings,
    _make_agents,
    _query_agent,
    _timeout_fallback_action,
    run_simulation,
    run_suite,
)


def _parse_args():
    from cli import parse_args

    return parse_args()


if __name__ == "__main__":
    from cli import main

    main()
