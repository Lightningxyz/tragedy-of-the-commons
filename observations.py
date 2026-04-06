"""
Observation and demand pipelines used by simulation orchestration.
"""

from __future__ import annotations

import random

from config import DemandRegime, RealismConfig
from environment import CommonsEnv


def _demand_multiplier(regime: DemandRegime, round_num: int) -> float:
    if regime.name == "none":
        return 0.0
    if (
        regime.crisis_rounds
        and regime.crisis_multiplier is not None
        and round_num <= regime.crisis_rounds
    ):
        return regime.crisis_multiplier
    if regime.crisis_rounds and regime.post_crisis_multiplier is not None:
        return regime.post_crisis_multiplier
    return regime.multiplier


def _need_visibility_block(
    agents: list,
    recent_reported_needs: dict[str, int | None],
    observations: dict[str, dict],
    need_visibility: str,
    prompt_mode: str,
) -> str:
    if need_visibility == "private":
        return ""

    label = (
        "Verified operational need estimates this quarter"
        if need_visibility == "audited"
        else "Stated operational need estimates from last quarter"
    )
    if prompt_mode != "naturalistic":
        label = (
            "Audited private demand estimates this round"
            if need_visibility == "audited"
            else "Reported public demand estimates from last round"
        )

    lines = [f"\n{label}:"]

    if need_visibility == "audited":
        for agent in agents:
            observation = observations.get(agent.name, {})
            need = observation.get("private_need")
            lines.append(f"  - {agent.alias}: {'unknown' if need is None else need}")
    else:
        for agent in agents:
            rep_need = recent_reported_needs.get(agent.name)
            lines.append(f"  - {agent.alias}: {'unknown' if rep_need is None else rep_need}")

    return "\n".join(lines) + "\n"


def _observed_env_summary(
    env: CommonsEnv,
    prompt_mode: str,
    realism: RealismConfig,
    demand_regime: DemandRegime,
    round_num: int,
    agent,
    rng: random.Random,
    turn_exports: list[dict],
) -> tuple[str, dict]:
    true_pool = env.pool
    reported_pool = true_pool
    source = "current"

    if realism.report_delay_rounds and len(turn_exports) >= realism.report_delay_rounds:
        reported_pool = turn_exports[-realism.report_delay_rounds]["pool_after"]
        source = f"{realism.report_delay_rounds}_round_delay"

    noisy_pool = reported_pool
    if realism.observation_noise_pct:
        noise = rng.uniform(-realism.observation_noise_pct, realism.observation_noise_pct)
        noisy_pool = round(reported_pool * (1 + noise))
    noisy_pool = max(0, min(env.max_capacity, int(noisy_pool)))

    cap = env.per_agent_cap
    cap_label = (
        "Per-Organization Request Cap This Quarter"
        if prompt_mode == "naturalistic"
        else "Per-Agent Extraction Cap This Round"
    )
    lines = [
        f"Resource Pool: {noisy_pool}/{env.max_capacity}",
        f"{cap_label}: {cap}",
        f"Growth Rate: {env.growth_rate:.0%}",
    ]

    private_need = None
    demand_multiplier = _demand_multiplier(demand_regime, round_num)
    if demand_multiplier > 0:
        demand_weight = getattr(
            agent,
            "demand_weight",
            min(1.0, getattr(agent, "extraction_weight", 1.0)),
        )
        if demand_weight is None:
            demand_weight = min(1.0, getattr(agent, "extraction_weight", 1.0))
        # Anchor need to INITIAL pool cap so private need does not shrink
        # as the aquifer depletes (which would be ecologically backwards).
        initial_cap = env.initial_per_agent_cap
        base_need = max(1, round(initial_cap * demand_weight * demand_multiplier))
        shock = rng.uniform(
            1 - realism.private_demand_shock_pct,
            1 + realism.private_demand_shock_pct,
        )
        private_need = max(0, round(base_need * shock))
        if prompt_mode == "naturalistic":
            lines.append(f"Your private operational need estimate this quarter: {private_need}")
        else:
            lines.append(f"Private demand shock this round: need estimate = {private_need}")

    return "\n".join(lines) + "\n", {
        "true_pool": true_pool,
        "reported_pool": reported_pool,
        "observed_pool": noisy_pool,
        "observation_source": source,
        "observed_cap": cap,
        "private_need": private_need,
        "demand_regime": demand_regime.name,
        "demand_multiplier": demand_multiplier,
    }
