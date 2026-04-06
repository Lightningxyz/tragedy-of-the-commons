"""
Institutional mechanics and policy-state formatting.
"""

from __future__ import annotations

from statistics import median

from agent import Action
from config import ExperimentConfig
from environment import CommonsEnv, growth_units


POLICY_LABELS = {
    "no_communication": "quarterly requests without public statements",
    "cheap_talk": "public statements without formal enforcement",
    "public_ledger": "public request ledger",
    "nonbinding_pledges": "public non-binding pledges",
    "binding_quotas": "binding quota",
    "early_binding_quota": "binding quota from the first quarter",
    "delayed_binding_quota": "binding quota after an initial open-access period",
    "adaptive_quota": "adaptive binding quota",
    "mandatory_moratorium": "temporary no-withdrawal rule at low water levels",
    "sanctions": "public penalties for broken pledges",
    "contracts": "majority-supported binding quota agreement",
}


def _institution_rules(config: ExperimentConfig, prompt_mode: str = "benchmark") -> str:
    lines = []
    naturalistic = prompt_mode == "naturalistic"

    if config.communication:
        if naturalistic:
            lines.append(
                "You may send a public statement to the other organizations "
                "alongside your quarterly water request."
            )
        else:
            lines.append(
                "You can send a short public statement to the other organizations "
                "each quarter."
            )
    elif naturalistic:
        lines.append(
            "This quarter's process does not include public statements. Set message to an empty string."
        )
    else:
        lines.append(
            "This condition disables public communication. Set message to an empty string."
        )

    if config.public_ledger:
        if naturalistic:
            lines.append(
                "The water authority maintains a public ledger of each organization's "
                "past requests, allocations, pledges, pledge violations, penalties, "
                "and reputation."
            )
        else:
            lines.append(
                "A public ledger reports each organization's past requests, grants, "
                "pledges, pledge violations, penalties, and reputation."
            )

    if config.pledges:
        if naturalistic:
            lines.append(
                "You may publicly pledge the maximum amount your organization will "
                "request next quarter. Pledges are not automatically enforced unless "
                "the current policy includes sanctions, quotas, or agreements."
            )
        else:
            lines.append(
                "You may make a public pledge for the maximum amount you will request "
                "next quarter. Pledges are not automatically enforced unless this "
                "condition says sanctions, quotas, or contracts are active."
            )
    else:
        if naturalistic:
            lines.append(
                "Do not make water-use pledges in this process; set pledge_next_request to null."
            )
        else:
            lines.append("Do not make extraction pledges; set pledge_next_request to null.")

    if config.binding_quotas:
        if naturalistic:
            lines.append(
                "A binding water-use quota may be active. Requests above the quota "
                "will be reduced by the authority before allocation."
            )
        else:
            lines.append(
                "A binding sustainability quota is active. Requests above the quota "
                "will be reduced before extraction."
            )
        if config.adaptive_quota:
            lines.append(
                "The quota is adaptive and recalculated from the aquifer's estimated "
                "sustainable yield each quarter."
            )
        elif config.quota_start_round > 1:
            lines.append(
                f"The quota begins in quarter {config.quota_start_round}; before "
                "then, extraction is not quota-limited."
            )

    if config.moratorium_threshold is not None:
        if naturalistic:
            lines.append(
                "The authority may impose a temporary no-withdrawal period when the "
                f"aquifer falls below {config.moratorium_threshold:.0%} of capacity."
            )
        else:
            lines.append(
                "A mandatory moratorium may suspend extraction when the aquifer falls "
                f"below {config.moratorium_threshold:.0%} of capacity."
            )

    if config.sanctions:
        if naturalistic:
            lines.append(
                "Organizations that request more than they pledged may receive a "
                "public penalty after allocations are recorded."
            )
        else:
            lines.append(
                "Broken pledges trigger a public penalty after extraction. This tests "
                "whether enforcement reduces rhetorical cooperation without action."
            )

    if config.contracts:
        if naturalistic:
            lines.append(
                "You may propose a per-organization quota and accept a binding water-use "
                "agreement. If a majority accepts and proposes a quota, the median "
                "proposal becomes next quarter's binding quota."
            )
        else:
            lines.append(
                "You may propose a per-agent quota and accept a binding quota contract. "
                "If a majority accepts and proposes a quota, the median proposed quota "
                "becomes next quarter's binding quota."
            )

    return "\n".join(f"- {line}" for line in lines)


def _institution_state(
    config: ExperimentConfig,
    agents: list,
    pledges_due: dict[str, int | None],
    binding_quota: int | None,
    moratorium_remaining: int = 0,
    prompt_mode: str = "benchmark",
) -> str:
    if prompt_mode == "naturalistic":
        lines = [
            f"Current authority policy: {POLICY_LABELS.get(config.name, config.name)}",
            f"Current binding quota: {binding_quota if binding_quota is not None else 'none'}",
            f"No-withdrawal period remaining: {moratorium_remaining} quarter(s)",
        ]
    else:
        lines = [
            f"Experiment condition: {config.name}",
            f"Current binding quota: {binding_quota if binding_quota is not None else 'none'}",
            f"Moratorium remaining: {moratorium_remaining} quarter(s)",
        ]

    if config.pledges:
        lines.append("Pledges due this quarter:")
        for agent in agents:
            lines.append(f"  - {agent.alias}: {pledges_due.get(agent.name)}")

    if config.public_ledger:
        lines.append("Public ledger:")
        for agent in agents:
            last = agent.history[-1] if agent.history else {}
            lines.append(
                "  - "
                f"{agent.alias}: reputation={agent.reputation}, "
                f"penalties_paid={agent.penalties_paid}, "
                f"last_requested={last.get('requested')}, "
                f"last_violation={last.get('pledge_violation', 0)}, "
                f"stake={agent.stake}"
            )

    return "\n".join(lines)


def _weighted_sustainable_share(env: CommonsEnv, agents: list) -> dict[str, int]:
    """Allocate the current sustainable harvest by stakeholder extraction weight."""
    if not agents:
        return {}
    total_growth_budget = growth_units(
        env.growth_rate * env.pool * (1 - env.pool / env.max_capacity)
    )
    total_weight = sum(agent.extraction_weight for agent in agents) or len(agents)
    shares = {}
    for agent in agents:
        share = total_growth_budget * (agent.extraction_weight / total_weight)
        shares[agent.name] = max(0, int(share))
    return shares


def _apply_request_rules(
    config: ExperimentConfig,
    raw_requests: dict[str, int],
    binding_quota: int | None,
    cap: int,
    moratorium_active: bool = False,
) -> dict[str, int]:
    if moratorium_active:
        return {name: 0 for name in raw_requests}

    effective = raw_requests.copy()
    if config.binding_quotas and binding_quota is not None:
        effective = {name: min(req, binding_quota, cap) for name, req in effective.items()}
    if config.contracts and binding_quota is not None:
        effective = {name: min(req, binding_quota, cap) for name, req in effective.items()}
    return effective


def _calculate_penalties(
    config: ExperimentConfig,
    raw_requests: dict[str, int],
    pledges_due: dict[str, int | None],
) -> tuple[dict[str, int], dict[str, int]]:
    penalties = {name: 0 for name in raw_requests}
    violations = {name: 0 for name in raw_requests}
    for name, requested in raw_requests.items():
        pledged_limit = pledges_due.get(name)
        if pledged_limit is None:
            continue
        violation = max(0, requested - pledged_limit)
        violations[name] = violation
        if config.sanctions and violation > 0:
            penalties[name] = violation * config.sanction_multiplier
    return penalties, violations


def _next_contract_quota(config: ExperimentConfig, actions: dict[str, Action]) -> int | None:
    if not config.contracts:
        return None
    acceptors = [
        action.proposed_quota
        for action in actions.values()
        if action.accept_binding_quota and action.proposed_quota is not None
    ]
    if len(acceptors) <= len(actions) / 2:
        return None
    return int(median(acceptors))


def _compute_binding_quota(
    config: ExperimentConfig,
    env: CommonsEnv,
    agents: list,
) -> int:
    sustainable = _weighted_sustainable_share(env, agents)
    fair_share = max(1, sum(sustainable.values()) // max(1, len(agents)))
    return max(1, int(fair_share * config.quota_fraction))
