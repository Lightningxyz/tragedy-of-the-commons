"""
Scripted baseline agents for commons experiments.

These baselines make it easier to tell whether LLM behavior is genuinely
interesting or just a familiar threshold policy with nicer prose.
"""

from __future__ import annotations

import re

from agent import Action


def _parse_pool(env_summary: str) -> tuple[int, int]:
    match = re.search(r"Resource Pool:\s*(\d+)/(\d+)", env_summary)
    if not match:
        return 0, 1
    return int(match.group(1)), int(match.group(2))


class ScriptedAgent:
    def __init__(
        self,
        name: str,
        strategy: str,
        stake: str = "scripted baseline",
        extraction_weight: float = 1.0,
        demand_weight: float | None = None,
        alias: str | None = None,
    ):
        self.name = name
        self.alias = alias or name
        self.strategy = strategy
        self.persona = f"Scripted baseline strategy: {strategy}"
        self.provider = "scripted"
        self.model = strategy
        self.model_fallbacks: list[str] = []
        self.stake = stake
        self.extraction_weight = extraction_weight
        self.demand_weight = demand_weight

        self.inventory: int = 0
        self.history: list[dict] = []
        self.reputation: int = 0
        self.penalties_paid: int = 0
        self.last_model_used: str | None = strategy
        self.last_used_model_fallback: bool = False
        self.last_attempt_count: int = 0
        self.last_error: str | None = None

    async def decide(
        self,
        round_num: int,
        env_summary: str,
        agent_standings: str,
        recent_messages: dict[str, str],
        extraction_cap: int,
        institution_rules: str,
        institution_state: str,
    ) -> Action:
        pool, capacity = _parse_pool(env_summary)
        pct = pool / capacity if capacity else 0

        if self.strategy == "always_max":
            request = extraction_cap
            pledge = extraction_cap
            message = "I will take the maximum available share."
        elif self.strategy == "sustainable_share":
            request = max(0, min(extraction_cap, int(pool * 0.03)))
            pledge = request
            message = "I will stay near a conservative sustainable share."
        elif self.strategy == "tit_for_tat":
            previous_requests = [
                h["requested"] for h in self.history[-2:]
            ] or [max(1, extraction_cap // 2)]
            request = max(0, min(extraction_cap, round(sum(previous_requests) / len(previous_requests))))
            pledge = request
            message = "I match recent restraint and will punish persistent overuse by taking my share."
        elif self.strategy == "greedy_until_threshold":
            request = extraction_cap if pct > 0.5 else max(0, extraction_cap // 4)
            pledge = request
            message = "I extract aggressively while the pool is healthy, then slow down near the danger zone."
        elif self.strategy == "pledge_then_comply":
            pledge = max(0, extraction_cap // 2)
            request = pledge
            message = "I make a modest pledge and comply with it."
        else:
            request = 0 if pct < 0.25 else max(0, extraction_cap // 3)
            pledge = request
            message = "I prioritize preserving the commons over short-term extraction."

        return Action(
            private_scratchpad="Scripted agents do not use a scratchpad.",
            resource_request=request,
            reported_need=request,
            pledge_next_request=pledge,
            proposed_quota=pledge,
            accept_binding_quota=True,
            reasoning=f"Scripted {self.strategy} policy.",
            message=message,
        )

    def record_turn(
        self,
        round_num: int,
        requested: int,
        granted: int,
        pool_after: int,
        pledge: int | None = None,
        effective_request: int | None = None,
        penalty: int = 0,
        pledge_violation: int = 0,
        community_messages: dict[str, str] | None = None,
    ):
        self.inventory += granted
        self.inventory = max(0, self.inventory - penalty)
        self.penalties_paid += penalty
        if pledge_violation > 0:
            self.reputation -= 1
        self.history.append({
            "round": round_num,
            "requested": requested,
            "effective_request": effective_request if effective_request is not None else requested,
            "granted": granted,
            "pool_after": pool_after,
            "pledge_next_request": pledge,
            "penalty": penalty,
            "pledge_violation": pledge_violation,
            "community_messages": community_messages or {},
        })
