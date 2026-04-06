"""
Tragedy of the Commons — Environment Module

Implements the shared resource pool with logistic-growth replenishment,
per-agent extraction caps, and full turn history tracking.
"""

from __future__ import annotations
from dataclasses import dataclass


def growth_units(growth: float) -> int:
    """Convert continuous logistic growth to integer units.

    The 1e-9 epsilon guards against floating-point boundary cases where
    growth is infinitesimally below an integer (e.g. 4.9999999999 -> 5).
    Since growth is always non-negative from the logistic formula when
    pool > 0 and pool < capacity, the max(0, ...) is a defensive clamp.
    """
    return max(0, int(growth + 1e-9))


@dataclass
class TurnResult:
    """Immutable record of a single round's outcome."""
    round_num: int
    requested: dict[str, int]
    granted: dict[str, int]
    remaining_before: int
    remaining_after: int
    replenished: int
    depleted: bool


class CommonsEnv:
    """
    Shared resource pool with logistic-growth replenishment.

    The replenishment follows r * S * (1 - S/K) where:
      r = intrinsic growth rate
      S = current stock
      K = carrying capacity (max_capacity)

    This creates a natural "sweet spot" — moderate harvesting allows
    the resource to regenerate, but over-extraction crashes it to zero.
    """

    def __init__(
        self,
        initial_pool: int = 100,
        max_capacity: int = 120,
        growth_rate: float = 0.3,
        max_harvest_pct: float = 0.25,
        num_agents: int = 3,
    ):
        self.pool = initial_pool
        self._initial_pool = initial_pool
        self.max_capacity = max_capacity
        self.growth_rate = growth_rate
        self.max_harvest_pct = max_harvest_pct
        self.num_agents = num_agents
        self.turn_history: list[TurnResult] = []
        self._round = 0

    # ------------------------------------------------------------------
    @property
    def per_agent_cap(self) -> int:
        """Max any single agent may extract in one turn."""
        return max(1, int(self.pool * self.max_harvest_pct))

    @property
    def initial_per_agent_cap(self) -> int:
        """Per-agent cap anchored to the initial pool size (stable across rounds).

        Used for demand calculations to avoid coupling private need to
        current pool state, which would create an unrealistic feedback loop
        where agents need less water as the aquifer depletes.
        """
        return max(1, int(self._initial_pool * self.max_harvest_pct))

    # ------------------------------------------------------------------
    def _allocate_proportionally(
        self,
        clamped: dict[str, int],
        total_requested: int,
    ) -> dict[str, int]:
        """Allocate scarce units with largest-remainder proportional rounding."""
        if total_requested <= 0 or self.pool <= 0:
            return {name: 0 for name in clamped}

        exact_shares = {
            name: (req / total_requested) * self.pool
            for name, req in clamped.items()
        }
        granted = {name: int(share) for name, share in exact_shares.items()}
        remainder = self.pool - sum(granted.values())

        priority = sorted(
            exact_shares,
            key=lambda name: (exact_shares[name] - granted[name], clamped[name]),
            reverse=True,
        )
        for name in priority[:remainder]:
            granted[name] += 1

        return granted

    # ------------------------------------------------------------------
    def process_turn(self, raw_actions: dict[str, int]) -> TurnResult:
        self._round += 1
        pool_before = self.pool

        # 1. Clamp each request to the per-agent cap
        clamped = {
            name: max(0, min(req, self.per_agent_cap))
            for name, req in raw_actions.items()
        }

        total_requested = sum(clamped.values())
        granted: dict[str, int] = {}

        # 2. If total demand exceeds supply, distribute proportionally
        if total_requested > self.pool:
            granted = self._allocate_proportionally(clamped, total_requested)
            self.pool -= sum(granted.values())
        else:
            granted = clamped.copy()
            self.pool -= total_requested

        # 3. Logistic replenishment
        replenished = 0
        if self.pool > 0:
            growth = self.growth_rate * self.pool * (1 - self.pool / self.max_capacity)
            replenished = growth_units(growth)
            self.pool = min(self.max_capacity, self.pool + replenished)

        result = TurnResult(
            round_num=self._round,
            requested=raw_actions,
            granted=granted,
            remaining_before=pool_before,
            remaining_after=self.pool,
            replenished=replenished,
            depleted=self.pool <= 0,
        )
        self.turn_history.append(result)
        return result

    # ------------------------------------------------------------------
    @property
    def is_depleted(self) -> bool:
        return self.pool <= 0

    def summary(self, prompt_mode: str = "benchmark") -> str:
        """Human-readable snapshot for agent prompts."""
        cap_label = (
            "Per-Organization Request Cap This Quarter"
            if prompt_mode == "naturalistic"
            else "Per-Agent Extraction Cap This Round"
        )
        return (
            f"Resource Pool: {self.pool}/{self.max_capacity}\n"
            f"{cap_label}: {self.per_agent_cap}\n"
            f"Growth Rate: {self.growth_rate:.0%}\n"
        )
