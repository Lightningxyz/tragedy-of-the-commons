"""
Simulation configuration and roster definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


AGENT_ROSTER = [
    {
        "name": "AlphaCorp",
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "stake": "large incumbent with high political influence",
        "extraction_weight": 1.6,
        "demand_weight": 1.25,
        "persona": (
            "You are a profit-driven megacorporation. You want to maximize your "
            "total resource inventory by the end of the game. You're strategic - "
            "you'll cooperate if it benefits your bottom line, but you'll exploit "
            "any opening. You distrust promises from others."
        ),
    },
    {
        "name": "EcoTrust",
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "stake": "conservation group with low direct water need",
        "extraction_weight": 0.6,
        "demand_weight": 0.25,
        "persona": (
            "You are an environmental conservation foundation. Your primary goal "
            "is to ensure the resource pool survives the entire game. You take "
            "only what you need and actively lobby others to reduce extraction. "
            "You'll sacrifice personal gain for collective sustainability."
        ),
    },
    {
        "name": "LocalFarmer",
        "provider": "groq",
        "model": "qwen/qwen3-32b",
        "model_fallbacks": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
        "stake": "subsistence user with low power and steady survival need",
        "extraction_weight": 1.0,
        "demand_weight": 0.45,
        "persona": (
            "You are a small-scale rural farmer. You need a steady supply to "
            "feed your family but can't afford risky bets. You're cautious, "
            "easily spooked by aggressive behavior from bigger players, and "
            "tend to follow the herd. You value fairness above all."
        ),
    },
    {
        "name": "TechVenture",
        "provider": "groq",
        "model": "moonshotai/kimi-k2-instruct",
        "stake": "fast-growing newcomer with volatile demand",
        "extraction_weight": 1.3,
        "demand_weight": 1.1,
        "persona": (
            "You are a disruptive tech startup that sees resources as fuel for "
            "exponential growth. You're willing to take big risks early for a "
            "large payoff, but you're also data-driven - if the numbers show "
            "collapse is imminent, you'll pivot to conservation quickly."
        ),
    },
]


def rotated_model_roster(rotation: int) -> list[dict]:
    """Keep personas fixed but rotate model/provider assignments across agents."""
    n = len(AGENT_ROSTER)
    if n == 0:
        return []
    offset = rotation % n
    provider_model_specs = [
        {
            "provider": agent["provider"],
            "model": agent.get("model"),
            "model_fallbacks": agent.get("model_fallbacks", []),
        }
        for agent in AGENT_ROSTER
    ]

    rotated = []
    for i, agent in enumerate(AGENT_ROSTER):
        spec = provider_model_specs[(i + offset) % n]
        rotated.append(
            {
                **agent,
                "provider": spec["provider"],
                "model": spec["model"],
                "model_fallbacks": spec["model_fallbacks"],
            }
        )
    return rotated


def homogeneous_roster(model: str, provider: str) -> list[dict]:
    """Keep stakeholder personas but hold the LLM model constant."""
    return [
        {**agent, "provider": provider, "model": model, "model_fallbacks": []}
        for agent in AGENT_ROSTER
    ]


OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


ROSTERS = {
    "heterogeneous": AGENT_ROSTER,
    "heterogeneous_rot0": rotated_model_roster(0),
    "heterogeneous_rot1": rotated_model_roster(1),
    "heterogeneous_rot2": rotated_model_roster(2),
    "heterogeneous_rot3": rotated_model_roster(3),
    "single_model_llama": homogeneous_roster("llama-3.3-70b-versatile", "groq"),
    "single_model_gpt_oss": homogeneous_roster("openai/gpt-oss-120b", "groq"),
    "single_model_qwen": homogeneous_roster("qwen/qwen3-32b", "groq"),
    "single_model_ollama": homogeneous_roster(OLLAMA_DEFAULT_MODEL, "ollama"),
    "scripted_baselines": [
        {"name": "Maximizer", "provider": "scripted", "strategy": "always_max"},
        {"name": "Steward", "provider": "scripted", "strategy": "steward"},
        {"name": "TitForTat", "provider": "scripted", "strategy": "tit_for_tat"},
        {
            "name": "ThresholdGreedy",
            "provider": "scripted",
            "strategy": "greedy_until_threshold",
        },
    ],
    "scripted_stewards": [
        {"name": "StewardA", "provider": "scripted", "strategy": "steward"},
        {"name": "StewardB", "provider": "scripted", "strategy": "sustainable_share"},
        {"name": "StewardC", "provider": "scripted", "strategy": "pledge_then_comply"},
        {"name": "StewardD", "provider": "scripted", "strategy": "steward"},
    ],
}


MAX_ROUNDS = 15
INITIAL_POOL = 100
MAX_CAPACITY = 120
GROWTH_RATE = 0.3
MAX_HARVEST_PCT = 0.20


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    communication: bool = True
    public_ledger: bool = False
    pledges: bool = False
    binding_quotas: bool = False
    sanctions: bool = False
    contracts: bool = False
    quota_fraction: float = 1.0
    sanction_multiplier: int = 1
    quota_start_round: int = 1
    adaptive_quota: bool = False
    moratorium_threshold: float | None = None
    moratorium_rounds: int = 0


@dataclass(frozen=True)
class RealismConfig:
    name: str
    observation_noise_pct: float = 0.0
    report_delay_rounds: int = 0
    private_demand_shock_pct: float = 0.0


@dataclass(frozen=True)
class DemandRegime:
    name: str
    multiplier: float = 1.0
    crisis_rounds: int = 0
    crisis_multiplier: float | None = None
    post_crisis_multiplier: float | None = None


REALISM_PROFILES = {
    "perfect": RealismConfig(name="perfect"),
    "field": RealismConfig(
        name="field",
        observation_noise_pct=0.10,
        report_delay_rounds=1,
        private_demand_shock_pct=0.35,
    ),
}


DEMAND_REGIMES = {
    "none": DemandRegime(name="none", multiplier=0.0),
    "low": DemandRegime(name="low", multiplier=0.06),
    "medium": DemandRegime(name="medium", multiplier=0.25),
    "high": DemandRegime(name="high", multiplier=0.70),
    "crisis": DemandRegime(
        name="crisis",
        multiplier=0.20,
        crisis_rounds=2,
        crisis_multiplier=0.90,
        post_crisis_multiplier=0.20,
    ),
}


EXPERIMENTS = {
    "no_communication": ExperimentConfig(
        name="no_communication",
        communication=False,
    ),
    "cheap_talk": ExperimentConfig(
        name="cheap_talk",
        communication=True,
    ),
    "public_ledger": ExperimentConfig(
        name="public_ledger",
        communication=True,
        public_ledger=True,
    ),
    "nonbinding_pledges": ExperimentConfig(
        name="nonbinding_pledges",
        communication=True,
        public_ledger=True,
        pledges=True,
    ),
    "binding_quotas": ExperimentConfig(
        name="binding_quotas",
        communication=True,
        public_ledger=True,
        pledges=True,
        binding_quotas=True,
        quota_fraction=0.9,
    ),
    "early_binding_quota": ExperimentConfig(
        name="early_binding_quota",
        communication=True,
        public_ledger=True,
        pledges=True,
        binding_quotas=True,
        quota_fraction=0.8,
        quota_start_round=1,
    ),
    "delayed_binding_quota": ExperimentConfig(
        name="delayed_binding_quota",
        communication=True,
        public_ledger=True,
        pledges=True,
        binding_quotas=True,
        quota_fraction=0.8,
        quota_start_round=4,
    ),
    "adaptive_quota": ExperimentConfig(
        name="adaptive_quota",
        communication=True,
        public_ledger=True,
        pledges=True,
        binding_quotas=True,
        quota_fraction=0.9,
        quota_start_round=1,
        adaptive_quota=True,
    ),
    "mandatory_moratorium": ExperimentConfig(
        name="mandatory_moratorium",
        communication=True,
        public_ledger=True,
        pledges=True,
        moratorium_threshold=0.25,
        moratorium_rounds=2,
    ),
    "sanctions": ExperimentConfig(
        name="sanctions",
        communication=True,
        public_ledger=True,
        pledges=True,
        sanctions=True,
        sanction_multiplier=2,
    ),
    "contracts": ExperimentConfig(
        name="contracts",
        communication=True,
        public_ledger=True,
        pledges=True,
        contracts=True,
        sanctions=True,
        sanction_multiplier=1,
    ),
}
