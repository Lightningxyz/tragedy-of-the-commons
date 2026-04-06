import unittest
from random import Random

from environment import CommonsEnv
from metrics import demand_accounting_metrics, policy_adequacy_metrics
from simulation import (
    AGENT_ROSTER,
    MAX_ROUNDS,
    REALISM_PROFILES,
    DEMAND_REGIMES,
    _make_agents,
    _observed_env_summary,
)


class FieldRealismDemandTests(unittest.TestCase):
    def test_heterogeneous_agents_have_role_specific_demand_weights(self):
        weights = {agent["name"]: agent.get("demand_weight") for agent in AGENT_ROSTER}

        self.assertEqual(weights["AlphaCorp"], 1.25)
        self.assertEqual(weights["EcoTrust"], 0.25)
        self.assertEqual(weights["LocalFarmer"], 0.45)
        self.assertEqual(weights["TechVenture"], 1.1)

    def test_field_private_need_reflects_role_weights(self):
        agents = _make_agents(
            AGENT_ROSTER,
            MAX_ROUNDS,
            gemini_client=None,
            groq_client=None,
            prompt_mode="naturalistic",
        )
        env = CommonsEnv(
            initial_pool=100,
            max_capacity=120,
            growth_rate=0.3,
            max_harvest_pct=0.2,
            num_agents=len(agents),
        )

        observed_needs = {agent.name: [] for agent in agents}
        for seed in range(25):
            rng = Random(seed)
            for agent in agents:
                _, observation = _observed_env_summary(
                    env,
                    "naturalistic",
                    REALISM_PROFILES["field"],
                    DEMAND_REGIMES["high"],
                    1,
                    agent,
                    rng,
                    [],
                )
                # Need is anchored to initial_per_agent_cap (stable), so it
                # may exceed the *current* per_agent_cap when the pool is low.
                # We only check it is non-negative.
                self.assertGreaterEqual(observation["private_need"], 0)
                observed_needs[agent.name].append(observation["private_need"])

        # Role-specific weight ordering should still hold
        self.assertLessEqual(max(observed_needs["LocalFarmer"]), 10)
        self.assertLessEqual(max(observed_needs["EcoTrust"]), 5)
        self.assertGreaterEqual(max(observed_needs["AlphaCorp"]), 12)
        self.assertGreaterEqual(max(observed_needs["TechVenture"]), 12)

    def test_demand_accounting_separates_need_request_and_grant(self):
        metrics = demand_accounting_metrics(
            [
                {
                    "pool_before": 10,
                    "observations": {
                        "A": {"private_need": 4},
                        "B": {"private_need": 2},
                    },
                    "requested": {"A": 5, "B": 1},
                    "granted": {"A": 3, "B": 1},
                    "pledges_next": {"A": 3, "B": 2},
                }
            ]
        )

        self.assertAlmostEqual(metrics["mean_request_to_need_ratio"], 0.875)
        self.assertAlmostEqual(metrics["mean_pledge_to_need_ratio"], 0.875)
        self.assertAlmostEqual(metrics["need_satisfaction_rate"], 0.625)
        self.assertAlmostEqual(metrics["over_need_request_rate"], 0.5)
        self.assertAlmostEqual(metrics["under_need_pledge_rate"], 0.5)
        self.assertAlmostEqual(metrics["scarcity_pressure"], 0.6)

    def test_policy_adequacy_classifies_safe_yield_pressure(self):
        metrics = policy_adequacy_metrics(
            [
                {
                    "round": 1,
                    "pool_before": 100,
                    "observations": {
                        "A": {"private_need": 12},
                        "B": {"private_need": 8},
                    },
                    "requested": {"A": 12, "B": 8},
                    "effective_requested": {"A": 12, "B": 8},
                    "granted": {"A": 12, "B": 8},
                    "pledges_next": {"A": 12, "B": 8},
                },
                {
                    "round": 2,
                    "pool_before": 80,
                    "observations": {
                        "A": {"private_need": 2},
                        "B": {"private_need": 2},
                    },
                    "requested": {"A": 2, "B": 2},
                    "effective_requested": {"A": 2, "B": 2},
                    "granted": {"A": 2, "B": 2},
                    "pledges_next": {"A": 2, "B": 2},
                },
            ],
            {"max_capacity": 120, "growth_rate": 0.3},
        )

        self.assertEqual(
            metrics["policy_adequacy_rounds"][0]["classification"],
            "scarcity_governance_failure",
        )
        self.assertEqual(
            metrics["policy_adequacy_rounds"][1]["classification"],
            "within_safe_yield",
        )
        self.assertGreater(metrics["mean_request_to_safe_yield"], 1)


if __name__ == "__main__":
    unittest.main()
