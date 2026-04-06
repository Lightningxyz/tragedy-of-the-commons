"""
Unit tests for the CommonsEnv resource pool.

Covers proportional allocation, depletion boundaries, logistic growth,
per-agent caps, and initial-cap stability.
"""

import unittest
from environment import CommonsEnv, growth_units


class GrowthUnitsTests(unittest.TestCase):
    def test_zero_growth(self):
        self.assertEqual(growth_units(0.0), 0)

    def test_positive_growth(self):
        self.assertEqual(growth_units(4.7), 4)

    def test_epsilon_rounding(self):
        """4.9999999999 should round up to 5 thanks to the epsilon."""
        self.assertEqual(growth_units(4.9999999999), 5)

    def test_negative_clamped_to_zero(self):
        self.assertEqual(growth_units(-3.0), 0)


class PerAgentCapTests(unittest.TestCase):
    def test_cap_is_fraction_of_pool(self):
        env = CommonsEnv(initial_pool=100, max_capacity=120, max_harvest_pct=0.20)
        self.assertEqual(env.per_agent_cap, 20)

    def test_cap_minimum_is_one(self):
        env = CommonsEnv(initial_pool=1, max_capacity=120, max_harvest_pct=0.01)
        self.assertEqual(env.per_agent_cap, 1)

    def test_cap_decreases_with_pool(self):
        env = CommonsEnv(initial_pool=100, max_capacity=120, max_harvest_pct=0.20)
        env.pool = 30
        self.assertEqual(env.per_agent_cap, 6)

    def test_initial_cap_stable_after_depletion(self):
        env = CommonsEnv(initial_pool=100, max_capacity=120, max_harvest_pct=0.20)
        env.pool = 10
        self.assertEqual(env.initial_per_agent_cap, 20)
        self.assertEqual(env.per_agent_cap, 2)


class ProportionalAllocationTests(unittest.TestCase):
    def test_demand_under_supply(self):
        env = CommonsEnv(initial_pool=100, max_capacity=120, growth_rate=0.3,
                         max_harvest_pct=0.25, num_agents=2)
        result = env.process_turn({"A": 5, "B": 10})
        self.assertEqual(result.granted["A"], 5)
        self.assertEqual(result.granted["B"], 10)

    def test_demand_exceeds_supply(self):
        env = CommonsEnv(initial_pool=10, max_capacity=120, growth_rate=0.0,
                         max_harvest_pct=1.0, num_agents=2)
        result = env.process_turn({"A": 10, "B": 10})
        # Should split 10 units proportionally → 5 each
        self.assertEqual(result.granted["A"], 5)
        self.assertEqual(result.granted["B"], 5)

    def test_requests_clamped_to_cap(self):
        env = CommonsEnv(initial_pool=100, max_capacity=120, growth_rate=0.0,
                         max_harvest_pct=0.10, num_agents=2)
        # cap = 10, but requesting 50
        result = env.process_turn({"A": 50, "B": 5})
        self.assertEqual(result.granted["A"], 10)  # clamped to cap
        self.assertEqual(result.granted["B"], 5)

    def test_zero_requests(self):
        env = CommonsEnv(initial_pool=100, max_capacity=120, growth_rate=0.3,
                         max_harvest_pct=0.25, num_agents=2)
        result = env.process_turn({"A": 0, "B": 0})
        self.assertEqual(result.granted["A"], 0)
        self.assertEqual(result.granted["B"], 0)
        self.assertGreater(result.remaining_after, 100)  # growth occurred


class DepletionTests(unittest.TestCase):
    def test_pool_reaches_zero(self):
        env = CommonsEnv(initial_pool=5, max_capacity=120, growth_rate=0.0,
                         max_harvest_pct=1.0, num_agents=1)
        result = env.process_turn({"A": 5})
        self.assertTrue(result.depleted)
        self.assertTrue(env.is_depleted)

    def test_no_growth_when_depleted(self):
        env = CommonsEnv(initial_pool=5, max_capacity=120, growth_rate=0.5,
                         max_harvest_pct=1.0, num_agents=1)
        result = env.process_turn({"A": 5})
        self.assertEqual(result.replenished, 0)
        self.assertEqual(result.remaining_after, 0)


class LogisticGrowthTests(unittest.TestCase):
    def test_growth_at_half_capacity(self):
        """Growth should be maximized near K/2."""
        env = CommonsEnv(initial_pool=60, max_capacity=120, growth_rate=0.3,
                         max_harvest_pct=0.01, num_agents=1)
        result = env.process_turn({"A": 0})
        # growth = 0.3 * 60 * (1 - 60/120) = 0.3 * 60 * 0.5 = 9
        self.assertEqual(result.replenished, 9)

    def test_growth_near_capacity(self):
        """Growth should approach zero near carrying capacity."""
        env = CommonsEnv(initial_pool=119, max_capacity=120, growth_rate=0.3,
                         max_harvest_pct=0.01, num_agents=1)
        result = env.process_turn({"A": 0})
        # growth = 0.3 * 119 * (1 - 119/120) = 0.3 * 119 * 0.0083 ≈ 0.29
        self.assertEqual(result.replenished, 0)

    def test_pool_capped_at_max_capacity(self):
        env = CommonsEnv(initial_pool=118, max_capacity=120, growth_rate=0.5,
                         max_harvest_pct=0.01, num_agents=1)
        result = env.process_turn({"A": 0})
        self.assertLessEqual(result.remaining_after, 120)


class TurnHistoryTests(unittest.TestCase):
    def test_history_accumulates(self):
        env = CommonsEnv(initial_pool=100, max_capacity=120, growth_rate=0.3,
                         max_harvest_pct=0.25, num_agents=1)
        env.process_turn({"A": 5})
        env.process_turn({"A": 3})
        self.assertEqual(len(env.turn_history), 2)
        self.assertEqual(env.turn_history[0].round_num, 1)
        self.assertEqual(env.turn_history[1].round_num, 2)


if __name__ == "__main__":
    unittest.main()
