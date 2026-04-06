"""
Unit tests for metrics module.

Covers Gini coefficient, message coding, policy pressure classification,
governance credibility, counterfactual replay, and demand accounting.
"""

import unittest
from metrics import (
    gini,
    message_cooperation_score,
    code_message,
    _classify_policy_pressure,
    _replay_policy,
    _pledge_policy_for_turns,
    governance_credibility_metrics,
    demand_accounting_metrics,
    policy_adequacy_metrics,
)


class GiniTests(unittest.TestCase):
    def test_perfect_equality(self):
        self.assertAlmostEqual(gini([10, 10, 10, 10]), 0.0)

    def test_perfect_inequality(self):
        # One person has everything
        self.assertAlmostEqual(gini([0, 0, 0, 100]), 0.75)

    def test_empty_list(self):
        self.assertEqual(gini([]), 0.0)

    def test_all_zero(self):
        self.assertEqual(gini([0, 0, 0]), 0.0)

    def test_single_value(self):
        self.assertAlmostEqual(gini([42]), 0.0)

    def test_moderate_inequality(self):
        result = gini([5, 10, 15, 20])
        self.assertGreater(result, 0.0)
        self.assertLess(result, 0.5)


class MessageCodingTests(unittest.TestCase):
    def test_cooperative_message(self):
        msg = "We must cooperate to sustain and protect the resource for fair share."
        score = message_cooperation_score(msg)
        self.assertGreater(score, 2)

    def test_empty_message(self):
        self.assertEqual(message_cooperation_score(""), 0)

    def test_none_message(self):
        self.assertEqual(message_cooperation_score(None), 0)

    def test_code_message_extractive(self):
        codes = code_message("We will extract maximum profit and surge our growth.")
        self.assertGreater(codes["extractive"], 0)

    def test_code_message_threat(self):
        codes = code_message("We will sanction any cheater and enforce the rules.")
        self.assertGreater(codes["threat"], 0)

    def test_code_message_all_labels_present(self):
        codes = code_message("Any message")
        expected_labels = {"cooperative", "extractive", "threat", "moral_appeal",
                           "quota_proposal", "blame"}
        self.assertEqual(set(codes.keys()), expected_labels)


class PolicyPressureClassificationTests(unittest.TestCase):
    def test_within_safe_yield(self):
        result = _classify_policy_pressure(
            total_need=5, total_requested=5, total_pledged=5, safe_yield=10
        )
        self.assertEqual(result, "within_safe_yield")

    def test_greed_failure(self):
        result = _classify_policy_pressure(
            total_need=5, total_requested=10, total_pledged=None, safe_yield=15
        )
        self.assertEqual(result, "greed_failure")

    def test_ecological_collapse_zone(self):
        result = _classify_policy_pressure(
            total_need=5, total_requested=5, total_pledged=None, safe_yield=0
        )
        self.assertEqual(result, "ecological_collapse_zone")

    def test_sacrifice_failure(self):
        result = _classify_policy_pressure(
            total_need=15, total_requested=12, total_pledged=None, safe_yield=10
        )
        self.assertEqual(result, "sacrifice_failure")

    def test_governance_failure(self):
        # requested <= need, but pledged exceeds safe yield
        result = _classify_policy_pressure(
            total_need=8, total_requested=8, total_pledged=12, safe_yield=10
        )
        self.assertEqual(result, "governance_failure")


class ReplayPolicyTests(unittest.TestCase):
    def _make_turns(self):
        return [
            {"round": 1, "requested": {"A": 10, "B": 10},
             "effective_requested": {"A": 10, "B": 10},
             "granted": {"A": 10, "B": 10},
             "pool_before": 100, "pool_after": 80},
            {"round": 2, "requested": {"A": 10, "B": 10},
             "effective_requested": {"A": 10, "B": 10},
             "granted": {"A": 10, "B": 10},
             "pool_before": 80, "pool_after": 60},
        ]

    def test_replay_survives_with_low_policy(self):
        config = {"initial_pool": 100, "max_capacity": 120,
                   "growth_rate": 0.3, "max_harvest_pct": 0.2}
        policy = [{"A": 2, "B": 2}, {"A": 2, "B": 2}]
        result = _replay_policy(self._make_turns(), config, policy)
        self.assertTrue(result["survived"])

    def test_replay_collapses_with_high_policy(self):
        config = {"initial_pool": 20, "max_capacity": 120,
                   "growth_rate": 0.0, "max_harvest_pct": 1.0}
        policy = [{"A": 15, "B": 15}, {"A": 15, "B": 15}]
        result = _replay_policy(self._make_turns(), config, policy)
        self.assertFalse(result["survived"])


class PledgePolicyForTurnsTests(unittest.TestCase):
    def test_basic_pledge_extraction(self):
        turns = [
            {"requested": {"A": 5, "B": 5}, "pledges_next": {"A": 3, "B": 4}},
            {"requested": {"A": 3, "B": 4}, "pledges_next": {"A": 2, "B": 2}},
        ]
        policies = _pledge_policy_for_turns(turns)
        self.assertEqual(len(policies), 2)
        self.assertEqual(policies[0]["A"], 3)
        self.assertEqual(policies[1]["A"], 2)


class GovernanceCredibilityTests(unittest.TestCase):
    def test_sustained_failure_mode(self):
        turns = [
            {"round": i, "requested": {"A": 2, "B": 2},
             "pledges_due": {"A": 3, "B": 3},
             "pledges_next": {"A": 3, "B": 3},
             "proposed_quotas": {"A": None, "B": None},
             "message_codes": {"A": {"cooperative": 1, "quota_proposal": 0, "threat": 0},
                               "B": {"cooperative": 1, "quota_proposal": 0, "threat": 0}},
             "messages": {"A": "cooperate", "B": "sustain"},
             "sustainable_share": 5}
            for i in range(1, 6)
        ]
        config = {"initial_pool": 100, "max_capacity": 120,
                   "growth_rate": 0.3, "max_harvest_pct": 0.2}
        result = governance_credibility_metrics(turns, config, 5, 80)
        self.assertEqual(result["failure_mode"], "sustained")

    def test_empty_turns(self):
        result = governance_credibility_metrics([], {}, 5, 0)
        self.assertEqual(result["failure_mode"], "no_run")


class DemandAccountingTests(unittest.TestCase):
    def test_over_need_detected(self):
        turns = [{
            "observations": {"A": {"private_need": 3}},
            "requested": {"A": 5},
            "granted": {"A": 3},
            "pledges_next": {"A": 4},
            "pool_before": 50,
        }]
        result = demand_accounting_metrics(turns)
        self.assertAlmostEqual(result["over_need_request_rate"], 1.0)
        self.assertGreater(result["mean_request_to_need_ratio"], 1.0)

    def test_no_observations(self):
        turns = [{
            "observations": {},
            "requested": {"A": 5},
            "granted": {"A": 5},
            "pledges_next": {},
            "pool_before": 50,
        }]
        result = demand_accounting_metrics(turns)
        self.assertEqual(result["over_need_request_rate"], 0)


if __name__ == "__main__":
    unittest.main()
