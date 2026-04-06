"""
Unit tests for simulation helper functions.

Covers request rules, penalty calculation, contract voting,
demand multipliers, and binding quota computation.
"""

import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch

from agent import Action, LLMAgent
from environment import CommonsEnv
from simulation import (
    AGENT_ROSTER,
    ROSTERS,
    ExperimentConfig,
    DemandRegime,
    _apply_request_rules,
    _build_agent_rngs,
    _calculate_penalties,
    _next_contract_quota,
    _demand_multiplier,
    _compute_binding_quota,
    _timeout_fallback_action,
    _weighted_sustainable_share,
    rotated_model_roster,
    run_simulation,
)


class ApplyRequestRulesTests(unittest.TestCase):
    def test_moratorium_zeroes_all(self):
        config = ExperimentConfig(name="test", binding_quotas=True)
        result = _apply_request_rules(
            config, {"A": 10, "B": 15}, binding_quota=20, cap=25,
            moratorium_active=True,
        )
        self.assertEqual(result, {"A": 0, "B": 0})

    def test_binding_quota_clamps(self):
        config = ExperimentConfig(name="test", binding_quotas=True)
        result = _apply_request_rules(
            config, {"A": 10, "B": 15}, binding_quota=8, cap=25,
            moratorium_active=False,
        )
        self.assertEqual(result["A"], 8)
        self.assertEqual(result["B"], 8)

    def test_cap_clamps_within_quota(self):
        config = ExperimentConfig(name="test", binding_quotas=True)
        result = _apply_request_rules(
            config, {"A": 10, "B": 15}, binding_quota=20, cap=7,
            moratorium_active=False,
        )
        self.assertEqual(result["A"], 7)
        self.assertEqual(result["B"], 7)

    def test_no_quota_passes_through(self):
        config = ExperimentConfig(name="test", binding_quotas=False)
        result = _apply_request_rules(
            config, {"A": 10, "B": 15}, binding_quota=None, cap=25,
        )
        self.assertEqual(result, {"A": 10, "B": 15})

    def test_contracts_clamp_like_quotas(self):
        config = ExperimentConfig(name="test", contracts=True)
        result = _apply_request_rules(
            config, {"A": 10, "B": 15}, binding_quota=5, cap=25,
        )
        self.assertEqual(result["A"], 5)
        self.assertEqual(result["B"], 5)


class CalculatePenaltiesTests(unittest.TestCase):
    def test_no_pledge_no_penalty(self):
        config = ExperimentConfig(name="test", sanctions=True, sanction_multiplier=2)
        penalties, violations = _calculate_penalties(
            config, {"A": 10}, {"A": None},
        )
        self.assertEqual(penalties["A"], 0)
        self.assertEqual(violations["A"], 0)

    def test_pledge_honored_no_penalty(self):
        config = ExperimentConfig(name="test", sanctions=True, sanction_multiplier=2)
        penalties, violations = _calculate_penalties(
            config, {"A": 5}, {"A": 8},
        )
        self.assertEqual(penalties["A"], 0)
        self.assertEqual(violations["A"], 0)

    def test_pledge_broken_with_sanctions(self):
        config = ExperimentConfig(name="test", sanctions=True, sanction_multiplier=2)
        penalties, violations = _calculate_penalties(
            config, {"A": 12}, {"A": 8},
        )
        self.assertEqual(violations["A"], 4)
        self.assertEqual(penalties["A"], 8)  # 4 * 2

    def test_pledge_broken_without_sanctions(self):
        config = ExperimentConfig(name="test", sanctions=False)
        penalties, violations = _calculate_penalties(
            config, {"A": 12}, {"A": 8},
        )
        self.assertEqual(violations["A"], 4)
        self.assertEqual(penalties["A"], 0)


class ContractQuotaTests(unittest.TestCase):
    def _action(self, accept: bool, quota: int | None) -> Action:
        return Action(
            private_scratchpad="test",
            resource_request=5,
            reasoning="test",
            message="test",
            accept_binding_quota=accept,
            proposed_quota=quota,
        )

    def test_majority_accepts_median(self):
        config = ExperimentConfig(name="test", contracts=True)
        actions = {
            "A": self._action(True, 10),
            "B": self._action(True, 6),
            "C": self._action(False, None),
        }
        # 2 out of 3 accept → majority
        result = _next_contract_quota(config, actions)
        self.assertEqual(result, 8)  # median of [6, 10]

    def test_no_majority_returns_none(self):
        config = ExperimentConfig(name="test", contracts=True)
        actions = {
            "A": self._action(True, 10),
            "B": self._action(False, None),
            "C": self._action(False, None),
            "D": self._action(False, None),
        }
        result = _next_contract_quota(config, actions)
        self.assertIsNone(result)

    def test_contracts_disabled_returns_none(self):
        config = ExperimentConfig(name="test", contracts=False)
        actions = {"A": self._action(True, 10)}
        result = _next_contract_quota(config, actions)
        self.assertIsNone(result)


class DemandMultiplierTests(unittest.TestCase):
    def test_none_regime(self):
        regime = DemandRegime(name="none", multiplier=0.0)
        self.assertEqual(_demand_multiplier(regime, 1), 0.0)

    def test_constant_regime(self):
        regime = DemandRegime(name="medium", multiplier=0.25)
        self.assertEqual(_demand_multiplier(regime, 1), 0.25)
        self.assertEqual(_demand_multiplier(regime, 10), 0.25)

    def test_crisis_during_crisis_rounds(self):
        regime = DemandRegime(
            name="crisis", multiplier=0.20,
            crisis_rounds=2, crisis_multiplier=0.90, post_crisis_multiplier=0.20,
        )
        self.assertEqual(_demand_multiplier(regime, 1), 0.90)
        self.assertEqual(_demand_multiplier(regime, 2), 0.90)

    def test_crisis_after_crisis_rounds(self):
        regime = DemandRegime(
            name="crisis", multiplier=0.20,
            crisis_rounds=2, crisis_multiplier=0.90, post_crisis_multiplier=0.20,
        )
        self.assertEqual(_demand_multiplier(regime, 3), 0.20)


class BindingQuotaTests(unittest.TestCase):
    def test_quota_scales_with_growth(self):
        config = ExperimentConfig(name="test", binding_quotas=True, quota_fraction=1.0)
        env = CommonsEnv(initial_pool=100, max_capacity=120, growth_rate=0.3,
                         max_harvest_pct=0.2, num_agents=2)
        agent_a = SimpleNamespace(extraction_weight=1.0, name="A")
        agent_b = SimpleNamespace(extraction_weight=1.0, name="B")
        quota = _compute_binding_quota(config, env, [agent_a, agent_b])
        self.assertGreater(quota, 0)

    def test_quota_fraction_reduces(self):
        config_full = ExperimentConfig(name="t1", binding_quotas=True, quota_fraction=1.0)
        config_half = ExperimentConfig(name="t2", binding_quotas=True, quota_fraction=0.5)
        env = CommonsEnv(initial_pool=100, max_capacity=120, growth_rate=0.3,
                         max_harvest_pct=0.2, num_agents=2)
        agent = SimpleNamespace(extraction_weight=1.0, name="A")
        q_full = _compute_binding_quota(config_full, env, [agent])
        q_half = _compute_binding_quota(config_half, env, [agent])
        self.assertGreaterEqual(q_full, q_half)


class WeightedSustainableShareTests(unittest.TestCase):
    def test_equal_weights(self):
        env = CommonsEnv(initial_pool=100, max_capacity=120, growth_rate=0.3,
                         max_harvest_pct=0.2, num_agents=2)
        a = SimpleNamespace(extraction_weight=1.0, name="A")
        b = SimpleNamespace(extraction_weight=1.0, name="B")
        shares = _weighted_sustainable_share(env, [a, b])
        self.assertEqual(shares["A"], shares["B"])

    def test_unequal_weights(self):
        env = CommonsEnv(initial_pool=100, max_capacity=120, growth_rate=0.3,
                         max_harvest_pct=0.2, num_agents=2)
        a = SimpleNamespace(extraction_weight=2.0, name="A")
        b = SimpleNamespace(extraction_weight=1.0, name="B")
        shares = _weighted_sustainable_share(env, [a, b])
        self.assertGreater(shares["A"], shares["B"])

    def test_empty_agents(self):
        env = CommonsEnv(initial_pool=100, max_capacity=120, growth_rate=0.3)
        self.assertEqual(_weighted_sustainable_share(env, []), {})


class AgentRngForkTests(unittest.TestCase):
    def test_explicit_seed_is_deterministic_per_agent(self):
        agents = [SimpleNamespace(name="A"), SimpleNamespace(name="B")]
        rngs1 = _build_agent_rngs(agents, seed=123)
        rngs2 = _build_agent_rngs(agents, seed=123)

        self.assertAlmostEqual(rngs1["A"].random(), rngs2["A"].random())
        self.assertAlmostEqual(rngs1["B"].random(), rngs2["B"].random())

    def test_none_seed_is_not_forced_deterministic(self):
        agents = [SimpleNamespace(name="A"), SimpleNamespace(name="B")]
        rngs1 = _build_agent_rngs(agents, seed=None)
        rngs2 = _build_agent_rngs(agents, seed=None)

        # Extremely unlikely to collide if backed by fresh system entropy.
        self.assertNotEqual(rngs1["A"].random(), rngs2["A"].random())
        self.assertNotEqual(rngs1["B"].random(), rngs2["B"].random())


class TimeoutFallbackTests(unittest.TestCase):
    def test_timeout_fallback_action_is_valid_and_conservative(self):
        agent = SimpleNamespace(
            name="A",
            last_model_used="model-x",
            last_used_model_fallback=False,
            last_error=None,
        )
        action = _timeout_fallback_action(
            agent,
            extraction_cap=20,
            error=TimeoutError("timed out waiting for provider"),
        )
        self.assertEqual(action.resource_request, 5)
        self.assertTrue(action.reasoning.startswith("[TIMEOUT FALLBACK]"))
        self.assertTrue(action.message)
        self.assertTrue(agent.last_used_model_fallback)
        self.assertIsNone(agent.last_model_used)
        self.assertIsNotNone(agent.last_error)


class RotatedRosterTests(unittest.TestCase):
    def test_rotated_roster_preserves_personas_but_changes_model_assignment(self):
        rot0 = rotated_model_roster(0)
        rot1 = rotated_model_roster(1)
        self.assertEqual(len(rot0), len(AGENT_ROSTER))

        # Persona of each index remains fixed.
        for i in range(len(AGENT_ROSTER)):
            self.assertEqual(rot0[i]["persona"], AGENT_ROSTER[i]["persona"])
            self.assertEqual(rot1[i]["persona"], AGENT_ROSTER[i]["persona"])

        # At least one model assignment changes under rotation.
        changed = any(
            rot1[i].get("model") != AGENT_ROSTER[i].get("model")
            for i in range(len(AGENT_ROSTER))
        )
        self.assertTrue(changed)


class ExportMetadataTests(unittest.TestCase):
    def test_run_export_includes_research_metadata(self):
        out = run_simulation(
            config=ExperimentConfig(name="test_no_comm", communication=False),
            roster_name="scripted_baselines",
            prompt_mode="benchmark",
            realism_name="perfect",
            demand_regime_name="medium",
            need_visibility="private",
            seed=123,
            temperature=0.0,
            max_rounds=1,
            render=False,
            sleep_seconds=0,
            agent_timeout_seconds=5.0,
        )
        with open(out) as f:
            data = json.load(f)

        self.assertIn("protocol", data)
        self.assertIn("runtime", data)
        self.assertIn("clean_run", data)
        self.assertIn("fallback_events", data)
        self.assertIn("exclusion_reasons", data)
        self.assertEqual(data["protocol"]["prompt_version"], "compact_v1")
        self.assertEqual(data["protocol"]["protocol_version"], "research_v1")
        self.assertEqual(data["clean_run"], True)
        self.assertEqual(data["fallback_events"], 0)

    def test_run_export_marks_fallback_contamination(self):
        with patch(
            "simulation_core._query_agent",
            side_effect=TimeoutError("forced timeout for metadata test"),
        ):
            out = run_simulation(
                config=ExperimentConfig(name="test_no_comm", communication=False),
                roster_name="heterogeneous",
                prompt_mode="benchmark",
                realism_name="perfect",
                demand_regime_name="medium",
                need_visibility="private",
                seed=123,
                temperature=0.0,
                max_rounds=1,
                render=False,
                sleep_seconds=0,
                agent_timeout_seconds=0.01,
            )

        with open(out) as f:
            data = json.load(f)

        self.assertFalse(data["clean_run"])
        self.assertGreater(data["fallback_events"], 0)
        self.assertIn("provider_fallbacks_present", data["exclusion_reasons"])

    def test_rotated_rosters_are_registered(self):
        self.assertIn("heterogeneous_rot0", ROSTERS)
        self.assertIn("heterogeneous_rot1", ROSTERS)
        self.assertIn("heterogeneous_rot2", ROSTERS)
        self.assertIn("heterogeneous_rot3", ROSTERS)


class ReasoningCanonicalizationTests(unittest.TestCase):
    def test_legacy_rationale_is_mapped_to_reasoning(self):
        agent = LLMAgent(
            name="TestAgent",
            persona="test persona",
            provider="groq",
            model="test-model",
            groq_client=None,
            gemini_client=None,
        )
        action = agent._normalize_action_data(
            {
                "private_scratchpad": "x",
                "resource_request": 3,
                "rationale": "legacy rationale key",
                "message": "ok",
            },
            extraction_cap=10,
        )
        self.assertEqual(action.reasoning, "legacy rationale key")


if __name__ == "__main__":
    unittest.main()
