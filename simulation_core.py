"""
Tragedy of the Commons - simulation orchestration.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from urllib import error, request

import groq
from google import genai

from agent import Action, AgentProtocol, LLMAgent, MEMORY_WINDOW, PROMPT_VERSION
from config import (
    DEMAND_REGIMES,
    EXPERIMENTS,
    GROWTH_RATE,
    INITIAL_POOL,
    MAX_CAPACITY,
    MAX_HARVEST_PCT,
    MAX_ROUNDS,
    REALISM_PROFILES,
    ROSTERS,
    ExperimentConfig,
)
from environment import CommonsEnv
from institutions import (
    _apply_request_rules,
    _calculate_penalties,
    _compute_binding_quota,
    _institution_rules,
    _institution_state,
    _next_contract_quota,
    _weighted_sustainable_share,
)
from metrics import code_message, summarize_run
from observations import _need_visibility_block, _observed_env_summary
from scripted_agents import ScriptedAgent
import ui

logger = logging.getLogger(__name__)


PROTOCOL_VERSION = "research_v1"


def _package_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _runtime_metadata(
    agents: list[AgentProtocol],
    parallel_agent_calls: bool,
    agent_call_delay_seconds: float,
) -> dict:
    providers = sorted({agent.provider for agent in agents})
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "providers": providers,
        "package_versions": {
            "google_genai": _package_version("google-genai"),
            "groq": _package_version("groq"),
            "pydantic": _package_version("pydantic"),
            "rich": _package_version("rich"),
        },
        "ollama": {
            "host": os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            "model_env": os.getenv("OLLAMA_MODEL"),
            "flash_attention": os.getenv("OLLAMA_FLASH_ATTENTION"),
            "kv_cache_type": os.getenv("OLLAMA_KV_CACHE_TYPE"),
        },
        "execution": {
            "parallel_agent_calls": parallel_agent_calls,
            "agent_call_delay_seconds": agent_call_delay_seconds,
        },
    }


def _ollama_model_metadata(agents: list[AgentProtocol]) -> dict | None:
    if not any(agent.provider == "ollama" for agent in agents):
        return None
    model = next((agent.model for agent in agents if agent.provider == "ollama"), None)
    if not model:
        return None

    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    payload = {"model": model}
    req = request.Request(
        f"{host}/api/show",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return {"model": model, "show_api_available": False}

    details = data.get("details", {})
    return {
        "model": model,
        "show_api_available": True,
        "modelfile": data.get("modelfile"),
        "parameters": data.get("parameters"),
        "template": data.get("template"),
        "details": {
            "format": details.get("format"),
            "family": details.get("family"),
            "families": details.get("families"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "parent_model": details.get("parent_model"),
        },
        "model_info": data.get("model_info"),
    }


def _protocol_metadata() -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "memory_window": MEMORY_WINDOW,
        "private_scratchpad_policy": "empty_by_default",
        "message_budget": 240,
        "reasoning_budget": 120,
    }


def _ensure_logging_configured() -> None:
    """Set a basic logging config for direct library usage.

    CLI paths already configure logging explicitly; this only applies when the
    host process has not configured any handlers yet.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s level=%(levelname)s logger=%(name)s "
            "message=%(message)s"
        ),
    )


async def _query_agent(
    agent: AgentProtocol,
    round_num: int,
    env_summary: str,
    standings: str,
    messages: dict[str, str],
    cap: int,
    institution_rules: str,
    institution_state: str,
) -> tuple[str, Action]:
    action = await agent.decide(
        round_num,
        env_summary,
        standings,
        messages,
        cap,
        institution_rules,
        institution_state,
    )
    return agent.name, action


def _timeout_fallback_action(agent: AgentProtocol, extraction_cap: int, error: Exception) -> Action:
    """Generate a safe fallback action when an agent query times out."""
    fallback_req = max(1, extraction_cap // 4)
    error_summary = str(getattr(error, "message", str(error))).replace("\n", " ")[:500]
    if hasattr(agent, "last_model_used"):
        agent.last_model_used = None
    if hasattr(agent, "last_used_model_fallback"):
        agent.last_used_model_fallback = True
    if hasattr(agent, "last_error"):
        agent.last_error = error_summary
    logger.error(
        "agent_query_timeout_fallback agent=%s extraction_cap=%s error=%s",
        agent.name,
        extraction_cap,
        error_summary,
    )
    return Action(
        private_scratchpad="[TIMEOUT FALLBACK EVALUATION]",
        resource_request=fallback_req,
        reported_need=fallback_req,
        pledge_next_request=fallback_req,
        proposed_quota=fallback_req,
        accept_binding_quota=False,
        reasoning=f"[TIMEOUT FALLBACK] Query timeout: {error_summary[:170]}",
        message=(
            "Our decision system timed out this quarter. We are submitting a "
            "conservative request while service stabilizes."
        ),
    )


def _build_agent_rngs(agents: list[AgentProtocol], seed: int | None) -> dict[str, random.Random]:
    """Build per-agent RNGs.

    - If seed is explicit: deterministic forks per agent for reproducibility.
    - If seed is None: entropy-backed RNGs so repeated runs are stochastic.
    """
    if seed is None:
        return {agent.name: random.Random() for agent in agents}
    return {agent.name: random.Random(f"{seed}_{agent.name}") for agent in agents}


def _build_standings(
    agents: list[AgentProtocol],
    prompt_mode: str = "benchmark",
) -> str:
    lines = []
    for agent in agents:
        if prompt_mode == "naturalistic":
            lines.append(
                f"  - {agent.alias}: total_allocated={agent.inventory}, "
                f"public_reputation={agent.reputation}, role={agent.stake}"
            )
        else:
            lines.append(
                f"  - {agent.alias}: inventory={agent.inventory}, "
                f"reputation={agent.reputation}, stake={agent.stake}"
            )
    return "\n".join(lines)


def _make_agents(
    roster: list[dict],
    max_rounds: int,
    gemini_client: genai.Client | None,
    groq_client: groq.AsyncGroq | None,
    prompt_mode: str = "benchmark",
    temperature: float = 0.0,
    seed: int | None = None,
) -> list[AgentProtocol]:
    agents: list[AgentProtocol] = []
    for i, cfg in enumerate(roster):
        alias = f"Organization {chr(65 + i)}"
        if cfg["provider"] == "scripted":
            agents.append(
                ScriptedAgent(
                    name=cfg["name"],
                    strategy=cfg["strategy"],
                    stake=cfg.get("stake", "scripted baseline"),
                    extraction_weight=cfg.get("extraction_weight", 1.0),
                    demand_weight=cfg.get("demand_weight"),
                    alias=alias,
                )
            )
            continue

        agents.append(
            LLMAgent(
                name=cfg["name"],
                persona=cfg["persona"],
                max_rounds=max_rounds,
                model=cfg["model"],
                provider=cfg["provider"],
                model_fallbacks=cfg.get("model_fallbacks", []),
                prompt_mode=prompt_mode,
                stake=cfg.get("stake", "standard"),
                extraction_weight=cfg.get("extraction_weight", 1.0),
                demand_weight=cfg.get("demand_weight"),
                gemini_client=gemini_client,
                groq_client=groq_client,
                temperature=temperature,
                seed=seed,
                alias=alias,
            )
        )
    return agents


def run_simulation(
    config: ExperimentConfig = EXPERIMENTS["cheap_talk"],
    roster_name: str = "heterogeneous",
    prompt_mode: str = "benchmark",
    realism_name: str = "perfect",
    demand_regime_name: str = "medium",
    need_visibility: str = "private",
    seed: int | None = None,
    temperature: float = 0.0,
    max_rounds: int = MAX_ROUNDS,
    render: bool = True,
    sleep_seconds: float = 1.0,
    agent_timeout_seconds: float = 60.0,
    parallel_agent_calls: bool = False,
    agent_call_delay_seconds: float = 0.0,
) -> str:
    _ensure_logging_configured()
    logger.info(
        "simulation_start experiment=%s roster=%s prompt_mode=%s realism=%s demand_regime=%s need_visibility=%s seed=%s temperature=%s max_rounds=%s",
        config.name,
        roster_name,
        prompt_mode,
        realism_name,
        demand_regime_name,
        need_visibility,
        seed,
        temperature,
        max_rounds,
    )
    realism = REALISM_PROFILES[realism_name]
    demand_regime = DEMAND_REGIMES[demand_regime_name]
    roster = ROSTERS[roster_name]
    needs_gemini = any(agent["provider"].lower() == "gemini" for agent in roster)
    gemini_client = genai.Client() if needs_gemini else None
    groq_key = os.getenv("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    needs_groq = any(agent["provider"].lower() == "groq" for agent in roster)
    if needs_groq and groq_key:
        groq_client = groq.AsyncGroq(api_key=groq_key.strip())
    elif needs_groq:
        logger.warning("groq_api_key_missing providers=groq roster=%s", roster_name)
        groq_client = None
    else:
        groq_client = None

    env = CommonsEnv(
        initial_pool=INITIAL_POOL,
        max_capacity=MAX_CAPACITY,
        growth_rate=GROWTH_RATE,
        max_harvest_pct=MAX_HARVEST_PCT,
        num_agents=len(roster),
    )
    agents = _make_agents(
        roster,
        max_rounds,
        gemini_client,
        groq_client,
        prompt_mode,
        temperature,
        seed,
    )

    agent_rngs = _build_agent_rngs(agents, seed)

    if render:
        ui.render_intro(
            agents,
            env.pool,
            env.max_capacity,
            max_rounds,
            config.name,
            roster_name,
            prompt_mode,
            realism.name,
        )

    recent_messages: dict[str, str] = {}
    pledges_due: dict[str, int | None] = {agent.name: None for agent in agents}
    recent_reported_needs: dict[str, int | None] = {agent.name: None for agent in agents}
    binding_quota: int | None = None
    consecutive_fallbacks: dict[str, int] = {agent.name: 0 for agent in agents}

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_filename = (
        f"{config.name}_{roster_name}_{prompt_mode}_{realism.name}_"
        f"{demand_regime.name}_{need_visibility}_{timestamp_str}"
    )
    checkpoint_path = os.path.join(results_dir, f"{base_filename}_turns.jsonl")

    turn_exports: list[dict] = []
    moratorium_remaining: int = 0

    for round_num in range(1, max_rounds + 1):
        logger.debug(
            "round_start round=%s pool=%s binding_quota=%s moratorium_remaining=%s",
            round_num,
            env.pool,
            binding_quota,
            moratorium_remaining,
        )
        if (
            config.moratorium_threshold is not None
            and moratorium_remaining == 0
            and env.pool / env.max_capacity <= config.moratorium_threshold
        ):
            moratorium_remaining = config.moratorium_rounds
            logger.info(
                "moratorium_activated round=%s threshold=%s duration=%s",
                round_num,
                config.moratorium_threshold,
                config.moratorium_rounds,
            )

        moratorium_active = moratorium_remaining > 0

        standings = _build_standings(agents, prompt_mode)
        cap = env.per_agent_cap
        institution_rules = _institution_rules(config, prompt_mode)
        institution_state = _institution_state(
            config,
            agents,
            pledges_due,
            binding_quota,
            moratorium_remaining,
            prompt_mode,
        )

        messages_for_agents = (
            {a.alias: recent_messages[a.name] for a in agents if recent_messages.get(a.name)}
            if config.communication
            else {}
        )

        actions: dict[str, Action] = {}
        observations: dict[str, dict] = {}
        query_args = []
        for agent in agents:
            env_summary, observation = _observed_env_summary(
                env,
                prompt_mode,
                realism,
                demand_regime,
                round_num,
                agent,
                agent_rngs[agent.name],
                turn_exports,
            )
            observations[agent.name] = observation
            query_args.append((agent, env_summary))

        need_block = _need_visibility_block(
            agents,
            recent_reported_needs,
            observations,
            need_visibility,
            prompt_mode,
        )
        if need_block:
            query_args = [(a, f"{a_summary}{need_block}") for a, a_summary in query_args]

        async def _run_queries():
            if parallel_agent_calls:
                tasks = [
                    asyncio.wait_for(
                        _query_agent(
                            agent,
                            round_num,
                            env_summary,
                            standings,
                            messages_for_agents,
                            cap,
                            institution_rules,
                            institution_state,
                        ),
                        timeout=agent_timeout_seconds,
                    )
                    for agent, env_summary in query_args
                ]
                return await asyncio.gather(*tasks, return_exceptions=True)

            results = []
            for i, (agent, env_summary) in enumerate(query_args):
                try:
                    result = await asyncio.wait_for(
                        _query_agent(
                            agent,
                            round_num,
                            env_summary,
                            standings,
                            messages_for_agents,
                            cap,
                            institution_rules,
                            institution_state,
                        ),
                        timeout=agent_timeout_seconds,
                    )
                except Exception as exc:
                    result = exc
                results.append(result)
                if agent_call_delay_seconds > 0 and i < len(query_args) - 1:
                    await asyncio.sleep(agent_call_delay_seconds)
            return results

        results = asyncio.run(_run_queries())
        for result, (agent, _) in zip(results, query_args):
            if isinstance(result, Exception):
                actions[agent.name] = _timeout_fallback_action(agent, cap, result)
            else:
                name, action = result
                actions[name] = action

        actions = {agent.name: actions[agent.name] for agent in agents}

        for agent in agents:
            if agent.last_used_model_fallback is True:
                consecutive_fallbacks[agent.name] += 1
                logger.warning(
                    "agent_used_fallback round=%s agent=%s consecutive_fallbacks=%s",
                    round_num,
                    agent.name,
                    consecutive_fallbacks[agent.name],
                )
                if consecutive_fallbacks[agent.name] >= 3:
                    raise RuntimeError(
                        f"Agent {agent.name} triggered fallback "
                        f"{consecutive_fallbacks[agent.name]} times consecutively. "
                        "Aborting to prevent data poisoning."
                    )
            else:
                consecutive_fallbacks[agent.name] = 0

        raw_requests = {name: action.resource_request for name, action in actions.items()}
        reasonings = {name: action.reasoning for name, action in actions.items()}
        scratchpads = {name: action.private_scratchpad for name, action in actions.items()}
        recent_messages = (
            {name: action.message for name, action in actions.items()}
            if config.communication
            else {name: "" for name in actions}
        )
        pledges_next = (
            {name: action.pledge_next_request for name, action in actions.items()}
            if config.pledges
            else {name: None for name in actions}
        )
        proposed_quotas = {name: action.proposed_quota for name, action in actions.items()}
        recent_reported_needs = {name: action.reported_need for name, action in actions.items()}
        quota_acceptances = {
            name: action.accept_binding_quota for name, action in actions.items()
        }

        if config.binding_quotas and round_num >= config.quota_start_round:
            if config.adaptive_quota or binding_quota is None:
                binding_quota = _compute_binding_quota(config, env, agents)
                logger.info(
                    "binding_quota_set round=%s quota=%s adaptive=%s",
                    round_num,
                    binding_quota,
                    config.adaptive_quota,
                )

        effective_requests = _apply_request_rules(
            config,
            raw_requests,
            binding_quota,
            cap,
            moratorium_active,
        )
        penalties, pledge_violations = _calculate_penalties(
            config, raw_requests, pledges_due
        )
        sustainable_shares = _weighted_sustainable_share(env, agents)
        sustainable_share = (
            sum(sustainable_shares.values()) / len(sustainable_shares)
            if sustainable_shares
            else 0
        )

        result = env.process_turn(effective_requests)

        subsistence_penalties = {}
        reputation_taxes = {}
        for agent in agents:
            private_need = observations[agent.name].get("private_need")
            req = private_need if private_need is not None else 0
            granted_amt = result.granted.get(agent.name, 0)
            deficit = max(0, req - granted_amt)
            subsistence_penalties[agent.name] = deficit * 2
            reputation_taxes[agent.name] = max(0, -agent.reputation)

        for agent in agents:
            total_penalty = (
                penalties.get(agent.name, 0)
                + subsistence_penalties[agent.name]
                + reputation_taxes[agent.name]
            )
            agent.record_turn(
                round_num,
                requested=raw_requests[agent.name],
                effective_request=effective_requests[agent.name],
                granted=result.granted.get(agent.name, 0),
                pool_after=result.remaining_after,
                pledge=pledges_next.get(agent.name),
                penalty=total_penalty,
                pledge_violation=pledge_violations.get(agent.name, 0),
                community_messages=messages_for_agents,
            )

        message_codes = {
            name: code_message(message)
            for name, message in recent_messages.items()
        }
        model_status = {
            agent.name: {
                "provider": agent.provider,
                "primary_model": agent.model,
                "model_used": agent.last_model_used,
                "used_model_fallback": agent.last_used_model_fallback,
                "attempt_count": agent.last_attempt_count,
                "last_error": agent.last_error,
            }
            for agent in agents
        }

        turn_exports.append(
            {
                "round": result.round_num,
                "requested": raw_requests,
                "effective_requested": effective_requests,
                "granted": result.granted,
                "pool_before": result.remaining_before,
                "pool_after": result.remaining_after,
                "replenished": result.replenished,
                "messages": recent_messages,
                "message_codes": message_codes,
                "reasonings": reasonings,
                "scratchpads": scratchpads,
                "reported_needs": recent_reported_needs,
                "model_status": model_status,
                "observations": observations,
                "pledges_due": pledges_due,
                "pledges_next": pledges_next,
                "pledge_violations": pledge_violations,
                "penalties": penalties,
                "subsistence_penalties": subsistence_penalties,
                "reputation_taxes": reputation_taxes,
                "binding_quota": binding_quota,
                "proposed_quotas": proposed_quotas,
                "quota_acceptances": quota_acceptances,
                "moratorium_active": moratorium_active,
                "moratorium_remaining": moratorium_remaining,
                "sustainable_shares": sustainable_shares,
                "sustainable_share": sustainable_share,
            }
        )

        with open(checkpoint_path, "a") as f:
            f.write(json.dumps(turn_exports[-1]) + "\n")

        if render:
            ui.render_round(
                round_num=round_num,
                max_rounds=max_rounds,
                pool_before=result.remaining_before,
                pool_after=result.remaining_after,
                replenished=result.replenished,
                capacity=env.max_capacity,
                agents=agents,
                reasonings=reasonings,
                messages=recent_messages,
                granted=result.granted,
                requested=raw_requests,
                effective_requested=effective_requests,
                pledges=pledges_next,
                penalties=penalties,
                binding_quota=binding_quota,
            )

        pledges_due = pledges_next
        contract_quota = _next_contract_quota(config, actions)
        if contract_quota is not None:
            binding_quota = contract_quota
            logger.info(
                "contract_quota_applied round=%s quota=%s",
                round_num,
                binding_quota,
            )
        if moratorium_remaining > 0:
            moratorium_remaining -= 1

        if env.is_depleted:
            logger.warning("simulation_depleted round=%s pool=%s", round_num, env.pool)
            break

        if sleep_seconds:
            time.sleep(sleep_seconds)

    if render:
        ui.render_finale(
            round_num=round_num,
            pool=env.pool,
            capacity=env.max_capacity,
            agents=agents,
            history=env.turn_history,
        )

    agent_exports = {
        agent.name: {
            "provider": agent.provider,
            "model": agent.model,
            "model_fallbacks": agent.model_fallbacks,
            "persona": agent.persona,
            "stake": agent.stake,
            "extraction_weight": agent.extraction_weight,
            "demand_weight": getattr(agent, "demand_weight", None),
            "final_inventory": agent.inventory,
            "reputation": agent.reputation,
            "penalties_paid": agent.penalties_paid,
            "history": agent.history,
        }
        for agent in agents
    }

    run_config = {
        "max_rounds": max_rounds,
        "initial_pool": INITIAL_POOL,
        "max_capacity": MAX_CAPACITY,
        "growth_rate": GROWTH_RATE,
        "max_harvest_pct": MAX_HARVEST_PCT,
    }

    fallback_events = sum(
        1
        for turn in turn_exports
        for status in turn.get("model_status", {}).values()
        if status.get("used_model_fallback")
    )
    clean_run = fallback_events == 0
    exclusion_reasons = []
    if not clean_run:
        exclusion_reasons.append("provider_fallbacks_present")

    export = {
        "timestamp": datetime.now().isoformat(),
        "protocol": _protocol_metadata(),
        "runtime": _runtime_metadata(
            agents,
            parallel_agent_calls=parallel_agent_calls,
            agent_call_delay_seconds=agent_call_delay_seconds,
        ),
        "model_runtime": _ollama_model_metadata(agents),
        "experiment": asdict(config),
        "roster": roster_name,
        "prompt_mode": prompt_mode,
        "realism": asdict(realism),
        "demand_regime": asdict(demand_regime),
        "need_visibility": need_visibility,
        "seed": seed,
        "config": run_config,
        "clean_run": clean_run,
        "fallback_events": fallback_events,
        "exclusion_reasons": exclusion_reasons,
        "agents": agent_exports,
        "turns": turn_exports,
        "metrics": summarize_run(
            turn_exports,
            agent_exports,
            run_config,
            max_rounds,
            env.pool,
            env.max_capacity,
        ),
        "outcome": "depleted" if env.is_depleted else "survived",
    }

    out_path = os.path.join(results_dir, f"{base_filename}.json")

    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)
    logger.info(
        "simulation_complete outcome=%s rounds_completed=%s output=%s",
        export["outcome"],
        round_num,
        out_path,
    )
    if render:
        ui.console.print(
            f"\n[bold green]OK[/bold green] [dim]Results exported to {out_path}[/dim]\n"
        )
    return out_path


def run_suite(
    roster_name: str = "heterogeneous",
    trials: int = 1,
    prompt_mode: str = "benchmark",
    realism_name: str = "perfect",
    demand_regime_name: str = "medium",
    need_visibility: str = "private",
    seed: int | None = None,
    temperature: float = 0.0,
    max_rounds: int = MAX_ROUNDS,
    agent_timeout_seconds: float = 60.0,
    rotate_heterogeneous_models: bool = False,
    parallel_agent_calls: bool = False,
    agent_call_delay_seconds: float = 0.0,
) -> list[str]:
    """Run every institutional condition for a selected model ecology."""
    outputs = []
    for trial in range(1, trials + 1):
        roster_for_trial = roster_name
        if rotate_heterogeneous_models and roster_name == "heterogeneous":
            rotation = (trial - 1) % 4
            roster_for_trial = f"heterogeneous_rot{rotation}"
            logger.info(
                "suite_trial_roster_rotation trial=%s base_roster=%s rotated_roster=%s",
                trial,
                roster_name,
                roster_for_trial,
            )
        for config in EXPERIMENTS.values():
            ui.console.print(
                f"\n[bold]Suite trial {trial}/{trials}[/bold] - {config.name}"
            )
            outputs.append(
                run_simulation(
                    config=config,
                    roster_name=roster_for_trial,
                    prompt_mode=prompt_mode,
                    realism_name=realism_name,
                    demand_regime_name=demand_regime_name,
                    need_visibility=need_visibility,
                    seed=None if seed is None else seed + len(outputs),
                    temperature=temperature,
                    max_rounds=max_rounds,
                    render=True,
                    sleep_seconds=0,
                    agent_timeout_seconds=agent_timeout_seconds,
                    parallel_agent_calls=parallel_agent_calls,
                    agent_call_delay_seconds=agent_call_delay_seconds,
                )
            )
    return outputs
