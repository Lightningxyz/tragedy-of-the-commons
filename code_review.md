# Peer Review: Multi-Agent Commons Simulation Framework

**Reviewer:** Veteran Reviewer, Journal of Artificial Intelligence Research  
**Manuscript status:** Major Revision Required  
**Date:** 2026-04-06

---

## Summary of Contribution

This manuscript presents an LLM-driven multi-agent simulation of the Tragedy of the Commons, where heterogeneous language model agents compete for a shared freshwater aquifer under varying institutional conditions (cheap talk, pledges, binding quotas, sanctions, contracts). The framework is architecturally ambitious: it combines multiple LLM providers via Groq, implements logistic-growth resource dynamics, supports dual prompt framings (benchmark vs. naturalistic), introduces observation noise and demand shocks for ecological validity, and computes a rich battery of governance metrics including counterfactual pledge-policy replay. The experimental design targets six concrete hypotheses about rhetoric-action divergence, institutional effectiveness, and prompt framing effects.

**Verdict:** The research question is timely and well-motivated. The engineering is substantially above-average for this kind of work. However, several design choices introduce **threats to internal validity** that must be addressed before the results can support causal claims. I recommend a **major revision** focused on the issues below.

---

## Strengths

### S1. Thoughtful Experimental Design
The experiment matrix ([EXPERIMENT_MATRIX.md](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/EXPERIMENT_MATRIX.md)) is a genuine strength. The six hypotheses are well-structured, the institutional conditions form a clean progression (no communication → cheap talk → ledger → pledges → quotas → sanctions → contracts), and the inclusion of scripted baselines is essential for separating LLM behavior from trivially explainable threshold policies.

### S2. Alias-Based Bias Masking
Using anonymous aliases (`Organization A`, `Organization B`) in the public ledger while preserving internal names for logging ([agent.py:66](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/agent.py#L66), [simulation.py:706](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/simulation.py#L706)) is an excellent methodological decision. This mitigates positional and name-salience biases known to affect LLM reasoning, which is a common oversight in multi-agent LLM research.

### S3. Chain-of-Thought Scratchpad
The `private_scratchpad` field ([agent.py:22-25](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/agent.py#L22-L25)) that agents fill before declaring their action is a sound cognitive patch. It encourages deliberative reasoning while keeping the reasoning private (not broadcast to other agents). This is well-designed.

### S4. Robust Failure Handling
The multi-model fallback chain with exponential backoff ([agent.py:393-415](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/agent.py#L393-L415)), conservative fallback actions ([agent.py:306-325](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/agent.py#L306-L325)), and the 3-consecutive-fallback circuit breaker ([simulation.py:889-895](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/simulation.py#L889-L895)) demonstrate engineering maturity. The JSONL checkpointing ([simulation.py:1009-1010](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/simulation.py#L1009-L1010)) protects against data loss during long runs.

### S5. Counterfactual Pledge Replay
The `governance_credibility_metrics` function ([metrics.py:408-553](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/metrics.py#L408-L553)) replays what *would* have happened if agents followed their own pledges. This is an analytically novel approach to measuring the rhetoric-action gap and is the strongest methodological contribution.

### S6. Post-hoc LLM Judge with Provenance
The [judge_messages.py](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/judge_messages.py) module is properly isolated from agent decisions, uses a distinct model by default, and records provider/model/rubric versioning. This separates measurement from intervention, which is critical for reproducibility.

---

## Major Issues (Must Address)

### M1. Concurrent Agent Queries Break Determinism

> **Severity: Critical (Reproducibility)**

```python
# simulation.py:839-885
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    futures = {
        pool.submit(_query_agent, ...): agent
        for agent, env_summary in query_args
    }
    for future in concurrent.futures.as_completed(futures):
        name, action = future.result()
        actions[name] = action
```

`as_completed` returns futures in **arrival order**, which is non-deterministic. While you re-sort actions on line 887, the **shared `rng` object** ([simulation.py:757](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/simulation.py#L757)) is consumed **before** the concurrent block (in `_observed_env_summary`, lines 843-852), so the RNG state appears safe for *this* round. However:

1. The observation summaries are generated **sequentially** inside the thread pool setup loop, but the `rng` is shared across agents. If the loop order ever changes, or if `_observed_env_summary` is parallelized, the noise draws will diverge.
2. The `random` module's `Random` class is **not thread-safe** — concurrent reads from the same `rng` instance would produce undefined behavior.

**Recommendation:** Either (a) give each agent a **per-agent `rng` fork** seeded deterministically from the master seed, or (b) pre-draw all stochastic values before entering the thread pool. Document that agent-query parallelism is for latency only and does not affect state.

---

### M2. Symmetric Memory Creates an Information Asymmetry Artifact

> **Severity: High (Internal Validity)**

All agents see the same `agent_standings` block and the same `messages_for_agents` dict. However, the standings block uses `agent.alias` for *other* agents but gives each agent its own full history ([agent.py:234-248](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/agent.py#L234-L248)). Crucially, **each agent sees its own private history including what it requested, received, and what community messages were sent** — but it only sees *current-round* messages from others, not their full request histories.

This is fine as a design choice, but it means agents have **asymmetric information depth about self vs. others** that isn't documented. When you claim agents have "symmetric memory" (from past conversation context), this is misleading.

**Recommendation:** Explicitly document the memory model in the manuscript: agents have deep self-memory (sliding window of 5 rounds) but only current-round observations of others' messages and aggregate standings.

---

### M3. The `private_scratchpad` Is Not Provably Private

> **Severity: High (Methodological)**

The scratchpad text is included in the JSON response and **logged in the turn exports** ([simulation.py:989](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/simulation.py#L989) via `reasonings`). It is *not* included in the `messages_for_agents` dict sent to other agents — good. But there's a subtle issue: the `reasonings` variable on line 898 captures `action.reasoning` (the *public* rationale), not `private_scratchpad`. So the scratchpad is **not** shown anywhere in the UI or inter-agent communication. However, it **is** persisted in the JSON export inside each agent's history but only via the Action object, not via `turn_exports`.

Wait — actually, `reasonings` only captures the public-facing `reasoning` field. The `private_scratchpad` is part of the `Action` object but is never explicitly serialized to `turn_exports`. It's effectively lost unless someone reconstructs it from the raw LLM response.

**Recommendation:** Explicitly serialize `private_scratchpad` in `turn_exports` (under a `scratchpads` key) for post-hoc analysis of agent reasoning quality. This data is invaluable for qualitative analysis of whether CoT actually improved decision-making.

---

### M4. Demand Regime Parameterization Couples Need to Cap

> **Severity: High (Ecological Validity)**

```python
# simulation.py:598
base_need = max(1, round(cap * demand_weight * demand_multiplier))
```

Private need is derived from `cap * demand_weight * demand_multiplier`, where `cap = pool * max_harvest_pct`. This means **private need decreases as the pool depletes**, creating a feedback loop where agents *need* less precisely when scarcity bites hardest. This is ecologically backwards — real-world water demand is driven by crop cycles, cooling systems, and human consumption, not by how much water happens to be left.

This means subsistence penalties (deficit × 2) become less punishing as the pool drains, undermining the intended subsistence-constraint mechanism.

**Recommendation:** Decouple private need from pool state. Use a fixed base need (e.g., `initial_cap * demand_weight * demand_multiplier`) or sample from a distribution anchored to initial conditions. The current design conflates the observation and the state being observed.

---

### M5. No Interface Contract Between `LLMAgent` and `ScriptedAgent`

> **Severity: Medium-High (Maintainability / Correctness)**

`LLMAgent` and `ScriptedAgent` share an identical public API (`decide`, `record_turn`, identical instance attributes) but have **no common base class or Protocol**. This is a duck-typing tightrope:

- If any attribute is added to one and not the other, silent failures will occur at runtime.
- `_make_agents` returns `list[LLMAgent]` ([simulation.py:703](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/simulation.py#L703)) but actually returns mixed types.
- The type annotations throughout `simulation.py` declare `list[LLMAgent]`, which is a lie when scripted agents are in the roster.

**Recommendation:** Define a `Protocol` (or abstract base class):

```python
class Agent(Protocol):
    name: str
    alias: str
    inventory: int
    reputation: int
    history: list[dict]
    extraction_weight: float
    # ...
    def decide(self, ...) -> Action: ...
    def record_turn(self, ...) -> None: ...
```

---

### M6. Insufficient Test Coverage

> **Severity: Medium-High (Reproducibility)**

The only test file is [test_realism.py](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/tests/test_realism.py) with 4 test cases. There are **no tests** for:

- `CommonsEnv.process_turn` (proportional allocation, depletion boundary)
- `_apply_request_rules` (quota clamping, moratorium zeroing)
- `_calculate_penalties` (sanction arithmetic)
- `_next_contract_quota` (median voting logic)
- `_normalize_action_data` (schema drift handling, clamping)
- `gini` coefficient (known values)
- `governance_credibility_metrics` (failure mode classification)
- The counterfactual replay engine (`_replay_policy`)

For a journal submission, **every metric and every mechanistic function must have deterministic unit tests**. Readers need to trust that the metrics measure what the paper claims.

**Recommendation:** Add at minimum:
- `test_environment.py`: proportional allocation edge cases, depletion, growth at boundary
- `test_metrics.py`: gini, governance credibility, replay policy, failure mode classifier
- `test_simulation.py`: request rules, penalty calculation, contract voting

---

## Minor Issues

### m1. Hardcoded Configuration Parameters

`MAX_ROUNDS`, `INITIAL_POOL`, `MAX_CAPACITY`, `GROWTH_RATE`, `MAX_HARVEST_PCT` are module-level constants ([simulation.py:129-133](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/simulation.py#L129-L133)) also hardcoded inside `run_simulation`'s export block ([simulation.py:1079-1085](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/simulation.py#L1079-L1085)). If someone changes the constants at the top, the export will still use the module-level values — but this creates fragility. Consider passing these through `ExperimentConfig` or a unified `SimulationConfig` dataclass.

### m2. `env_summary` Variable Shadowing in Need Visibility Block

```python
# simulation.py:864-867
query_args = [
    (agent, f"{env_summary}{need_block}")
    for agent, env_summary in query_args
]
```

The list comprehension **shadows** the loop variable `env_summary` from the outer scope. This works correctly because the outer `env_summary` is the last agent's value and is overwritten, but it's confusing and brittle. Use a different variable name.

### m3. `reasoning` vs `rationale` Alias Ambiguity

The `Action` model declares `reasoning` with `alias="rationale"` ([agent.py:48-52](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/agent.py#L48-L52)). Throughout the codebase, both `.reasoning` and `"rationale"` are used. This dual naming will confuse contributors and is error-prone during serialization. Choose one name.

### m4. Reputation Gravity Tax Scale

```python
# simulation.py:946
reputation_taxes[agent.name] = max(0, -agent.reputation) * 1
```

The `* 1` multiplier is a no-op. Either parameterize this (e.g., `reputation_tax_rate`) or document why it's intentionally 1×. Currently it looks like a placeholder that was never tuned.

### m5. Missing `reported_need` in Scripted Agents

`ScriptedAgent.decide()` never sets `reported_need` on the returned `Action`. The field defaults to `None` via Pydantic. This means scripted agents can never participate in need-visibility experiments, silently producing `null` entries in demand accounting. Either add a `reported_need` to scripted actions or raise when `need_visibility != "private"` with scripted rosters.

### m6. Growth Function Epsilon Hack

```python
# environment.py:13-14
def growth_units(growth: float) -> int:
    return max(0, int(growth + 1e-9))
```

The `1e-9` epsilon is a floating-point rounding guard, but `int()` truncates toward zero. For growth values like `4.9999999999`, `int(5.0000000009)` = `5` — correct. But for negative growth (which can't happen due to `max(0, ...)`), this would silently round up. The logic is safe but should be documented.

### m7. SVG Plot Has No Y-Axis Labels

[report_run.py:60-117](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/report_run.py#L60-L117) renders a detailed SVG plot but omits Y-axis tick labels and gridlines. For a journal figure, this is insufficient.

### m8. No `__init__.py` in Project Root

The project uses relative imports (`from agent import ...`) but has no `__init__.py`. This works when running from the `commons_sim` directory but will break if imported as a package.

---

## Architectural Observations

### A1. Monolithic `simulation.py` (1252 lines)

This file handles configuration, agent construction, round orchestration, demand calculation, quota computation, penalty calculation, contract voting, moratorium logic, result serialization, CLI parsing, and suite management. It should be decomposed:

| Responsibility | Suggested Module |
|---|---|
| Configuration dataclasses + rosters | `config.py` |
| Round orchestration loop | `simulation.py` (reduced) |
| Institutional mechanics (quotas, penalties, contracts) | `institutions.py` |
| Demand/observation pipeline | `observations.py` |
| CLI + suite runner | `cli.py` or `__main__.py` |

### A2. No Logging Framework

The codebase uses `print()` for warnings ([simulation.py:766](file:///Users/karan/Desktop/Projects/multiagent/commons_sim/simulation.py#L766)) and Rich console for UI output. There is no structured logging for debugging API errors, fallback chains, or run metadata. For a research tool that makes hundreds of API calls, `logging` with configurable levels is essential.

### A3. `results.json` in Project Root

There's a stale `results.json` (14KB) sitting alongside the source code. This should be `.gitignore`d.

---

## Questions for the Authors

1. **H1 directionality:** You hypothesize communication increases cooperative *rhetoric* more than it reduces *extraction*. But your framework also introduces subsistence penalties and reputation gravity, which create extraction pressure independent of communication. How do you plan to disentangle these effects?

2. **Temperature choice:** The default temperature of 0.8 is quite high for a study measuring behavioral consistency. Have you run sensitivity analyses at `temperature=0.0`? The `--temperature` flag exists but the experiment matrix doesn't specify a temperature sweep.

3. **Model heterogeneity confound:** In the `heterogeneous` roster, you assign different *models* to different *personas*. AlphaCorp gets Llama-70B; EcoTrust gets GPT-OSS-120B. This means you cannot separate the effect of *persona* from *model capability*. The homogeneous rosters help, but the primary heterogeneous condition has this confound baked in.

4. **Demand regime in experiment matrix:** The matrix specifies institutional conditions and rosters but doesn't specify which demand regime to use. The CLI defaults to `medium`. Is this intentional?

5. **Pledge sufficiency threshold:** The `governance_credibility_metrics` function considers a pledge "sufficient" if `mean_pledge <= sustainable_share`. But sustainable share is weighted by extraction weight, which means powerful agents have higher sustainable shares. Is this the intended equity model?

---

## Recommendation

**Major Revision.** The framework is architecturally sound and addresses a genuinely interesting question. The experimental design is above the bar. However, the reproducibility threats (M1, M6), the need-cap coupling (M4), and the type-safety gaps (M5) must be resolved before the results can be trusted for quantitative claims. The private scratchpad serialization (M3) is an easy win that would substantially strengthen qualitative analysis.

The codebase is clearly the work of someone who understands both the game theory and the engineering. With the revisions above, this has the bones to be a strong contribution.

---

## Summary Table

| Issue | Severity | Effort | Section |
|---|---|---|---|
| M1: Thread-unsafe RNG / non-determinism | Critical | Low | Reproducibility |
| M2: Asymmetric memory documentation | High | Low | Validity |
| M3: Scratchpad not serialized | High | Low | Methodology |
| M4: Need coupled to pool state | High | Medium | Validity |
| M5: No Agent Protocol/ABC | Medium-High | Medium | Maintainability |
| M6: Insufficient test coverage | Medium-High | High | Reproducibility |
| m1: Hardcoded config duplication | Minor | Low | Maintainability |
| m2: Variable shadowing | Minor | Low | Code quality |
| m3: reasoning/rationale alias | Minor | Low | Code quality |
| m4: Reputation tax no-op multiplier | Minor | Low | Design |
| m5: Scripted agents missing reported_need | Minor | Low | Correctness |
| m6: Growth epsilon undocumented | Minor | Low | Documentation |
| m7: SVG missing Y-axis labels | Minor | Medium | Reporting |
| m8: Missing `__init__.py` | Minor | Low | Packaging |
