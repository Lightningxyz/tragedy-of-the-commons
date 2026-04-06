# Tragedy of the Commons Multi-Agent Simulation - Detailed Code Explanation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Module-by-Module Explanation](#module-by-module-explanation)
   - [agent.py](#agentpy)
   - [environment.py](#environmentpy)
   - [simulation_core.py](#simulation_corepy)
   - [config.py](#configpy)
   - [institutions.py](#institutionspy)
   - [metrics.py](#metricspy)
   - [observations.py](#observationspy)
   - [scripted_agents.py](#scripted_agentspy)
   - [cli.py](#clipy)
   - [ui.py](#uipy)
   - [judge_messages.py](#judge_messagespy)
   - [analyze_results.py](#analyze_resultspy)
   - [report_run.py](#report_runpy)
   - [logging_utils.py](#logging_utilspy)
4. [Data Flow](#data-flow)
5. [Key Design Patterns](#key-design-patterns)
6. [Configuration and Experiments](#configuration-and-experiments)

---

## Overview

This is a **multi-agent simulation framework** that models the "Tragedy of the Commons" - a classic game theory scenario where multiple parties share a limited resource (in this case, a freshwater aquifer). The framework uses **Large Language Models (LLMs)** as agents, allowing them to make decisions about resource extraction, communicate with each other, and respond to various institutional rules.

### Core Concepts

1. **Resource Pool**: A shared aquifer with logistic growth dynamics (replenishes naturally but can be depleted)
2. **Agents**: LLM-powered entities representing organizations that extract water
3. **Rounds**: The simulation runs for a fixed number of quarters (default: 15)
4. **Institutions**: Different governance mechanisms (communication, pledges, quotas, sanctions, contracts)
5. **Metrics**: Comprehensive analysis of behavior, cooperation, and sustainability

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Entry Point                      │
│                         (cli.py)                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Simulation Core                           │
│                   (simulation_core.py)                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Main Loop: For each round                          │   │
│  │    1. Generate observations for each agent          │   │
│  │    2. Query agents for decisions (parallel)         │   │
│  │    3. Apply institutional rules                     │   │
│  │    4. Process environment turn                      │   │
│  │    5. Calculate penalties and update state          │   │
│  │    6. Record and export results                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │                │                │
┌─────────▼──────┐  ┌─────▼──────────┐  ┌──▼───────────────┐
│    Agents      │  │   Environment  │  │   Institutions   │
│   (agent.py)   │  │ (environment.  │  │ (institutions.   │
│                │  │     py)        │  │      py)         │
│ • LLMAgent     │  │                │  │                  │
│ • ScriptedAgent│  │ • Pool dynamics│  │ • Quotas         │
│ • Memory       │  │ • Growth       │  │ • Penalties      │
│ • Decisions    │  │ • Allocation   │  │ • Contracts      │
└────────────────┘  └────────────────┘  └──────────────────┘
          │                │                │
┌─────────▼────────────────▼────────────────▼───────────────┐
│                      Metrics & Analysis                     │
│  (metrics.py, analyze_results.py, report_run.py)           │
│                                                             │
│  • Survival analysis    • Governance credibility           │
│  • Inequality (Gini)    • Policy adequacy                  │
│  • Message coding       • Counterfactual replay            │
│  • Demand accounting      (what if pledges were binding?)  │
└─────────────────────────────────────────────────────────────┘
```

---

## Module-by-Module Explanation

### agent.py

**Purpose**: Implements the LLM-powered agents that make extraction decisions.

#### Key Components:

1. **Action Class** (lines 21-61)
   ```python
   class Action(BaseModel):
       private_scratchpad: str      # Internal reasoning (not shared)
       resource_request: int        # How much to extract
       pledge_next_request: int|None # Promise for next round
       reported_need: int|None      # Public stated need
       proposed_quota: int|None     # Suggested limit for all
       accept_binding_quota: bool   # Accept enforced limits?
       reasoning: str               # Public explanation
       message: str                 # Broadcast to others
   ```
   This is the structured output schema that every agent must produce. Pydantic validates the data types and constraints.

2. **AgentProtocol** (lines 65-113)
   A `Protocol` (structural interface) that defines what methods and attributes any agent must have:
   - `decide()`: Make a decision given the current state
   - `record_turn()`: Remember what happened this round
   - Various attributes: `name`, `inventory`, `reputation`, `history`, etc.

3. **LLMAgent Class** (lines 196-568)
   The main agent implementation:

   **Initialization** (`__init__`):
   - Stores agent configuration (name, persona, model, provider)
   - Initializes state: `inventory=0`, `reputation=0`, `history=[]`
   - Sets up LLM clients (Gemini or Groq)

   **System Prompts** (lines 121-193):
   Two templates:
   - `BENCHMARK_SYSTEM_TEMPLATE`: Direct game framing ("extraction", "rounds")
   - `NATURALISTIC_SYSTEM_TEMPLATE`: Realistic framing ("water requests", "quarters", "organizations")
   
   Both explain the resource dynamics, subsistence constraints, and reputation effects.

   **Memory Model** (lines 199-207):
   - Each agent maintains a **sliding window of 5 past rounds**
   - They see their own full history (requests, grants, penalties, community messages)
   - They see only **current-round** information about others (standings, messages)
   - This creates **asymmetric information depth**: deep self-knowledge, shallow other-knowledge

   **Decision Process** (`decide()` method, lines 451-537):
   1. Build system prompt (includes institutional rules)
   2. Build user prompt (includes current state, history, messages from others)
   3. Try primary model with up to 2 retries
   4. If fails, try fallback models
   5. If all fail, return conservative fallback action
   6. Exponential backoff for rate limit errors (429)

   **Action Normalization** (`_normalize_action_data()`, lines 341-376):
   - Handles schema drift from weaker models
   - Clamps values to valid ranges
   - Provides defaults for missing fields
   - Truncates text to max lengths

   **Fallback Action** (`_fallback_action()`, lines 379-406):
   - Returns a safe, conservative action when the LLM fails
   - Requests 1/4 of the extraction cap
   - Logs the error for debugging

   **Recording Turns** (`record_turn()`, lines 540-568):
   - Updates inventory (add granted, subtract penalties)
   - Updates reputation (decrease if pledge violated)
   - Appends to history (sliding window keeps last 5)

---

### environment.py

**Purpose**: Implements the shared resource pool with logistic growth dynamics.

#### Key Components:

1. **growth_units()** (lines 12-20)
   ```python
   def growth_units(growth: float) -> int:
       return max(0, int(growth + 1e-9))
   ```
   Converts continuous logistic growth to integer units. The `1e-9` epsilon handles floating-point rounding (e.g., 4.9999999999 → 5).

2. **TurnResult** (lines 23-33)
   Immutable record of a round's outcome:
   - `round_num`: Which round
   - `requested`: What each agent asked for
   - `granted`: What each agent actually received
   - `remaining_before/after`: Pool levels
   - `replenished`: How much grew back
   - `depleted`: Did the pool reach zero?

3. **CommonsEnv Class** (lines 35-165)
   The resource pool simulation:

   **Initialization**:
   - `pool`: Current water level (starts at `initial_pool`, default 100)
   - `max_capacity`: Maximum pool size (default 120)
   - `growth_rate`: How fast it replenishes (default 0.3 = 30%)
   - `max_harvest_pct`: Max fraction any agent can request (default 0.25 = 25%)

   **Logistic Growth Formula**:
   ```
   growth = growth_rate × pool × (1 - pool/max_capacity)
   ```
   - Growth is **maximized at half capacity** (K/2 = 60)
   - Growth **approaches zero** near capacity (120) or depletion (0)
   - This creates a "sweet spot" for sustainable harvesting

   **Per-Agent Cap** (`per_agent_cap` property):
   ```python
   @property
   def per_agent_cap(self) -> int:
       return max(1, int(self.pool * self.max_harvest_pct))
   ```
   - Each agent can request up to 25% of current pool
   - Minimum of 1 (so requests are always possible)
   - **Decreases as pool depletes** (creating scarcity pressure)

   **Initial Per-Agent Cap** (`initial_per_agent_cap` property):
   - Anchored to the **initial pool size** (100), not current pool
   - Used for demand calculations to avoid coupling private need to pool state
   - This is important: agents' **private needs don't shrink** as the aquifer depletes

   **Proportional Allocation** (`_allocate_proportionally()`, lines 82-106):
   When total demand exceeds supply:
   1. Calculate exact shares: `agent_share = (agent_request / total_request) × pool`
   2. Grant integer parts: `granted[agent] = int(exact_share)`
   3. Distribute remainder using **largest-remainder method**:
      - Sort by fractional remainder (descending)
      - Give 1 extra unit to top agents until remainder exhausted
   This ensures fair distribution when water is scarce.

   **Process Turn** (`process_turn()`, lines 109-147):
   The core simulation step:
   1. **Clamp requests** to per-agent cap
   2. **Check total demand**:
      - If demand ≤ pool: everyone gets what they requested
      - If demand > pool: proportional allocation
   3. **Subtract extraction** from pool
   4. **Calculate logistic growth** and add back
   5. **Cap at max_capacity** (can't exceed carrying capacity)
   6. **Record result** in turn history
   7. **Return TurnResult** with all details

---

### simulation_core.py

**Purpose**: The main orchestration loop that runs the simulation.

#### Key Functions:

1. **_ensure_logging_configured()** (lines 50-65)
   Sets up basic logging if not already configured. CLI paths configure logging explicitly; this is for library usage.

2. **_query_agent()** (lines 68-87)
   Async wrapper that calls `agent.decide()` and returns `(agent_name, action)`.

3. **_timeout_fallback_action()** (lines 90-118)
   Creates a safe fallback action when an agent query times out:
   - Requests 1/4 of cap
   - Sets fallback flags on agent
   - Logs the error

4. **_build_agent_rngs()** (lines 121-129)
   Creates per-agent random number generators:
   - If seed provided: deterministic forks (reproducible)
   - If seed=None: entropy-backed RNGs (stochastic)
   This ensures each agent gets consistent noise/demand shocks.

5. **_build_standings()** (lines 132-148)
   Formats the public leaderboard shown to all agents:
   - Each agent's inventory, reputation, stake
   - Uses aliases (Organization A, B, C) for anonymity
   - Different labels for benchmark vs naturalistic mode

6. **_make_agents()** (lines 151-195)
   Factory function that creates agents from roster configuration:
   - If provider="scripted": creates `ScriptedAgent`
   - Otherwise: creates `LLMAgent` with appropriate model/client
   - Assigns aliases (Organization A, B, C, ...)

#### Main Simulation Loop: **run_simulation()** (lines 198-640)

This is the heart of the system. Here's the flow:

**Setup Phase** (lines 212-270):
1. Configure logging
2. Load experiment config, roster, realism profile, demand regime
3. Initialize LLM clients (Gemini if needed, Groq if API key present)
4. Create environment (`CommonsEnv`)
5. Create agents (`_make_agents()`)
6. Build per-agent RNGs
7. Render intro UI (if `render=True`)

**State Tracking** (lines 271-287):
- `recent_messages`: Public statements from last round
- `pledges_due`: What each agent pledged for this round
- `recent_reported_needs`: What each agent reported needing
- `binding_quota`: Current enforced quota (if any)
- `consecutive_fallbacks`: Track agent API failures
- `checkpoint_path`: JSONL file for crash recovery

**Main Loop** (lines 289-568):
For each round (1 to max_rounds):

1. **Check Moratorium** (lines 297-310):
   - If pool falls below threshold, activate moratorium
   - No extraction allowed for N rounds

2. **Build Context** (lines 312-328):
   - Standings (who has what)
   - Institution rules (what's allowed)
   - Institution state (current pledges, ledger, quotas)
   - Messages from others (if communication enabled)

3. **Generate Observations** (lines 330-355):
   For each agent:
   - Call `_observed_env_summary()` to generate their view
   - This includes: pool level (possibly noisy/delayed), private need (if demand regime active)
   - Build `query_args` list with (agent, env_summary) pairs

4. **Add Need Visibility Block** (lines 347-355):
   - If `need_visibility="audited"`: show verified private needs
   - If `need_visibility="public"`: show reported needs from last round
   - Append to each agent's prompt

5. **Query Agents** (lines 357-383):
   - Run `_query_agent()` for each agent in parallel (asyncio)
   - Timeout per agent: `agent_timeout_seconds` (default 60s)
   - If timeout: use `_timeout_fallback_action()`
   - Collect all actions

6. **Check Fallback Chain** (lines 386-402):
   - Track consecutive fallbacks per agent
   - If agent uses fallback 3 times consecutively: **abort** (data poisoning risk)

7. **Extract Action Data** (lines 404-421):
   - Raw requests, reasonings, scratchpads
   - Messages, pledges, proposed quotas
   - Reported needs, quota acceptances

8. **Apply Institutional Rules** (lines 423-448):
   - **Binding Quotas**: If enabled and round ≥ quota_start_round, compute quota
   - **Effective Requests**: Apply quota caps, moratorium zeroing
   - **Penalties**: Calculate for broken pledges (if sanctions enabled)
   - **Sustainable Share**: Calculate what each agent *should* extract

9. **Process Environment Turn** (line 450):
   - Call `env.process_turn(effective_requests)`
   - Returns `TurnResult` with allocations, new pool level, growth

10. **Calculate Penalties** (lines 452-467):
    - **Subsistence penalties**: If agent didn't meet private need → deficit × 2
    - **Reputation taxes**: If reputation negative → recurring penalty
    - **Pledge penalties**: If broke pledge + sanctions enabled → violation × multiplier
    - Total penalty = sum of all three

11. **Record Turn** (lines 468-478):
    For each agent, call `agent.record_turn()` with:
    - Round number, requested, granted, pool_after
    - Pledge, penalty, pledge_violation
    - Community messages (what others said)

12. **Export Turn Data** (lines 480-529):
    - Message codes (regex-based rhetoric analysis)
    - Model status (which model was used, fallback info)
    - Observations (what each agent saw)
    - All pledges, penalties, quotas, moratorium status
    - Write to JSONL checkpoint file

13. **Render UI** (lines 531-548):
    - Show round header, pool visualization
    - Agent scoreboard (requested, granted, inventory, messages)
    - Reasoning panels for each agent

14. **Update State** (lines 550-560):
    - `pledges_due` = this round's pledges (for next round's penalty calculation)
    - Check for contract quota (if contracts enabled)
    - Decrement moratorium counter

15. **Check Depletion** (lines 562-564):
    - If pool ≤ 0: break (game over)

16. **Sleep** (lines 566-567):
    - Optional delay between rounds (for UI readability)

**Export Phase** (lines 569-639):
1. Render finale UI (final standings, outcome)
2. Build agent exports (final state, full history)
3. Build run config (parameters)
4. Compute metrics via `summarize_run()`
5. Export full JSON with all data
6. Log completion

---

### config.py

**Purpose**: Centralized configuration for experiments, agents, and parameters.

#### Key Components:

1. **AGENT_ROSTER** (lines 10-68)
   Default heterogeneous roster with 4 agents:
   - **AlphaCorp**: Profit-driven megacorp, uses Llama-70B, high extraction weight (1.6)
   - **EcoTrust**: Conservation foundation, uses GPT-OSS-120B, low extraction (0.6)
   - **LocalFarmer**: Subsistence farmer, uses Qwen-32B with fallbacks, moderate extraction (1.0)
   - **TechVenture**: Risk-taking startup, uses Kimi-K2, high extraction (1.3)

   Each has:
   - `persona`: Character description fed to the LLM
   - `stake`: Role description (shown to other agents)
   - `extraction_weight`: How much they "deserve" (affects sustainable share calculation)
   - `demand_weight`: How much private need they have (0.25-1.25× multiplier)

2. **rotated_model_roster()** (lines 71-97)
   Keeps personas fixed but rotates model assignments. Used for sensitivity analysis:
   - `heterogeneous_rot0`: Original assignment
   - `heterogeneous_rot1`: Shift by 1
   - `heterogeneous_rot2`: Shift by 2
   - `heterogeneous_rot3`: Shift by 3
   This helps separate persona effects from model capability effects.

3. **homogeneous_roster()** (lines 100-105)
   All agents use the same model. Used for controlled comparisons:
   - `single_model_llama`: All use Llama-70B
   - `single_model_gpt_oss`: All use GPT-OSS-120B
   - `single_model_qwen`: All use Qwen-32B

4. **ROSTERS** (lines 108-133)
   Dictionary of all available rosters:
   - Heterogeneous (4 variants with model rotation)
   - Homogeneous (3 variants, one per model)
   - Scripted baselines (Maximizer, Steward, TitForTat, ThresholdGreedy)
   - Scripted stewards (4 cooperative strategies)

5. **Parameters** (lines 136-140):
   ```python
   MAX_ROUNDS = 15          # Default game length
   INITIAL_POOL = 100       # Starting water level
   MAX_CAPACITY = 120       # Carrying capacity
   GROWTH_RATE = 0.3        # 30% logistic growth
   MAX_HARVEST_PCT = 0.20   # 20% per-agent cap
   ```

6. **ExperimentConfig** (lines 143-158)
   Dataclass defining institutional conditions:
   - `communication`: Allow public statements?
   - `public_ledger`: Show past requests/penalties?
   - `pledges`: Allow promises for future behavior?
   - `binding_quotas`: Enforce extraction limits?
   - `sanctions`: Penalty for broken pledges?
   - `contracts`: Majority-voted binding quotas?
   - `quota_fraction`: How strict is the quota? (0.8 = 80% of sustainable)
   - `moratorium_threshold`: Pool level triggering no-extraction period
   - `moratorium_rounds`: How long does moratorium last?

7. **EXPERIMENTS** (lines 203-284)
   Pre-defined experiment configurations:
   - `no_communication`: Baseline, no talking
   - `cheap_talk`: Public statements, no enforcement
   - `public_ledger`: Past behavior is visible
   - `nonbinding_pledges`: Promises without consequences
   - `binding_quotas`: Enforced limits (various start times)
   - `adaptive_quota`: Quota recalculated each round
   - `mandatory_moratorium`: Emergency shutdown at low pool
   - `sanctions`: Penalties for lying
   - `contracts`: Democratic quota setting

8. **RealismConfig** (lines 160-166)
   Controls observation quality:
   - `observation_noise_pct`: Add noise to pool reports (0% = perfect, 10% = field)
   - `report_delay_rounds`: Delay in pool reports (0 = current, 1 = last round's level)
   - `private_demand_shock_pct`: Random variation in private need (0% = stable, 35% = volatile)

9. **REALISM_PROFILES** (lines 177-185):
   - `perfect`: No noise, no delay, no shocks
   - `field`: 10% noise, 1-round delay, 35% demand shocks

10. **DemandRegime** (lines 168-175)
    Controls private operational need:
    - `multiplier`: Base need as fraction of initial cap
    - `crisis_rounds`: Number of crisis rounds at start
    - `crisis_multiplier`: Higher need during crisis
    - `post_crisis_multiplier`: Need after crisis ends

11. **DEMAND_REGIMES** (lines 188-200):
    - `none`: No private need (multiplier=0)
    - `low`: 6% of initial cap
    - `medium`: 25% of initial cap
    - `high`: 70% of initial cap
    - `crisis`: 20% base, 90% for first 2 rounds, then 20%

---

### institutions.py

**Purpose**: Implements the mechanics of governance institutions (quotas, penalties, contracts).

#### Key Functions:

1. **_institution_rules()** (lines 29-147)
   Generates the rules text shown to agents in their system prompt:
   - Explains what communication is allowed
   - Describes pledges and whether they're enforced
   - Explains binding quotas and when they apply
   - Describes moratorium rules
   - Explains sanctions for broken pledges
   - Describes contract voting mechanism
   - Different wording for benchmark vs naturalistic mode

2. **_institution_state()** (lines 150-189)
   Generates the current institutional state shown each round:
   - Current binding quota (if any)
   - Moratorium remaining rounds
   - Pledges due this round (what each agent promised)
   - Public ledger (reputation, penalties, last requests, violations)

3. **_weighted_sustainable_share()** (lines 192-204)
   Calculates how much each agent *should* extract for sustainability:
   ```python
   total_growth_budget = growth_rate × pool × (1 - pool/capacity)
   agent_share = total_growth_budget × (agent_weight / total_weight)
   ```
   - Only the **growth** can be sustainably harvested
   - Divided proportionally by extraction weight
   - Powerful agents (high weight) get larger sustainable shares

4. **_apply_request_rules()** (lines 207-222)
   Transforms raw requests into effective requests:
   - If moratorium active: all requests → 0
   - If binding quotas: clamp to min(request, quota, cap)
   - If contracts: same clamping as quotas

5. **_calculate_penalties()** (lines 225-240)
   Computes penalties for broken pledges:
   - For each agent, check if `requested > pledged_limit`
   - Violation = `max(0, requested - pledged_limit)`
   - If sanctions enabled: penalty = violation × sanction_multiplier
   - Returns (penalties_dict, violations_dict)

6. **_next_contract_quota()** (lines 243-253)
   Implements democratic quota setting:
   - Collect proposed quotas from agents who accept binding quotas
   - If majority accepts (len(acceptors) > len(actions)/2):
     - Quota = median of proposed quotas
   - Otherwise: no contract this round

7. **_compute_binding_quota()** (lines 256-263)
   Calculates the binding quota:
   - Fair share = sustainable_share / num_agents
   - Quota = fair_share × quota_fraction
   - Minimum of 1 (always allow some extraction)

---

### metrics.py

**Purpose**: Comprehensive analysis of simulation outcomes.

#### Key Functions:

1. **gini()** (lines 53-64)
   Calculates Gini coefficient (inequality measure):
   - 0 = perfect equality (everyone has same)
   - 1 = perfect inequality (one person has everything)
   - Formula: `(2 × weighted_sum) / (n × total) - (n+1)/n`

2. **message_cooperation_score()** (lines 67-69)
   Counts cooperative terms in a message:
   - Regex matches: sustain, conserve, cooperate, fair, protect, pledge, etc.
   - Returns count (higher = more cooperative rhetoric)

3. **code_message()** (lines 72-77)
   Multi-label rhetoric analysis:
   - `cooperative`: Pro-social language
   - `extractive`: Profit/growth language
   - `threat`: Enforcement/sanction language
   - `moral_appeal`: Fairness/future generations
   - `quota_proposal`: Rule-setting language
   - `blame`: Accusations of others

4. **ReplayState** (lines 80-107)
   Simulates a counterfactual policy:
   - Takes a sequence of requests
   - Runs them through the environment dynamics
   - Returns whether the pool survived and final level
   - Used to answer: "What if agents had followed their pledges?"

5. **_replay_policy()** (lines 110-136)
   Replays a counterfactual extraction policy:
   - Starts from initial conditions
   - For each round, uses policy requests (instead of actual requests)
   - Processes through environment
   - Returns: survived?, collapse_round, final_pool

6. **_pledge_policy_for_turns()** (lines 139-153)
   Extracts pledges as a counterfactual policy:
   - For each round, use next-round pledges as the policy
   - If no pledge, use last round's policy (inertia)
   - Returns list of policy dicts (one per round)

7. **demand_accounting_metrics()** (lines 156-269)
   Analyzes the relationship between private need and behavior:
   - Request-to-need ratio (are agents asking for more than they need?)
   - Pledge-to-need ratio (are pledges realistic?)
   - Need satisfaction rate (do agents get what they need?)
   - Over-need request rate (how often do agents over-request?)
   - Under-need pledge rate (how often do agents under-promise?)
   - Scarcity pressure (total need / pool level)
   - Per-agent demand profiles

8. **policy_adequacy_metrics()** (lines 306-405)
   Evaluates whether governance is sufficient for sustainability:
   - Compares need, requests, pledges against **safe yield** (ecological limit)
   - Classifies each round:
     - `within_safe_yield`: Everything is sustainable
     - `greed_failure`: Requesting more than needed
     - `sacrifice_failure`: Need exceeds safe yield
     - `governance_failure`: Pledges exceed safe yield
     - `scarcity_governance_failure`: Need + pledges both exceed
     - `ecological_collapse_zone`: No sustainable extraction possible
   - Feasibility rate: How often is total need ≤ safe yield?

9. **governance_credibility_metrics()** (lines 408-553)
   Measures whether governance language translated into action:
   - Pledge strength: How ambitious are pledges relative to sustainable share?
   - Pledge compliance rate: How often are pledges honored?
   - Late governance index: Did governance appear too late to help?
   - Empty governance score: Cooperative talk but no sustainability
   - Failure mode classification:
     - `sustained`: Survived
     - `deception_failure`: Broken pledges caused collapse
     - `weak_pledge_failure`: Pledges were too weak
     - `late_governance_failure`: Governance came too late
     - `empty_governance_failure`: Cooperative talk, no action
     - `coordination_or_threshold_failure`: Other failure
   - Counterfactual analysis: Would the pool have survived if pledges were binding?

10. **summarize_run()** (lines 556-652)
    Master function that combines all metrics:
    - Survival, rounds completed, final pool
    - Total harvested, Gini coefficient
    - Message cooperation scores
    - Request-to-sustainable-share ratios
    - Model fallback statistics
    - Pledge statistics (made, evaluated, broken, hypocrisy rate)
    - All governance credibility metrics
    - All demand accounting metrics
    - All policy adequacy metrics

---

### observations.py

**Purpose**: Generates agent-specific observations with realism effects.

#### Key Functions:

1. **_demand_multiplier()** (lines 13-24)
   Returns the demand multiplier for a given round:
   - If regime is "none": 0 (no private need)
   - If during crisis rounds: crisis_multiplier (higher need)
   - If after crisis: post_crisis_multiplier
   - Otherwise: base multiplier

2. **_need_visibility_block()** (lines 27-61)
   Generates the need visibility section of the prompt:
   - If "private": empty (no need info shown)
   - If "audited": shows verified private needs (from observations)
   - If "public": shows reported needs from last round (could be lies)
   - Different labels for benchmark vs naturalistic mode

3. **_observed_env_summary()** (lines 64-133)
   Generates each agent's view of the environment:
   
   **Pool Observation**:
   - True pool level
   - If report_delay_rounds > 0: use pool from N rounds ago
   - If observation_noise_pct > 0: add random noise (±10%)
   - Clamp to [0, max_capacity]
   
   **Private Need** (if demand_multiplier > 0):
   - Base need = initial_cap × demand_weight × demand_multiplier
   - If private_demand_shock_pct > 0: multiply by random shock (±35%)
   - Anchor to **initial** cap (not current pool) so need doesn't shrink with depletion
   
   Returns: (formatted_prompt_string, observation_dict)

---

### scripted_agents.py

**Purpose**: Simple rule-based agents for baseline comparison.

#### ScriptedAgent Class:

Implements the same interface as `LLMAgent` but with hardcoded strategies:

1. **always_max**: Always request the maximum cap
2. **sustainable_share**: Request ~3% of pool (conservative)
3. **tit_for_tat**: Mirror the average of opponents' last 2 requests
4. **greedy_until_threshold**: Max request if pool > 50%, else 1/4 cap
5. **pledge_then_comply**: Pledge half cap and comply

Each strategy returns an `Action` with appropriate values. Used to establish baselines and test whether LLM behavior is genuinely interesting or just a fancy threshold policy.

---

### cli.py

**Purpose**: Command-line interface for running simulations.

#### Arguments:
- `--experiment`: Which institutional condition (or "suite" for all)
- `--roster`: Which agent roster
- `--prompt-mode`: "benchmark" or "naturalistic"
- `--realism`: "perfect" or "field"
- `--demand-regime`: "none", "low", "medium", "high", "crisis"
- `--need-visibility`: "private", "public", "audited"
- `--seed`: Random seed for reproducibility
- `--no-sleep`: Skip per-round delay
- `--temperature`: LLM temperature (0.0 = deterministic)
- `--trials`: Number of repeated runs
- `--max-rounds`: Game length
- `--agent-timeout-seconds`: Per-agent decision timeout
- `--rotate-heterogeneous-models`: Rotate model assignments across trials
- `--log-level`: Logging verbosity
- `--log-file`: Optional log file path
- `--force-log-config`: Reset logging handlers

#### Main Flow:
1. Parse arguments
2. Configure logging
3. If experiment="suite": run all experiments via `run_suite()`
4. Otherwise: run single experiment for N trials

---

### ui.py

**Purpose**: Rich terminal dashboard for live visualization.

#### Key Functions:

1. **_pool_color()** (lines 22-26)
   Maps pool health to color gradient:
   - Red (#ff4444) at 0%
   - Orange (#ff8800) at 25%
   - Yellow (#ffcc00) at 50%
   - Light green (#88cc00) at 75%
   - Green (#00cc66) at 100%

2. **_pool_bar()** (lines 29-35)
   ASCII bar chart: `████████░░░░ 50/100`

3. **render_round()** (lines 38-137)
   Displays each round:
   - Header with round number
   - Pool panel (before/after, replenishment, bar chart)
   - Agent scoreboard table (requested, granted, inventory, messages)
   - Reasoning panels (one per agent, color-coded)

4. **render_finale()** (lines 139-183)
   End-of-game summary:
   - Title: "TRAGEDY OF THE COMMONS" (if depleted) or "SIMULATION COMPLETE" (if survived)
   - Final standings table with medals
   - Average per-round harvest

5. **render_intro()** (lines 186-212)
   Pre-game banner showing:
   - Agent names
   - Starting pool
   - Number of rounds
   - Experiment, roster, prompt mode, realism

---

### judge_messages.py

**Purpose**: Post-hoc LLM coding of public messages (separate from agent decisions).

#### Key Components:

1. **SYSTEM_PROMPT** (lines 33-70)
   Instructions for the judge model:
   - Code messages on 9 dimensions (0=absent, 1=present, 2=strong)
   - Labels: cooperative_rhetoric, extractive_rhetoric, quota_proposal, pledge, threat_or_sanction, blame, moral_appeal, urgency_or_panic, deception_suspicion
   - Return JSON with coded_messages

2. **judge_file()** (lines 113-162)
   Main function:
   - Load result JSON
   - For each turn, extract messages
   - Call judge model (Groq)
   - Append results to data
   - Save augmented JSON

Used for validation: compare judge-coded labels against regex-based labels from `metrics.py`.

---

### analyze_results.py

**Purpose**: Aggregate multiple simulation runs into a researcher-friendly CSV.

#### Key Functions:

1. **_load_runs()** (lines 68-105)
   Loads all JSON results from a directory:
   - Filters by max_provider_fallback_rate (exclude unreliable runs)
   - Recomputes metrics if missing
   - Returns list of run data

2. **aggregate_runs()** (lines 112-258)
   Groups runs by experimental condition:
   - (experiment, roster, prompt_mode, realism, demand_regime, need_visibility)
   - For each group, computes means and standard deviations
   - Returns list of summary rows

3. **write_csv()** (lines 261-266)
   Writes aggregated data to CSV with 45 fields:
   - Survival rate, final pool, total harvested
   - Gini coefficient, hypocrisy rate
   - Pledge metrics, governance metrics
   - Demand accounting metrics
   - Policy adequacy metrics
   - Failure modes

4. **print_table()** (lines 269-282)
   Console output of aggregated results (compact format)

---

### report_run.py

**Purpose**: Generate human-readable Markdown report and SVG plot for a single run.

#### Key Functions:

1. **write_svg_plot()** (lines 63-138)
   Creates an SVG line chart showing:
   - Pool level over time (teal)
   - Total requested (red)
   - Total effective (blue)
   - Total pledged next (purple)
   - Y-axis: 0 to max(max_capacity, max_request)
   - X-axis: round numbers
   - Legend at bottom

2. **write_markdown_report()** (lines 149-280)
   Generates a comprehensive report:
   - Header with run parameters
   - Embedded SVG plot
   - Key metrics table
   - Demand audit (per-agent need vs request vs grant)
   - Policy adequacy (per-round classification)
   - Per-round table (pool, need, requests, grants)
   - Final messages
   - Interpretation based on failure mode

3. **_interpretation()** (lines 283-300)
   Generates narrative based on failure mode:
   - Weak pledge failure → commitments too weak
   - Deception failure → broken pledges
   - Late governance → appeared too late
   - Empty governance → cooperative talk, no action
   - Sustained → survived

---

### logging_utils.py

**Purpose**: Configurable logging setup.

#### configure_logging() (lines 11-35)
- Sets logging level
- Adds stream handler (console)
- Optionally adds file handler
- Format: `timestamp level=LEVEL logger=NAME message=MSG`
- `force` parameter resets existing handlers

---

## Data Flow

### Single Round Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. OBSERVATION GENERATION                                        │
│                                                                  │
│ For each agent:                                                  │
│   - Get true pool level                                          │
│   - Apply delay (if realism.report_delay_rounds > 0)            │
│   - Add noise (if realism.observation_noise_pct > 0)            │
│   - Calculate private need (if demand_multiplier > 0)           │
│   - Add demand shock (if realism.private_demand_shock_pct > 0)  │
│   - Build env_summary string                                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. AGENT QUERY                                                   │
│                                                                  │
│ For each agent (parallel):                                       │
│   - Build system prompt (includes institution_rules)            │
│   - Build user prompt (env_summary + standings + messages +     │
│                        history + need_visibility)                │
│   - Call LLM (with retries and fallbacks)                       │
│   - Parse JSON response → Action                                │
│   - Normalize/clamp action values                               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. INSTITUTIONAL PROCESSING                                      │
│                                                                  │
│ - Compute binding quota (if enabled)                            │
│ - Apply request rules (quota caps, moratorium)                  │
│ - Calculate penalties (broken pledges + sanctions)              │
│ - Calculate sustainable shares                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. ENVIRONMENT PROCESSING                                        │
│                                                                  │
│ - Clamp requests to per-agent cap                               │
│ - If total demand > pool: proportional allocation               │
│ - Subtract extraction from pool                                 │
│ - Calculate logistic growth                                     │
│ - Add growth to pool (cap at max_capacity)                      │
│ - Record TurnResult                                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. AGENT STATE UPDATE                                            │
│                                                                  │
│ For each agent:                                                  │
│   - Add granted to inventory                                    │
│   - Subtract penalties (subsistence + reputation + pledge)      │
│   - Update reputation (if pledge violated)                      │
│   - Append to history (sliding window: keep last 5)             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 6. EXPORT & RENDER                                               │
│                                                                  │
│ - Export turn data to JSONL                                     │
│ - Render UI (pool bar, scoreboard, reasoning panels)            │
│ - Update state (pledges_due, binding_quota, moratorium)         │
└──────────────────────────────────────────────────────────────────┘
```

### Full Simulation Flow

```
START
  │
  ├─► Load configuration (experiment, roster, realism, demand_regime)
  │
  ├─► Initialize LLM clients (Gemini, Groq)
  │
  ├─► Create environment (CommonsEnv)
  │
  ├─► Create agents (LLMAgent or ScriptedAgent)
  │
  ├─► Build per-agent RNGs
  │
  ├─► Render intro UI
  │
  ├─► FOR round 1 to max_rounds:
  │     │
  │     ├─► Check moratorium trigger
  │     │
  │     ├─► Generate observations for each agent
  │     │
  │     ├─► Query agents for decisions (parallel)
  │     │
  │     ├─► Apply institutional rules
  │     │
  │     ├─► Process environment turn
  │     │
  │     ├─► Calculate penalties
  │     │
  │     ├─► Update agent state
  │     │
  │     ├─► Export turn data
  │     │
  │     ├─► Render UI
  │     │
  │     ├─► Check depletion → BREAK if pool ≤ 0
  │     │
  │     └─► Sleep (if render=True)
  │
  ├─► Render finale UI
  │
  ├─► Compute metrics (summarize_run)
  │
  ├─► Export full JSON
  │
  └─► END
```

---

## Key Design Patterns

### 1. **Sliding Window Memory**
Agents remember their own last 5 rounds in detail but only see current-round aggregate information about others. This creates realistic information asymmetry.

### 2. **Structured Output with Pydantic**
All agent decisions go through a validated `Action` schema. This ensures type safety and makes parsing reliable even with weaker LLMs.

### 3. **Fallback Chain**
Primary model → retry → fallback models → conservative default. This ensures the simulation never crashes due to API failures.

### 4. **Counterfactual Replay**
The metrics system can replay "what if" scenarios: what if agents had followed their pledges? This measures the rhetoric-action gap.

### 5. **Per-Agent RNG**
Each agent gets its own `random.Random` instance, seeded deterministically from the master seed. This ensures reproducibility while allowing parallel queries.

### 6. **JSONL Checkpointing**
Each round is appended to a JSONL file. If the simulation crashes, you can recover all completed rounds.

### 7. **Alias-Based Anonymity**
Agents are known as "Organization A", "Organization B", etc. in public communications. This reduces name-based biases while preserving internal logging.

---

## Configuration and Experiments

### Running a Single Experiment

```bash
python3 -m commons_sim.cli \
  --experiment binding_quotas \
  --roster single_model_llama \
  --prompt-mode naturalistic \
  --realism field \
  --demand-regime medium \
  --need-visibility private \
  --seed 42 \
  --trials 30 \
  --no-sleep
```

### Running the Full Suite

```bash
python3 -m commons_sim.cli \
  --experiment suite \
  --roster heterogeneous \
  --prompt-mode benchmark \
  --trials 30 \
  --no-sleep
```

This runs all 11 institutional conditions × 30 trials = 330 simulations.

### Analyzing Results

```bash
python3 -m commons_sim.analyze_results.py \
  --results-dir results \
  --out results/summary.csv \
  --max-provider-fallback-rate 0.1
```

This aggregates all JSON results into a CSV, excluding runs with >10% provider fallback rate.

### Generating Reports

```bash
python3 -m commons_sim.report_run.py results/latest_run.json --out-dir reports
```

This generates a Markdown report and SVG plot for a single run.

---

## Summary

This framework is a comprehensive system for studying LLM agent behavior in a commons governance scenario. It combines:

- **Game theory**: Tragedy of the Commons with logistic resource dynamics
- **Multi-agent systems**: Heterogeneous LLM agents with memory and communication
- **Institutional economics**: Various governance mechanisms (communication, pledges, quotas, sanctions, contracts)
- **Experimental design**: Controlled comparisons across prompt modes, realism levels, demand regimes
- **Rich metrics**: Survival, inequality, governance credibility, policy adequacy, demand accounting
- **Robust engineering**: Fallbacks, checkpointing, reproducibility, comprehensive testing

The codebase is well-structured, thoroughly documented, and suitable for academic research on LLM multi-agent systems.