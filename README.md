# Tragedy of the Commons

A multi-agent simulation framework for studying how LLM-backed organizations behave when exploiting a shared, regenerating resource.

This project models a repeated commons dilemma with stakeholder agents, public messaging, institutional interventions, and researcher-friendly run exports. It can be used as:

- a sandbox for multi-agent LLM behavior
- a governance-mechanism simulator
- a reproducible experiment harness for comparing communication and policy regimes

## Why This Project Is Interesting

Most LLM demos stop at a single agent making a single decision. This project looks at something messier and more realistic:

- multiple agents with conflicting incentives
- repeated interaction over many rounds
- shared-resource depletion and recovery
- public rhetoric that can diverge from actual behavior
- institutional mechanisms such as ledgers, pledges, quotas, sanctions, contracts, and moratoria

It is designed to make those interactions measurable. Runs export structured JSON with metrics for survival, extraction, inequality, need satisfaction, and governance credibility.

## Features

- LLM-backed and scripted agents
- heterogeneous stakeholder personas
- single-model and mixed-model rosters
- local Ollama support
- public communication between agents
- institutional conditions:
  - no communication
  - cheap talk
  - public ledger
  - nonbinding pledges
  - binding quotas
  - sanctions
  - negotiated contracts
  - moratoria
- result export with protocol/runtime metadata
- contamination tracking for fallback-affected runs
- aggregation script for summarizing multiple runs

## Quick Start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run a simple offline baseline:

```bash
python3 cli.py --experiment public_ledger --roster scripted_baselines --max-rounds 5 --no-sleep
```

Run the test suite:

```bash
PYTHONPATH=. pytest -q
```

## Running With Local Ollama

The repo supports a local single-model Ollama roster.

Example:

```bash
OLLAMA_MODEL='gemma3:4b' python3 cli.py --experiment public_ledger --roster single_model_ollama --max-rounds 5 --temperature 0.0
```

Optional environment variables:

```bash
export OLLAMA_HOST='http://127.0.0.1:11434'
export OLLAMA_MODEL='qwen2.5:7b'
export OLLAMA_FLASH_ATTENTION='1'
export OLLAMA_KV_CACHE_TYPE='q8_0'
```

## Running With Hosted APIs

Hosted APIs are supported, but repeated multi-agent runs can hit rate limits. The code reduces pressure by:

- querying agents sequentially by default
- supporting an inter-agent delay
- using a compact prompt budget

Example:

```bash
python3 cli.py --experiment public_ledger --roster single_model_qwen --agent-call-delay-seconds 1.5
```

## Example Experiments

Run one institutional condition:

```bash
python3 cli.py --experiment public_ledger --roster scripted_baselines
```

Run the whole institutional suite:

```bash
python3 cli.py --experiment suite --roster scripted_baselines
```

Run a local single-model LLM society:

```bash
OLLAMA_MODEL='gemma3:4b' python3 cli.py --experiment public_ledger --roster single_model_ollama --prompt-mode benchmark --realism perfect --demand-regime medium --need-visibility private --max-rounds 10
```

## Result Format

Each exported JSON includes:

- experiment configuration
- per-turn requests, grants, messages, and model status
- aggregate metrics
- protocol metadata
- runtime metadata
- cleanliness metadata:
  - `clean_run`
  - `fallback_events`
  - `exclusion_reasons`

This makes it easier to separate exploratory runs from dataset-quality runs.

## Aggregation

Aggregate runs into a CSV:

```bash
python3 analyze_results.py --results-dir results --out results/summary.csv
```

Aggregate only clean runs:

```bash
python3 analyze_results.py --results-dir results --out results/summary_clean.csv --clean-only
```

## Project Structure

- [cli.py](/Users/karan/Desktop/Projects/multiagent/commons_sim/cli.py): command-line entrypoint
- [simulation_core.py](/Users/karan/Desktop/Projects/multiagent/commons_sim/simulation_core.py): orchestration loop
- [agent.py](/Users/karan/Desktop/Projects/multiagent/commons_sim/agent.py): LLM agent logic
- [scripted_agents.py](/Users/karan/Desktop/Projects/multiagent/commons_sim/scripted_agents.py): non-LLM baselines
- [environment.py](/Users/karan/Desktop/Projects/multiagent/commons_sim/environment.py): shared-resource dynamics
- [institutions.py](/Users/karan/Desktop/Projects/multiagent/commons_sim/institutions.py): governance rules
- [metrics.py](/Users/karan/Desktop/Projects/multiagent/commons_sim/metrics.py): evaluation metrics
- [analyze_results.py](/Users/karan/Desktop/Projects/multiagent/commons_sim/analyze_results.py): multi-run aggregation
- [tests/](/Users/karan/Desktop/Projects/multiagent/commons_sim/tests): unit tests

## Resume-Friendly Summary

If you want to mention this project on a resume, a strong one-line version is:

Built a multi-agent LLM simulation of commons governance with institutional interventions, reproducible experiment exports, and behavioral metrics for sustainability, inequality, and rhetoric-action divergence.

Shorter version:

Built a multi-agent LLM commons-governance simulator with quotas, sanctions, contracts, and reproducible evaluation tooling.

## Notes

- Generated outputs are written to `results/` and `reports/`
- These directories are ignored by Git
- For large experimental sweeps, local or dedicated GPU inference is more practical than shared token-metered APIs
