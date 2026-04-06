# Tragedy of the Commons

Multi-agent commons-governance simulation with LLM-backed agents, scripted baselines, institutional interventions, and structured experiment exports.

## Overview

This repository implements a repeated shared-resource simulation in which multiple agents extract from a regenerating common pool under different governance conditions. Agents can communicate publicly, maintain limited memory, and operate under institutional rules such as ledgers, pledges, quotas, sanctions, contracts, and moratoria.

The codebase supports:

- LLM-backed agents
- scripted baseline agents
- single-model and heterogeneous rosters
- local Ollama inference
- structured JSON exports for downstream analysis

## Installation

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Running

Run a single experiment:

```bash
python3 cli.py --experiment public_ledger --roster scripted_baselines
```

Run the full institutional suite:

```bash
python3 cli.py --experiment suite --roster scripted_baselines
```

Run a local Ollama-backed single-model roster:

```bash
OLLAMA_MODEL='gemma3:4b' python3 cli.py --experiment public_ledger --roster single_model_ollama --max-rounds 5 --temperature 0.0
```

Run a hosted-model configuration with throttling:

```bash
python3 cli.py --experiment public_ledger --roster single_model_qwen --agent-call-delay-seconds 1.5
```

## CLI Options

Commonly used options:

- `--experiment`: institutional condition or `suite`
- `--roster`: model ecology / agent roster
- `--prompt-mode`: `benchmark` or `naturalistic`
- `--realism`: realism profile
- `--demand-regime`: private demand regime
- `--need-visibility`: `private`, `public`, or `audited`
- `--max-rounds`: number of rounds
- `--trials`: repeated runs
- `--temperature`: model temperature
- `--agent-call-delay-seconds`: delay between sequential agent calls
- `--parallel-agent-calls`: opt back into concurrent agent calls

## Local Ollama Configuration

Optional environment variables:

```bash
export OLLAMA_HOST='http://127.0.0.1:11434'
export OLLAMA_MODEL='qwen2.5:7b'
export OLLAMA_FLASH_ATTENTION='1'
export OLLAMA_KV_CACHE_TYPE='q8_0'
```

## Testing

Run the test suite:

```bash
PYTHONPATH=. pytest -q
```

## Results

Simulation outputs are written to `results/` as JSON and JSONL artifacts. Aggregated summaries can be generated with:

```bash
python3 analyze_results.py --results-dir results --out results/summary.csv
```

Aggregate only clean runs:

```bash
python3 analyze_results.py --results-dir results --out results/summary_clean.csv --clean-only
```

Each exported run includes:

- experiment configuration
- per-turn actions, grants, and messages
- aggregate metrics
- protocol metadata
- runtime metadata
- cleanliness metadata (`clean_run`, `fallback_events`, `exclusion_reasons`)

## Repository Structure

- `cli.py`: command-line entrypoint
- `simulation_core.py`: simulation orchestration
- `agent.py`: LLM agent implementation
- `scripted_agents.py`: scripted baselines
- `environment.py`: commons environment dynamics
- `institutions.py`: institutional rules and policy state
- `observations.py`: observation and demand pipelines
- `metrics.py`: run-level metrics
- `analyze_results.py`: aggregation over result files
- `judge_messages.py`: optional post-hoc message coding
- `tests/`: unit tests

## Notes

- Generated artifacts in `results/` and `reports/` are ignored by Git.
- Hosted APIs may require throttling for large multi-agent runs.
- Local or dedicated GPU inference is more practical for large experimental sweeps.
