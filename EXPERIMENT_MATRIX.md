# Experiment Matrix

This project should be framed as a governance credibility audit, not only as a
commons-collapse demo.

## Core Question

When LLM-based organizations govern a shared aquifer, do public statements and
pledges create real sustainable restraint, or do they create the appearance of
governance while extraction remains too high?

## Primary Factors

- Prompt mode: `benchmark`, `naturalistic`
- Realism profile: `perfect`, `field`
- Institution: `no_communication`, `cheap_talk`, `sanctions`, `early_binding_quota`,
  `delayed_binding_quota`, `adaptive_quota`, `mandatory_moratorium`, `contracts`
- Roster: `heterogeneous`, `single_model_llama`, `single_model_gpt_oss`,
  `scripted_baselines`, `scripted_stewards`
- Suggested trials per cell: 30-50

## Primary Metrics

- Survival rate
- Collapse round
- Total harvested
- Inventory Gini
- Mean request-to-sustainable-share
- Mean effective-request-to-sustainable-share
- Pledge strength
- Pledge compliance rate
- Empty governance score
- Pledge-policy survival from round 1
- Latest saving round under pledge policy
- Failure mode

## Hypotheses

- H1: Communication increases cooperative rhetoric more than it reduces extraction.
- H2: Sanctions reduce deception failures but not weak-pledge failures.
- H3: Early and adaptive quotas outperform delayed quotas.
- H4: Naturalistic framing changes extraction behavior relative to benchmark framing.
- H5: LLM societies show higher rhetoric-action divergence than scripted steward baselines.
- H6: Field realism (noisy/delayed observations plus private demand shocks) increases
  weak-pledge and late-governance failures relative to perfect information.

## Exclusion / Sensitivity Rule

Primary analyses should exclude runs where provider fallback rate exceeds 10%.
Report excluded runs separately as a model reliability sensitivity check.

Example:

```bash
python3 analyze_results.py --results-dir results --out results/summary_filtered.csv --max-provider-fallback-rate 0.1
```

## Optional LLM Judge

Use `judge_messages.py` only as a post-hoc coding instrument. The judge sees
public messages after a run is complete and must not affect agent decisions.
Report the judge provider, model, and rubric version if using these labels in a
paper.

Choose a Groq judge model that is not used by the agents. The default is read
from `GROQ_JUDGE_MODEL`, falling back to `gemma2-9b-it`.

```bash
python3 judge_messages.py results/<run>.json --model gemma2-9b-it
```

For primary claims, compare judge-coded labels against the transparent regex
labels and report any sensitivity.

## Recommended Batch

```bash
python3 -m simulation --experiment suite --roster heterogeneous --prompt-mode benchmark --trials 30 --no-sleep
python3 -m simulation --experiment suite --roster heterogeneous --prompt-mode naturalistic --trials 30 --no-sleep
python3 -m simulation --experiment suite --roster heterogeneous --prompt-mode naturalistic --realism field --trials 30 --seed 1000 --no-sleep
python3 -m simulation --experiment suite --roster scripted_baselines --prompt-mode benchmark --trials 30 --no-sleep
python3 -m simulation --experiment suite --roster scripted_stewards --prompt-mode naturalistic --trials 30 --no-sleep
python3 analyze_results.py --results-dir results --out results/summary.csv --max-provider-fallback-rate 0.1
```
