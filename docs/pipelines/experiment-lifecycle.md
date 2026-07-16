# Experiment lifecycle

## Stage 0 — preregistered, no CUDA

1. Freeze models/revisions, seed, conditions, schedule, endpoints, and causal predictions.
2. Generate the CPU plan and storage estimate.
3. Validate deterministic corpora, chat templates, tests, and the synthetic site.
4. Record that no GPU experiment has run.

This is the repository's current stage as of 2026-07-15.

## Stage 1 — capacity gate

After explicit authorization, run one effective batch and pause at step 1 for each confirmed
family. Record actual peak VRAM and disk use. If a physical batch OOMs, retry a smaller divisor of
64 while preserving the shared target-token denominator.

## Stage 2 — behavioral training

For one model at a time:

1. continue the correct run through step 1500;
2. evaluate all 18 checkpoints;
3. confirm that a behavioral OOCR effect exists before expensive causal interpretation;
4. run wrong-alias and wrong-implementation conditions with the same order and schedule;
5. evaluate intended and planted targets at every checkpoint.

Do not treat lower training loss as a monotone proxy for rule recovery; the earlier replication's
free-form endpoint peaked before the end of the epoch.

## Stage 3 — causal patching

Only models that pass the behavioral gate receive the complete correct-condition patching matrix.
Run across-sample patches over recipient time, then across-time patches with base and earlier
donors. Controls may receive a smaller confirmatory patch set, but they are not required for the
primary causal claim.

## Stage 4 — analysis and visualization

1. compute paired, function-clustered uncertainty and log-example curve AUCs;
2. export measured curves and patch JSON into the site payload;
3. inspect every selected view's measured/unprocessed badge;
4. write a dated results report with nulls and failure modes;
5. keep model-family conclusions separate before attempting a cross-model synthesis.

The per-run behavioral analysis command is:

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/analyze_results.py \
  --model olmo3-7b --condition correct
```

## Stage 5 — retention

Retain configs, manifests, metrics, adapter checkpoints, evaluation outputs, patch grids, and the
dated report. Retain only one rolling optimizer/RNG state per active run. Base weights remain in
the external Hugging Face cache and may be evicted model-by-model only after verifying that no
other process owns them.
