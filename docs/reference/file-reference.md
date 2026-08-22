# File reference

## Top level

| Path | Purpose |
|---|---|
| `README.md` | Project status, experiment summary, safe quickstart |
| `AGENTS.md` | Operational and contribution rules |
| `CLAUDE.md` | Pointer to the shared agent contract |
| `THIRD_PARTY_NOTICES.md` | Functions-task provenance and license notice |
| `pyproject.toml` / `uv.lock` | Package/tool configuration and locked environment |
| `.gpu-runs-enabled` | Ignored, user-authorized CUDA sentinel; intentionally absent by default |

## Package

| Path | Purpose |
|---|---|
| `oocr_training_dynamics/contracts.py` | Experiment enums, specs, schedules, run IDs |
| `oocr_training_dynamics/analysis.py` | Preregistered clustered intervals and curve summaries |
| `oocr_training_dynamics/models.py` | Model registry and parameter/storage calculations |
| `oocr_training_dynamics/data.py` | Function suite and matched data generation |
| `oocr_training_dynamics/tokenization.py` | Target boundaries and collation |
| `oocr_training_dynamics/semantics.py` | Safe generated-lambda scorer |
| `oocr_training_dynamics/metrics.py` | Curve and patch metrics |
| `oocr_training_dynamics/patching.py` | Pure patch plans and dirty prompt construction |
| `oocr_training_dynamics/weight_alignment.py` | Symmetric effective-weight metric, axis, and artifact contracts |
| `oocr_training_dynamics/artifacts.py` | Atomic JSON, hashes, checkpoint paths |
| `oocr_training_dynamics/planning.py` | Run/storage plan |
| `oocr_training_dynamics/gpu_guard.py` | Double authorization gate |
| `oocr_training_dynamics/runtime_*.py` | Gated model, training, evaluation, and patch execution |
| `oocr_training_dynamics/runtime_weight_alignment.py` | Gated effective-projection comparison runner |
| `oocr_training_dynamics/answer_lookup.py` | Pure answer-location sites and 27-row intervention registry |
| `oocr_training_dynamics/runtime_answer_lookup.py` | Gated batch-one line-terminator patching runtime |
| `oocr_training_dynamics/switched_answer_minsets.py` | Pure composite layer-swap and strict minset contracts |
| `oocr_training_dynamics/runtime_switched_answer_minsets.py` | Step-1500 into step-0 paired-answer swap collection and exact search |
| `oocr_training_dynamics/fourier_hardware_lineage.py` | Immutable cross-device reference, adapter, source, and hardware identity contract |
| `oocr_training_dynamics/fourier_{circuits,recall,frontier,disconnected}.py` | Pure Fourier/minset proposal and causal-verification contracts |
| `oocr_training_dynamics/runtime_fourier_*.py` | Full-prompt, batch-one, resumable Fourier/minset collection runtimes |

## Scripts

| Path | Purpose |
|---|---|
| `scripts/plan_experiments.py` | Write the CPU-only preregistered plan |
| `scripts/validate_tokenizers.py` | Probe processors/chat templates without weights |
| `scripts/run_training.py` | Train, capacity-pause, or resume one model/condition |
| `scripts/run_evaluation.py` | Evaluate every indexed checkpoint for one run |
| `scripts/run_patching.py` | Produce one across-sample/across-time patch plan |
| `scripts/run_patching_matrix.py` | Resume/skip through selected or full patching coverage |
| `scripts/run_weight_alignment.py` | Resume symmetric full-effective-weight comparisons across unordered checkpoint pairs |
| `scripts/run_activation_examples.py` | Measure checkpoint-indexed top cosine-matching prompt tokens for selectable activation cells |
| `scripts/run_answer_lookup.py` | Plan or resume the step-1500 answer-location patching atlas |
| `scripts/run_switched_answer_minsets.py` | Plan or resume cross-checkpoint paired-answer minset searches |
| `scripts/export_switched_answer_minset_site.py` | Refresh only the paired-answer minset manifest during collection |
| `scripts/export_answer_lookup_site.py` | Refresh answer-location chunks/manifests without rebuilding unrelated atlases |
| `scripts/export_fourier_site.py` | Refresh Fourier chunks, including an explicit external science-root export |
| `scripts/engaging/export_fourier_lineage_site.sh` | CPU-only, digest-gated projection of a frozen cluster lineage |
| `scripts/engaging/import_fourier_lineage_site.sh` | Atomic compact-chunk sync and local manifest merge for a cluster lineage |
| `scripts/engaging/run_h200_batch1_checkpoint_stage0.sh` | One-device batch-one grid, lineage registration, and Stage-0 harness |
| `scripts/engaging/run_h200_batch1_checkpoint_pair.sh` | Explicitly pin two independent Stage-0 tasks to a two-H200 allocation |
| `scripts/engaging/run_h200_full_recall.sh` | Digest-gated, shard-resumable full recall ladder for one registered H200 target |
| `scripts/analyze_results.py` | Compute frozen-adjusted AUCs and function-clustered intervals |
| `scripts/export_site.py` | Discover artifacts and rebuild the static site payload |

## Website and tests

| Path | Purpose |
|---|---|
| `site/index.html` | Semantic static page structure |
| `site/styles.css` | Responsive editorial visualization design |
| `site/app.js` | Interactive curves, checkpoint sliders, and patch heatmap |
| `site/data/experiment.json` | Committed preview or measured compact payload |
| `tests/` | Contract, corpus, metric, artifact, tokenization, and site regressions |

## Documentation

Use [docs/README.md](../README.md) as the index. Research claims live in dated experiment reports;
operational facts live in `docs/operations/`; frozen hypotheses live in
`docs/research/preregistration.md`.
