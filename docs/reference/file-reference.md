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
| `oocr_training_dynamics/artifacts.py` | Atomic JSON, hashes, checkpoint paths |
| `oocr_training_dynamics/planning.py` | Run/storage plan |
| `oocr_training_dynamics/gpu_guard.py` | Double authorization gate |
| `oocr_training_dynamics/runtime_*.py` | Gated model, training, evaluation, and patch execution |

## Scripts

| Path | Purpose |
|---|---|
| `scripts/plan_experiments.py` | Write the CPU-only preregistered plan |
| `scripts/validate_tokenizers.py` | Probe processors/chat templates without weights |
| `scripts/run_training.py` | Train, capacity-pause, or resume one model/condition |
| `scripts/run_evaluation.py` | Evaluate every indexed checkpoint for one run |
| `scripts/run_patching.py` | Produce one across-sample/across-time patch plan |
| `scripts/run_patching_matrix.py` | Resume/skip through selected or full patching coverage |
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
