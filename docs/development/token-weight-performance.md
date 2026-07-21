# Token-local weight-patching performance contract

## Immutable reference

The optimization target is the OLMo 3 7B token-local checkpoint-transfer path introduced at
commit `956bfa4089529e38aa496f8039e4fafcad26f150`. Its reference kernel uses patch batch size 8 and
asks the causal-LM head for logits at every prompt position before selecting the final position.
The intervention itself applies all seven donor LoRA contributions at one selected token and layer.

The first complete endpoint artifact is donor step 0 into recipient step 1500:

- raw artifact SHA-256: `bcae3e78904e14bee93c0518bdb2312fd9d29c558f815e06b79f6b4c16790848`;
- compact committed artifact SHA-256:
  `a07e7e6f1fd813f4fafb3e46a9c53381b98fda5bcd4ee7c75b4ac9eb49956ef0`;
- 19 functions, 32 layers, 111–123 prompt positions, and 71,456 probabilities;
- measured wall time from model-load start through raw artifact write: 1,210 seconds on the
  RTX 4090, with patch batch size 8.

The raw artifact remains ignored experiment state. The compact artifact is versioned under
`site/data/patches/` and is the portable output oracle.

## Candidate boundary

The initial optimized kernel changes only the causal-LM readout request from all sequence logits
to `logits_to_keep=1`, using the model family's native Transformers API. It does not change:

- the model, checkpoint, prompt, token axis, candidate IDs, or softmax;
- the order or arithmetic of any decoder block;
- the seven token-local LoRA output corrections;
- the selected-token or layer traversal order;
- reference patch batch size 8 unless a separately benchmarked candidate batch size is selected.

The production CLI retains `--token-weight-runtime reference`. An optimized runtime cannot become
the default merely because it is mathematically equivalent or numerically close.

## Acceptance ladder

1. CPU tests prove that the optimized readout requests exactly one logit position and that the
   recursive parity checker rejects even a `1e-15` probability change or schema drift.
2. The live reference kernel must reproduce the stored endpoint artifact exactly before any
   candidate comparison is trusted.
3. Candidate batch sizes are benchmarked on the same model load, checkpoint pair, function set,
   token axis, and GPU. Every nested serialized value must compare equal with Python's exact
   equality; tolerance-based acceptance is prohibited.
4. A winning candidate must reproduce all 19 endpoint records exactly, not only one function or a
   subset of cells.
5. Report wall time, speedup, peak allocated VRAM, GPU model, software revisions, and any OOM or
   parity failures. A nonexact candidate remains available only as failed benchmark evidence.
6. The unchanged reference runtime remains a rollback path and receives a regression test.

Run the single-function candidate sweep only at a safe experiment boundary:

```bash
uv run python scripts/benchmark_token_weight_runtime.py \
  --artifact-root /home/d0rb/Documents/Github/oocr-training-dynamics \
  --recipient-step 1500 --donor-step 0 --function-id identity \
  --candidate-batch-size 8 --candidate-batch-size 16 \
  --candidate-batch-size 32 --candidate-batch-size 64 \
  --output /home/d0rb/Documents/Github/oocr-training-dynamics/artifacts/benchmarks/token-weight-runtime.json \
  --confirm-gpu-run
```

Only after selecting an exact candidate should the same command be rerun with `--all-functions`
and that one batch size. Benchmark artifacts are ignored operational evidence; summarized results
belong in this document and the pull request.
