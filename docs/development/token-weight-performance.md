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

The accepted candidate keeps the reference full-sequence logit computation. For a patch in layer
`L`, it captures the recipient input to every decoder layer once at the exact live batch shape,
then injects the cached input to `L` and skips blocks `0..L-1`. Those blocks are causally upstream
of the intervention, so their outputs are invariant across token coordinates at that layer. Layer
`L` and every downstream block still execute normally.

The candidate does not change:

- the model, checkpoint, prompt, token axis, candidate IDs, or softmax;
- the order or arithmetic of the patched layer or any downstream decoder block;
- the seven token-local LoRA output corrections;
- the selected-token or layer traversal order;
- reference patch batch size 8 unless a separately benchmarked candidate batch size is selected.

The production CLI retains `--token-weight-runtime reference` as the explicit rollback default.
The exact-validated optimized path is selected with `--token-weight-runtime optimized`; changing
the CLI default is a separate operational decision rather than an implicit consequence of a
mathematical argument.

## Acceptance ladder

1. CPU tests prove that the cached-prefix path bypasses only upstream blocks, restores every
   temporary forward override even after an exception, and produces the exact same synthetic
   hidden state. The recursive parity checker rejects even a `1e-15` probability change or schema
   drift.
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

## Component isolation result

The first authorized OLMo 3 7B identity-function run on the RTX 4090 isolated the two proposed
components at batch size 8:

- fresh reference: 61.45 seconds and an exact match to the stored endpoint record;
- native `logits_to_keep=1`: 61.52 seconds and **rejected** because the first probability mismatch
  was `0.9911057353019714` versus `0.9921426773071289`;
- decoder-prefix reuse with the original full-sequence logits: 33.24 seconds, **exact equality**
  across the complete identity record, 1.85x speedup, and 15.70 GB peak allocated VRAM versus
  15.43 GB for reference.

The final-token request changes the LM-head GEMM shape and is therefore not bitwise equivalent on
this GPU, despite selecting the same mathematical row. It is retained only as rejected benchmark
evidence and is not used by the optimized production runtime.

## Full acceptance result

The complete donor-step-0 into recipient-step-1500 comparison passed on 2026-07-21:

- 19 functions, 32 layers, 111–123 positions, and 71,456 measured probabilities;
- fresh reference: 1,214.16 seconds, 15.45 GB peak allocated VRAM, and exact recursive equality
  with the immutable stored artifact;
- decoder-prefix reuse: 670.82 seconds, 15.91 GB peak allocated VRAM, and exact recursive equality
  with the fresh reference across every serialized value;
- measured speedup: **1.810x**, with no tolerance and no first mismatch.

The machine-readable evidence is
[`token-weight-runtime-20260721.json`](token-weight-runtime-20260721.json). Its source full-report
SHA-256 is `30591c0c947b44a645732943617478001749eefbc7405744275d1c22341c6d49`.
The optimized runtime therefore passes the acceptance ladder while the unchanged reference remains
available as a fail-safe rollback.

Run the single-function candidate sweep only at a safe experiment boundary:

```bash
uv run python scripts/benchmark_token_weight_runtime.py \
  --artifact-root /home/d0rb/Documents/Github/oocr-training-dynamics \
  --recipient-step 1500 --donor-step 0 --function-id identity \
  --candidate-kernel decoder-prefix-reuse \
  --candidate-batch-size 8 --candidate-batch-size 16 \
  --candidate-batch-size 32 --candidate-batch-size 64 \
  --output /home/d0rb/Documents/Github/oocr-training-dynamics/artifacts/benchmarks/token-weight-runtime.json \
  --confirm-gpu-run
```

The accepted candidate was rerun with `--all-functions` and batch size 8. Raw benchmark artifacts
remain ignored operational evidence; the compact result above and its machine-readable companion
are the versioned record.
