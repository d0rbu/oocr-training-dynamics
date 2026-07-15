# Correctness contract

This project makes experiment state explicit and fails before producing an ambiguous artifact.

## Frozen identities

- model identifiers include full 40-character revisions;
- a run key contains model, condition, and seed;
- the condition enum prevents ad hoc control names;
- checkpoint steps are strictly increasing, include frozen step 0, and end at step 1500;
- the function derangement is a bijection with no fixed points and preserves output type and
  augmentation compatibility.

Changing one of these is a new experiment contract, not a cosmetic refactor.

## Exact loss aggregation

For one effective batch, let `T` be the number of assistant target tokens over all 64 records.
Each microbatch computes a *sum* of token cross-entropies, divides by the shared `T`, and calls
backward. The accumulated gradient is therefore the gradient of the 64-record token-mean loss up
to floating-point reduction order. It is not an average of microbatch means.

Global norm clipping occurs once after all microbatches are accumulated. The pre-clip norm is
stored. Nonfinite loss or norm is fatal.

## Token boundaries

The model-family chat template is applied twice: once to the prompt with a generation marker and
once to the complete prompt/assistant response. The first tokenization must be an exact prefix of
the second. Only the additional assistant tokens receive labels; every prefix and padding token is
`-100`.

## Artifact integrity

- JSON writes use a temporary file followed by atomic replacement.
- Adapter weights use safetensors and receive a SHA-256 digest in the checkpoint index.
- A trained checkpoint cannot be indexed without an adapter path and digest.
- Step 0 cannot claim an adapter.
- Completed runs and unacknowledged partial runs are never overwritten.
- Resume requires a matching config, adapter, optimizer step, checkpoint index, metrics, and RNG
  state.

## Activation-patching integrity

The confirmatory primary interface remains fixed to `resid_post`. Exploratory branch interfaces
must resolve exactly one `self_attn` or `mlp` module per registered decoder layer and declare
whether the pre-hook input or post-hook output is replaced. Each interface writes to a distinct
artifact path. Token positions are reverse-indexed from the final model-tokenized prompt token;
across-name spans stop at both name endings, while same-prompt temporal spans continue to sequence
start.
Temporal donors must precede the recipient. Sample donors use the same checkpoint. A complete
patch row contains a finite correct-choice probability and raw recipient delta for every expected
layer and selected token position.

Raw cross-family patching is prohibited because hidden coordinates are not aligned merely because
two models share a residual width.

## Static and runtime checks

`beartype` protects public domain boundaries, `jaxtyping` expresses array shapes/dtypes, `ty`
checks static types, and property tests cover conservation/range invariants. CUDA runtime modules
are excluded from the numeric coverage threshold because importing them is cheap but valid model
execution is not a CPU unit test. They require dedicated post-authorization smoke tests described
in [testing.md](testing.md).
