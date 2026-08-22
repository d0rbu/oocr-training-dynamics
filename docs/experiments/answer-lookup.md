# Answer-location lookup experiment

**Registered:** 2026-08-17, before any GPU artifact for this experiment.

## Question

The late model may answer an A–E probe by letting the final prompt token attend to an earlier
option-line state that advertises “the correct answer is here.” This experiment intervenes on the
first autoregressive state that has consumed an entire answer option and its terminating line
break. It asks whether that state is a generic correctness marker, whether moving it redirects the
model to a different answer label, and whether duplicating it produces competition among several
advertised answers.

This is separate from the Fourier/minset analysis. It uses the correct-condition OLMo-3 7B
checkpoint at step 1500 and all 19 code-definition reflection probes.

## Exact site contract

For every rendered prompt, parse exactly one ordered A–E block. The site for a choice is the
unique tokenizer token whose character-offset span contains the **first newline terminating that
choice line**. This is a semantic line-ending site, not an assumption that the token string is
literally `\n`: OLMo's tokenizer sometimes merges the line ending with preceding text (for
example, `")\n"`) and commonly represents the E-option boundary as `"\n\n"`. Every artifact
stores the character index, token index, token ID, decoded token label, and token character span.

The recipient is always the exact clean prompt. The source and recipient models are the same
step-1500 model; only prompt and token position vary.

## Boundaries

The primary boundary is `attention_input`. At layer L, replacing the selected option token's
hidden state immediately before self-attention replaces that token's Q/K/V input. Its K/V can
therefore be read by later tokens, including the final answer position, in layer L itself. This is
the direct test of the lookup hypothesis.

`resid_post` is a propagation control. It replaces the selected token after the complete decoder
block, so its earliest possible effect on the final token is through attention in layer L+1.

Every scientific evaluation is batch one, full prompt, `use_cache=False`, under
`torch.inference_mode()`. The output is the entire A–E-normalized probability vector. The identity
intervention and a final unpatched rerun must each reproduce the clean baseline within `1e-6`.

## Intervention registry

There are 27 deterministic rows per function and boundary:

1. Four preservation controls transplant a correct-choice line ending into the clean correct
   location from (a) the identical clean prompt, (b) the same prompt with all answer contents
   deranged, (c) an unrelated non-coding MCQ with the same correct letter, and (d) an unrelated
   non-coding MCQ with a different correct letter.
2. Four erasures transplant each clean incorrect-choice line ending into the clean correct
   location.
3. Four moves simultaneously replace the correct location with one incorrect line state and the
   selected incorrect location with the correct line state. A move is therefore a two-site swap,
   not merely installation at a second location.
4. Fifteen duplications copy the clean correct line state to every nonempty subset of the four
   incorrect locations while leaving the original correct location untouched.

Across 32 layers, 19 functions, and two boundaries this is 32,832 patched forwards plus 152 source
capture forwards. Artifacts are row-resumable and written under
`artifacts/runs/olmo3-7b/correct/seed_20260715/answer_lookup/checkpoint_step_001500/`.

## Frozen directional predictions

- The identical-prompt control is an exact harness, not a scientific effect, and must be null.
- If the state is a generic “correct option here” marker, all three cross-prompt correct-line
  controls should have much smaller effects than incorrect-into-correct erasures.
- If the marker is location-causal, moving it should decrease the intended label and increase the
  selected wrong label. Reporting only P(intended) would miss this required redistribution, so
  all five probabilities are retained.
- If multiple markers compete, duplicating the correct state should distribute answer mass among
  advertised locations or increase A–E entropy. A null duplication result would instead support
  a content- or query-specific state that cannot be copied as a generic marker.
- An effect at `attention_input` followed one layer later by a corresponding `resid_post` pattern
  supports the proposed attention-mediated readout. A residual-only effect without the direct
  attention-input effect weakens that interpretation.
- Strong dependence on shuffled versus unrelated sources, or on same versus different source
  answer letters, means the state is not a context-invariant correctness marker.

The primary visual summary is the function-level distribution and the mean across all 19 fixed
functions. No row or layer is selected after inspecting outcomes. Missing cells remain explicitly
unprocessed; the website must never synthesize them.

## Run command

After a fresh explicit user statement that the GPU is free and creation of the ignored
`.gpu-runs-enabled` sentinel:

```bash
uv run python scripts/run_answer_lookup.py --confirm-gpu-run
```

Before that authorization, the CPU-only plan is:

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/run_answer_lookup.py --plan-only
```
