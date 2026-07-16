# Glossary

| Term | Meaning |
|---|---|
| OOCR | Out-of-context reasoning: recovering or using a latent rule in a context that does not restate the training demonstrations. |
| Functions task | A joint corpus in which opaque Python aliases are observed through input/output executions and later queried for their rule. |
| Intended target | The true function definition attached to the clean alias namespace. |
| Planted target | The definition implied by the corpus actually used for a control run. It equals the intended target only in the correct condition. |
| Wrong alias | A matched corpus where the I/O behavior is preserved but the alias used in prompts is permuted. |
| Wrong implementation | A matched corpus where the alias is preserved but outputs come from a permuted, type-compatible rule. |
| Frozen base | Step 0: the pinned pretrained/instruction-tuned checkpoint with no adapter. |
| Effective batch | The 64 records whose summed assistant-token loss shares one denominator and one optimizer step. |
| Microbatch | A memory-sized slice of the effective batch. Microbatches are not normalized independently. |
| `resid_post` | The residual-stream tensor emitted by a decoder block after its attention and MLP updates. |
| `attention_input` | Exact hidden-state argument passed into a decoder layer's self-attention module. |
| `attention_output` | Self-attention return after O projection and before later normalization or residual addition. |
| `mlp_input` | Exact hidden-state argument passed into a decoder layer's gated MLP module. |
| `mlp_output` | MLP return after down projection and before later normalization or residual addition. |
| Recipient | The prompt/model/checkpoint whose forward pass produces the patched answer. |
| Donor/source | The prompt/model/checkpoint supplying the replacement activation. |
| Across-sample patch | A different-name dirty state inserted into the clean prompt at the same checkpoint. |
| Across-time patch | A base/earlier-checkpoint state inserted into a later checkpoint on the same clean prompt. |
| Synthetic preview | Illustrative learning-curve values used only before measured behavioral artifacts exist; patch grids never use synthetic values. |
