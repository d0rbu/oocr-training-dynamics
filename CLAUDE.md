# CLAUDE.md — OOCR training dynamics

Follow [AGENTS.md](AGENTS.md). It is the source of truth for agent behavior in this repository.

The most important project-specific rules are:

- no GPU/model-weight work before explicit user release;
- no provisional Gemma substitution without confirmation;
- no silent overwrite of artifacts or completed runs;
- no presentation of synthetic visualization data as measured evidence;
- no raw cross-family activation patching;
- keep the preregistered seed, conditions, checkpoint schedule, and intended/planted outcomes
  fixed unless a dated follow-up explicitly changes the contract.

Before handoff, run the CPU-only validation in [AGENTS.md](AGENTS.md) and state clearly that it
does not validate live model execution.
