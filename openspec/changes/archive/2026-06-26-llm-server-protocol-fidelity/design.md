## Context

tinygrad's LLM server (`tinygrad/llm/cli.py`) is OpenAI-compatible but speaks a subset of the
contract. Today: prompts are assembled from hardcoded per-preset role strings in
`SimpleTokenizer.role()`/`end_turn()`/`prefix()`; the streaming path decodes each generated token
straight into `delta.content` (via `SimpleTokenizer.stream_decoder()`) until an EOS/EOT token, with
support for prefill (a final assistant message), `max_tokens`, and usage. There is **no** reasoning
parser (`<think>` traces pass through verbatim as content), **no** `enable_thinking` toggle, **no**
stop-sequence detection, no tool calling, and no structured output — none of `ThinkParser`,
`StopChecker`, or `enable_thinking` exists in the codebase. Generation runs through
`Transformer.generate()`, whose `T=1` decode step (forward + gumbel-max sample + KV update) is fused
into a TinyJit-compiled graph.

The architectural constraint that shapes this design: tinygrad's analogue of a CUDA kernel layer is
the **TinyJit-compiled graph**. Work that runs per token over the whole vocab/all requests must stay
in that graph; per-request text-stream work belongs on the host. vLLM/SGLang prove that tool/reasoning
parsing in a host language (Python) is more than fast enough even under high-throughput batching,
because it runs at token cadence on an already-host-resident token stream. tinygrad is itself Python
end-to-end at the frontend, so this layer sits exactly where it belongs.

This change is also explicitly **backend-agnostic**: everything here except constrained decoding is
host-side and benefits CUDA/AMD/Apple/CPU/WEBGPU equally. AMD is the current proof-point, not a
constraint on this work.

## Goals / Non-Goals

**Goals:**
- Render prompts from the model's chat template (incl. `tools`), with preset fallback.
- Tool/function calling: request fields + native-format parsing → OpenAI `tool_calls` (stream + non-stream).
- Per-model reasoning-format abstraction; `reasoning_effort` + token budgets enforced host-side.
- `response_format`: JSON mode (host-side) now; schema/grammar-constrained decoding as an isolated
  sampler-integrated workstream.
- Generalize `ThinkParser`/`StopChecker` into one pluggable `StreamParser`.

**Non-Goals:**
- Tier-A scalability (continuous batching, paged/tiered KV, scheduling, auth, rate limiting).
- `/v1/embeddings`, `/v1/completions`, `n>1`, `logprobs`, multimodal.
- Being an embedded backend for oMLX/LM Studio (blocked/fork — see proposal motivation).

## Decisions

**D1 — Jinja chat templates for prompt building, minimal engine.** Replace template-free role
formatting with rendering the model's chat template. Source the template from GGUF KV (already
embedded) so no extra model files are needed. *Engine choice:* prefer a minimal embedded Jinja engine
(a `minja`-style subset, à la llama.cpp) over adding the full `transformers` dependency, to preserve
tinygrad's dependency-light philosophy; `jinja2` is the fallback if template coverage demands it.
Validate rendered bytes against `transformers.apply_chat_template` in tests only (test-time dep, not
runtime). *Alternative rejected:* keep hardcoded role strings — cannot render `tools` and diverges
from canonical formatting per model.

**D2 — `StreamParser` abstraction (per-model classes now, autoparser later).** Introduce a single
streaming parser that consumes the host token-text stream and yields tagged segments
(`content` | `reasoning` | `tool_call` | `stop`). This is a new layer — today the server decodes
tokens directly to `content` with no parsing — so "refactor" means giving the existing
decode-to-content path a parser seam, not rewriting existing parser classes. Near-term, implement
per-model parser configs by lifting + adapting vLLM's Apache-2.0 parser logic (Hermes/Qwen tool format
+ Qwen/DeepSeek reasoning) — copy-and-adapt, since vLLM's are coupled to its request/delta objects.
Design the interface so a future **template-derived autoparser** (llama.cpp `common/chat.cpp` style,
which infers format from the Jinja template) can replace the hand-written configs without changing
callers. *Alternative rejected:* port llama.cpp C++ directly — wrong language, and the per-model path
ships value sooner.

**D3 — Reasoning-effort enforced host-side by close-sequence injection.** Map `reasoning_effort`
levels to per-model thinking-token budgets; count reasoning tokens as they stream; when the budget is
hit, inject the model's reasoning-close sequence into the input stream to force the transition to
content. This mirrors the existing `enable_thinking=false` empty-block trick but applies it
mid-generation. No kernel-graph change. *Alternative rejected:* train/prompt-only control — not
deterministic enough to honor an API budget.

**D4 — Structured output in two tiers.** Tier 1: `json_object` mode = prompt-guided + parse/validate/
retry, fully host-side, ships with this change. Tier 2: `json_schema` = grammar-constrained decoding
via a per-token logit mask. The mask is the *only* part of this change that crosses into the TinyJit
decode graph: it requires injecting host-computed, per-step-varying data into the sampler each token,
which fights TinyJit's static-graph assumption. Approach: feed the mask as an input buffer threaded
into the decode graph and keep grammar advancement off the critical path; prototype the per-token cost
before committing. *Alternative rejected:* schema-by-retry only — cannot guarantee conformance and
wastes tokens on hard schemas.

**D5 — Additive, backward-compatible API.** All new request/response fields are additive. The one
behavioral change is prompt bytes (D1); existing chat-UI behavior that the server *does* have today —
plain `content` streaming, prefill, `max_tokens`, usage — is preserved through `StreamParser`.
(Reasoning split and stop handling are not existing behaviors; they are introduced by this change and
default-off where a model has no think format / the request sets no `stop`.) Regression tests lock the
preserved behaviors; new tests lock the introduced ones.

## Risks / Trade-offs

- **Prompt-byte change breaks output-byte-exact callers** → mitigate with golden tests vs.
  `transformers.apply_chat_template` per catalog model; document the change as behavioral
  improvement, not schema break.
- **Jinja engine under-covers some templates** → start minimal, measure against catalog templates;
  fall back to `jinja2` (or preset formatting) where a template fails to render.
- **Per-model parser sprawl** → the catalog's actual tool-call/reasoning conventions must be
  enumerated before committing effort; the `StreamParser` interface keeps the autoparser as an escape
  hatch from N-parser maintenance.
- **Constrained-decoding hot-loop cost in TinyJit** → isolate as the final sub-workstream; prototype
  the logit-mask injection early to size graph-rebuild/dispatch and host→device copy overhead before
  building it out. Ship JSON mode (host-side) first so structured output has value without it.
- **vLLM coupling** → Apache-2.0 is fine to adapt; budget rewrite effort to decouple from vLLM
  internals.

## Migration Plan

1. Land `StreamParser` as a new seam over the current decode-to-`content` path, preserving identical
   observable behavior for the cases the server already handles — plain content streaming, prefill,
   `max_tokens`, usage (regression-locked) — and adding think-split + stop-detection behind it. No API
   change yet beyond surfacing `reasoning_content`.
2. Land Jinja prompt rendering behind model-template detection with preset fallback (D1).
3. Add tool calling (request fields + Hermes/Qwen parser + streaming deltas).
4. Add reasoning-format abstraction + `reasoning_effort`/budgets.
5. Add `json_object` mode.
6. Prototype, then add `json_schema` constrained decoding (sampler integration), opt-in only.

Rollback: each step is additive/independent; constrained decoding (step 6) is fully gated by
`response_format` so it can be disabled without affecting other features.

## Open Questions

- Minimal Jinja engine vs. `jinja2` runtime dep — decide after measuring catalog template coverage.
- How many distinct tool-call/reasoning formats does the current `cli.py` catalog actually span?
  (Determines per-model-class vs. autoparser priority — enumerate before step 3.)
- Exact `reasoning_effort` → token-budget mapping per model family (configurable defaults).
- Feasibility/perf of per-token logit masking inside TinyJit — resolve via early prototype (step 6).
