## Why

tinygrad already ships an OpenAI-compatible LLM server (`tinygrad/llm/cli.py`), but it speaks only a
*subset* of the OpenAI contract: prompts are built from hardcoded per-preset role strings (not the
model's chat template), there is no tool/function calling, there is no reasoning control at all (the
server streams the raw token text as `content`, so any `<think>` trace a model emits is passed through
verbatim with no split, no `reasoning_content`, and no on/off toggle), there is no stop-sequence
handling, and there is no structured-output support. This is fine for the bundled
chat UI but breaks the moment a real client (an agent framework, or an app of the kind that drives
llama.cpp/MLX backends) sends `tools`, `tool_choice`, `reasoning_effort`, or `response_format`.

Closing this gap turns tinygrad into a genuinely portable local inference server — one small codebase
that codegens to whatever accelerator is present (CUDA, AMD/HIP, Apple, CPU, WEBGPU) — a story
neither the CUDA-first servers (vLLM/SGLang) nor the Apple-only ones (MLX/oMLX) can tell, and that
llama.cpp can only tell with a far larger hand-maintained per-backend kernel tree. All of the work in
this change is host-side and backend-agnostic, so it benefits every backend equally.

## What Changes

- **Prompt building** moves from hardcoded preset role strings to rendering the model's Jinja chat
  template (embedded in GGUF KV), with `tools` rendered into the prompt. Preset role formatting is
  retained only as a fallback when a model ships no template. **BREAKING** for any caller relying on
  the exact current prompt bytes (output text should improve, not regress).
- **Tool / function calling**: accept `tools` and `tool_choice` (`auto`/`none`/`required`/named) in
  requests; parse model-emitted tool-call syntax (Hermes/Qwen-style first) into OpenAI `tool_calls`;
  stream incremental `tool_calls` deltas; set `finish_reason: "tool_calls"`.
- **Reasoning control**: add `reasoning_effort` (`low|medium|high`) and an optional thinking-token
  budget, enforced host-side by injecting the close-think sequence mid-generation. Introduce a
  per-model reasoning-format abstraction (think-tag, DeepSeek variants, harmony channels). The server
  has no reasoning parser today, so this is new behavior surfaced as `reasoning_content`.
- **Structured output**: add `response_format` JSON mode (prompt-guided, parse + validate + retry,
  host-side) first; then grammar/schema-constrained decoding via a per-token logit mask threaded into
  the TinyJit decode graph (separate, harder sub-workstream — see design).
- **Refactor**: introduce a single pluggable streaming `StreamParser` that yields tagged segments
  (`content`/`reasoning`/`tool_call`/`stop`), with per-model parser selection. (The server currently
  has no streaming parser layer at all — it decodes tokens straight to `content` — so this is a new
  abstraction, not a refactor of existing parser classes.)

Out of scope (tracked separately as the spec's parallel Tier-A track): continuous batching, paged/
tiered KV cache, request scheduling, auth/rate limiting, `/v1/embeddings`, `/v1/completions`,
multimodal. These are a different subsystem (scheduler/KV-paging), not protocol fidelity.

## Capabilities

### New Capabilities
- `chat-prompt-templating`: Build request prompts by rendering the model's Jinja chat template
  (including `tools`), sourced from GGUF/model metadata, with preset role formatting as fallback and
  ephemeral-reasoning stripping for prior turns.
- `tool-calling`: Accept tool definitions and `tool_choice` in chat-completions requests, parse
  model-generated tool calls into OpenAI `tool_calls` (streaming and non-streaming), and report
  `finish_reason: "tool_calls"`.
- `reasoning-control`: Per-model reasoning-trace parsing into `reasoning_content`, with
  `reasoning_effort` levels and thinking-token budgets enforced host-side, plus an `enable_thinking`
  toggle. (Net-new: the server currently does no reasoning parsing and has no `enable_thinking`.)
- `structured-output`: `response_format` support — JSON mode (prompt-guided + validate/retry) and
  schema/grammar-constrained decoding via a logit mask integrated with the sampler.

### Modified Capabilities
<!-- None: no existing openspec/specs/ capabilities. The current server behavior lives in code only;
     these new capabilities formalize and extend it. -->

## Impact

- **Code**: `tinygrad/llm/cli.py` (server request/response handling, `SimpleTokenizer` prompt
  assembly, new `StreamParser` layer over the current decode-to-`content` path), `tinygrad/llm/model.py`
  (`Transformer.generate()` sampler hook for constrained decoding), `tinygrad/llm/gguf.py` (expose
  embedded chat-template + tokenizer metadata). New parser/template modules under `tinygrad/llm/`.
- **API surface**: additive request fields (`tools`, `tool_choice`, `reasoning_effort`,
  `response_format`) and response fields (`delta.tool_calls`, `finish_reason: "tool_calls"`); changed
  prompt bytes (behavioral, not schema).
- **Dependencies**: introduces a Jinja rendering path. Decision (design.md): minimal embedded engine
  vs. `jinja2`/`transformers` — tinygrad has deliberately avoided heavy deps, so lean minimal.
- **Performance**: parsing/template/effort work is host-side, post-JIT, negligible at token cadence.
  Only constrained decoding touches the TinyJit decode graph (per-token logit mask) and must be sized
  carefully; it is isolated as the final sub-workstream.
- **Tests**: `test/unit/test_llm_server.py`, `test/null/test_llm_server.py`; new golden-prompt tests
  vs. `transformers.apply_chat_template` reference.
