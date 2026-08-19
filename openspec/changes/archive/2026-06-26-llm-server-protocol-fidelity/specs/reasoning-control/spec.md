## ADDED Requirements

### Requirement: Per-model reasoning-trace parsing

The server SHALL parse model reasoning traces into the OpenAI `reasoning_content` field using a
per-model reasoning format, rather than assuming a single hardcoded `<think>...</think>` convention.
Visible answer text SHALL be emitted as `content`.

#### Scenario: Model using think tags

- **WHEN** a model that uses `<think>...</think>` emits a reasoning trace followed by an answer
- **THEN** the trace is emitted as `reasoning_content` and the answer as `content`

#### Scenario: Model using a non-think reasoning format

- **WHEN** a model whose reasoning format differs from `<think>` (e.g. a channel-based or DeepSeek-
  style format) emits a reasoning trace
- **THEN** the server uses that model's configured reasoning format to split reasoning from content
  correctly

#### Scenario: Tag straddles a token boundary

- **WHEN** a reasoning open/close marker is split across two generated tokens
- **THEN** the parser still detects the marker and classifies the surrounding text correctly

### Requirement: Reasoning effort levels and token budget

The endpoint SHALL accept `reasoning_effort` (`"low"`, `"medium"`, `"high"`) and/or an explicit
thinking-token budget. The server SHALL enforce a budget host-side: when the reasoning-token budget is
reached, it SHALL inject the model's reasoning-close sequence to force transition to visible content.
This enforcement SHALL NOT require changes to the per-token kernel graph.

#### Scenario: Budget reached mid-reasoning

- **WHEN** the model is still emitting reasoning and the configured thinking-token budget is reached
- **THEN** the server injects the reasoning-close sequence so the model transitions to producing
  `content`

#### Scenario: reasoning_effort maps to a budget

- **WHEN** a request specifies `reasoning_effort: "low"`
- **THEN** the server applies the (configurable) low-effort token budget for that model

### Requirement: Thinking can be disabled

The endpoint SHALL continue to honor disabling thinking (via `enable_thinking` at the top level or in
`chat_template_kwargs`), causing the model to skip reasoning and produce content directly.

#### Scenario: Thinking disabled

- **WHEN** a request sets `enable_thinking: false`
- **THEN** the response contains no `reasoning_content` and the model emits visible content directly

### Requirement: Stop sequences apply to content only

`stop` sequences SHALL be detected on the visible `content` stream only and SHALL NOT terminate or be
matched against reasoning traces. Detection SHALL be correct when a stop sequence is split across
generated tokens.

#### Scenario: Stop string inside reasoning is ignored

- **WHEN** a configured stop string appears within a reasoning trace
- **THEN** generation is not stopped by that occurrence

#### Scenario: Stop string split across tokens

- **WHEN** a stop string is produced across multiple tokens in the content stream
- **THEN** the server detects it, stops generation, and does not emit the stop string or any text
  after it
