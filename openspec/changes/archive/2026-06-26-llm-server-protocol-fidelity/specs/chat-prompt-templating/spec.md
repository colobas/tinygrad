## ADDED Requirements

### Requirement: Render prompts from the model's chat template

The server SHALL build the request prompt by rendering the model's chat template (e.g. the Jinja
template embedded in GGUF metadata) over the request `messages`, rather than concatenating hardcoded
per-preset role strings. The template SHALL receive the message list and, when present, the request's
`tools` and any `chat_template_kwargs`.

#### Scenario: Template present in model metadata

- **WHEN** a chat-completions request is received for a model whose metadata contains a chat template
- **THEN** the server renders that template over the messages to produce the prompt token ids
- **AND** the rendered bytes match the reference produced by `transformers.apply_chat_template` for the
  same model and messages (validated by golden tests for each catalog model)

#### Scenario: Tools rendered into the prompt

- **WHEN** the request includes a non-empty `tools` array
- **THEN** the tool schemas are passed to the template and appear in the rendered prompt in the format
  the model expects

### Requirement: Fallback to preset role formatting

When a model exposes no usable chat template, the server SHALL fall back to the existing preset role
formatting so that previously supported models continue to work.

#### Scenario: Model without an embedded template

- **WHEN** a request targets a model whose metadata contains no chat template
- **THEN** the server uses the preset role formatting for that model's tokenizer preset
- **AND** the response is produced without error

### Requirement: Ephemeral prior-turn reasoning

The server SHALL NOT re-encode prior-turn reasoning traces into the prompt. Reasoning content from
earlier assistant turns (whether delivered as `reasoning_content` or embedded as think blocks in
`content`) SHALL be stripped before the turn is rendered into the prompt.

#### Scenario: Prior assistant turn contained a think block

- **WHEN** a non-final assistant message in the request contains a `<think>...</think>` block in its
  content
- **THEN** the think block is removed before that message is rendered into the prompt

#### Scenario: Final assistant message is a prefill

- **WHEN** the final message is an assistant message
- **THEN** it is treated as a generation prefill and its content is rendered without an end-of-turn
  marker so the model continues it
