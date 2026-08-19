## ADDED Requirements

### Requirement: Accept tool definitions and tool choice

The chat-completions endpoint SHALL accept the OpenAI `tools` array and `tool_choice` field
(`"auto"`, `"none"`, `"required"`, or a named function). The server SHALL make these available to
prompt rendering and SHALL honor `tool_choice` semantics when deciding whether a tool call is
permitted.

#### Scenario: tool_choice "none" suppresses tools

- **WHEN** a request includes `tools` and `tool_choice: "none"`
- **THEN** the model is not offered the tools for invocation and the response contains no `tool_calls`

#### Scenario: tool_choice named function

- **WHEN** a request includes `tool_choice` naming a specific function
- **THEN** the response, if it calls a tool, calls only that function

### Requirement: Parse model-generated tool calls

The server SHALL parse tool calls emitted by the model in that model's native syntax (Hermes/Qwen-
style first) and SHALL surface them as OpenAI `tool_calls` entries, each with an `id`, function
`name`, and JSON `arguments` string.

#### Scenario: Single tool call (non-streaming)

- **WHEN** the model emits one tool call in its native format and the request is non-streaming
- **THEN** the response message contains a `tool_calls` array with one entry whose `name` and
  `arguments` match the model output
- **AND** the response `finish_reason` is `"tool_calls"`

#### Scenario: Tool call arguments span multiple tokens (streaming)

- **WHEN** the request is streaming and the model emits a tool call whose `arguments` are produced
  across many tokens
- **THEN** the server streams incremental `delta.tool_calls` chunks (function `name` and `id` once,
  then incremental `arguments` fragments) that reassemble into the complete tool call
- **AND** the terminating chunk has `finish_reason: "tool_calls"`

### Requirement: Tool-call text excluded from content

Text consumed as part of a tool call SHALL NOT also appear in the assistant `content`/`delta.content`
stream.

#### Scenario: Mixed content then tool call

- **WHEN** the model emits visible content followed by a tool-call block
- **THEN** the visible content is emitted as `content` and the tool-call block is emitted only as
  `tool_calls`, with no duplication
