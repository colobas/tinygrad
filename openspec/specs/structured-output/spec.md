# structured-output Specification

## Purpose

Define how the LLM server produces structured output: a host-side JSON object mode with validation and retry, and schema/grammar-constrained decoding enforced via a per-token logit mask that stays off the decode loop's critical path.

## Requirements

### Requirement: JSON output mode

The endpoint SHALL accept `response_format` of type `json_object`. In this mode the server SHALL guide
the model to produce JSON, parse the result, and validate that it is well-formed JSON. If validation
fails, the server SHALL retry up to a configured limit before returning an error. This mode SHALL be
implemented host-side and SHALL NOT require changes to the per-token kernel graph.

#### Scenario: Valid JSON produced

- **WHEN** a request sets `response_format: { "type": "json_object" }` and the model produces
  well-formed JSON
- **THEN** the response `content` is that JSON and `finish_reason` is `"stop"`

#### Scenario: Invalid JSON triggers retry

- **WHEN** the model's output in JSON mode is not well-formed JSON
- **THEN** the server retries generation up to the configured retry limit
- **AND** if no attempt yields valid JSON, the server returns an error rather than malformed content

### Requirement: Schema/grammar-constrained decoding

The endpoint SHALL accept `response_format` of type `json_schema` and enforce it by constraining
decoding so that only tokens permitted by the schema/grammar can be sampled. Enforcement SHALL be
implemented as a per-token logit mask integrated with the sampler, and grammar advancement SHALL be
kept off the critical path of the decode loop.

#### Scenario: Output conforms to schema

- **WHEN** a request sets `response_format` to a `json_schema`
- **THEN** every token sampled is permitted by the grammar derived from that schema
- **AND** the final `content` validates against the schema

#### Scenario: Constrained decoding is opt-in

- **WHEN** a request does not specify a `json_schema` response format
- **THEN** decoding proceeds without any logit masking and the decode-loop performance is unchanged
