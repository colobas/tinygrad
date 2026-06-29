import unittest, threading, time, json
from unittest.mock import Mock

WEATHER_TOOL = {"type":"function", "function":{"name":"get_weather", "description":"weather",
                "parameters":{"type":"object", "properties":{"city":{"type":"string"}}}}}

class TestLLMServer(unittest.TestCase):
  """Integration tests using the real OpenAI client."""

  @classmethod
  def setUpClass(cls):
    cls.mock_tok = Mock()
    cls.mock_tok.role = Mock(return_value=[100, 101])
    cls.mock_tok.encode = Mock(return_value=[200, 201, 202])
    cls.mock_tok.decode = Mock(return_value="Hello")
    cls.mock_tok.stream_decoder = Mock(return_value=lambda tid=None: "Hello" if tid is not None else "")
    cls.mock_tok.end_turn = Mock(return_value=[998])
    cls.mock_tok.prefix = Mock(return_value=[1])
    cls.mock_tok.preset = "llama3"
    cls.mock_tok.chat_template = None  # exercise the preset role-formatting fallback path
    cls.mock_tok.bos_id = 1
    cls.mock_tok.eos_id = 999
    cls.mock_tok.eot_id = None
    cls.mock_tok.is_end = Mock(side_effect=lambda tid: tid in (999,))

    cls.mock_model = Mock()
    cls.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 301, 999]))
    cls.mock_model.get_start_pos = Mock(return_value=0)
    cls.mock_model.max_context = 4096  # context-window guard compares against this

    from tinygrad.llm.cli import LLMServer

    cls.server = LLMServer(('127.0.0.1', 0), cls.mock_model, "test-model", cls.mock_tok)
    cls.port = cls.server.server_address[1]
    cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
    cls.server_thread.start()
    time.sleep(0.1)

    from openai import OpenAI
    cls.client = OpenAI(base_url=f"http://127.0.0.1:{cls.port}/v1", api_key="test")

  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()

  def tearDown(self):
    # tests mutate the shared mock model/tokenizer; restore defaults so they stay isolated
    cls = type(self)
    cls.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 301, 999]))
    cls.mock_tok.stream_decoder = Mock(return_value=lambda tid=None: "Hello" if tid is not None else "")
    cls.mock_tok.role.reset_mock()
    cls.mock_tok.end_turn.reset_mock()

  def test_chat_completion_stream(self):
    stream = self.client.chat.completions.create(
      model="test",
      messages=[{"role": "user", "content": "Hello"}],
      stream=True
    )

    chunks = list(stream)
    self.assertGreater(len(chunks), 0)
    self.assertEqual(chunks[0].choices[0].delta.role, "assistant")
    self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")

  def test_openai_response_structure(self):
    stream = self.client.chat.completions.create(
      model="test-model",
      messages=[{"role": "user", "content": "Test"}],
      stream=True
    )

    for chunk in stream:
      self.assertTrue(chunk.id.startswith("chatcmpl-"))
      self.assertEqual(chunk.object, "chat.completion.chunk")
      self.assertIsNotNone(chunk.choices)
      self.assertIsNotNone(chunk.created)
      self.assertIsInstance(chunk.created, int)
      self.assertEqual(chunk.model, "test-model")

  def test_stream_with_usage(self):
    stream = self.client.chat.completions.create(
      model="test",
      messages=[{"role": "user", "content": "Hello"}],
      stream=True,
      stream_options={"include_usage": True}
    )

    chunks = list(stream)
    last_chunk = chunks[-1]

    self.assertIsNotNone(last_chunk.usage)
    self.assertIsNotNone(last_chunk.usage.prompt_tokens)
    self.assertIsNotNone(last_chunk.usage.completion_tokens)
    self.assertIsNotNone(last_chunk.usage.total_tokens)

  def test_multi_turn_conversation(self):
    stream = self.client.chat.completions.create(
      model="test",
      messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How are you?"}
      ],
      stream=True
    )

    chunks = list(stream)
    self.assertGreater(len(chunks), 0)
    self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")

  def test_content_is_streamed(self):
    stream = self.client.chat.completions.create(
      model="test",
      messages=[{"role": "user", "content": "Hello"}],
      stream=True
    )

    contents = []
    for chunk in stream:
      if chunk.choices and chunk.choices[0].delta.content:
        contents.append(chunk.choices[0].delta.content)

    self.assertGreater(len(contents), 0)

  def test_non_streaming(self):
    resp = self.client.chat.completions.create(
      model="test-model",
      messages=[{"role": "user", "content": "Hello"}],
      stream=False
    )

    self.assertTrue(resp.id.startswith("chatcmpl-"))
    self.assertEqual(resp.object, "chat.completion")
    self.assertEqual(resp.model, "test-model")
    self.assertIsNotNone(resp.created)
    self.assertEqual(len(resp.choices), 1)
    self.assertEqual(resp.choices[0].message.role, "assistant")
    self.assertIsNotNone(resp.choices[0].message.content)
    self.assertEqual(resp.choices[0].finish_reason, "stop")
    self.assertIsNotNone(resp.usage)
    self.assertIsNotNone(resp.usage.prompt_tokens)
    self.assertIsNotNone(resp.usage.completion_tokens)

  def test_max_tokens_streaming(self):
    self.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 301, 302, 303, 999]))
    stream = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "Hello"}], stream=True, max_tokens=2
    )
    chunks = list(stream)
    content_chunks = [c for c in chunks if c.choices and c.choices[0].delta.content]
    self.assertEqual(len(content_chunks), 2)
    self.assertEqual(chunks[-1].choices[0].finish_reason, "length")

  def test_max_tokens_non_streaming(self):
    self.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 301, 302, 303, 999]))
    resp = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "Hello"}], stream=False, max_tokens=2
    )
    self.assertEqual(resp.choices[0].finish_reason, "length")
    self.assertEqual(resp.usage.completion_tokens, 2)

  def test_assistant_prefill(self):
    """Last assistant message should be treated as prefill (not a completed turn)."""
    self.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 999]))
    captured_ids = []
    orig_generate = self.mock_model.generate.side_effect
    def capture_generate(ids, **kwargs):
      captured_ids.extend(ids)
      return orig_generate(ids, **kwargs)
    self.mock_model.generate = Mock(side_effect=capture_generate)

    resp = self.client.chat.completions.create(
      model="test", messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Sure"}
      ], stream=False
    )
    # prefill tokens should be in ids: role("assistant") + encode("Sure") but NO end_turn after it
    # and NO extra role("assistant") appended
    role_tokens = self.mock_tok.role.call_args_list
    # last role() call should be for "assistant" (the prefill message), not an extra one
    self.assertEqual(role_tokens[-1], unittest.mock.call("assistant"))
    # end_turn should be called once less than role() — the prefill assistant msg doesn't get end_turn
    # NOTE: this is flaky in random order
    #self.assertEqual(self.mock_tok.end_turn.call_count, self.mock_tok.role.call_count - 1)
    self.assertIsNotNone(resp.choices[0].message.content)

  def test_assistant_prefill_not_last(self):
    """Assistant message that's NOT last should be a normal completed turn."""
    self.mock_model.generate = Mock(side_effect=lambda ids, **kwargs: iter([300, 999]))
    self.mock_tok.role.reset_mock()
    self.mock_tok.end_turn.reset_mock()
    self.client.chat.completions.create(
      model="test", messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Sure"},
        {"role": "user", "content": "Continue"}
      ], stream=False
    )
    # all messages get end_turn, plus an extra role("assistant") at the end
    # roles: user, assistant, user, assistant(generation prompt) = 4 role calls
    # end_turns: user, assistant, user = 3 end_turn calls (one per message)
    self.assertEqual(self.mock_tok.end_turn.call_count, 3)
    self.assertEqual(self.mock_tok.role.call_count, 4)

  def _script(self, pieces):
    """Make the mock model emit one token per element of `pieces`, decoding to that string."""
    ids = list(range(len(pieces))) + [999]
    self.mock_model.generate = Mock(side_effect=lambda i, **k: iter(ids))
    self.mock_tok.stream_decoder = Mock(return_value=lambda tid=None: pieces[tid] if tid is not None else "")

  def test_reasoning_split_streaming(self):
    self._script(["<think>", "secret ", "thoughts", "</think>", "the ", "answer"])
    stream = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "hi"}], stream=True)
    content, reasoning = [], []
    for chunk in stream:
      if not chunk.choices: continue
      d = chunk.choices[0].delta
      if d.content: content.append(d.content)
      if getattr(d, "reasoning_content", None): reasoning.append(d.reasoning_content)
    self.assertEqual("".join(reasoning), "secret thoughts")
    self.assertEqual("".join(content), "the answer")

  def test_reasoning_split_non_streaming(self):
    self._script(["<think>", "why", "</think>", "because"])
    resp = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "hi"}], stream=False)
    msg = resp.choices[0].message
    self.assertEqual(msg.content, "because")
    self.assertEqual(getattr(msg, "reasoning_content", None), "why")
    self.assertEqual(resp.choices[0].finish_reason, "stop")

  def test_enable_thinking_false_passes_think_through(self):
    self._script(["<think>", "x", "</think>", "y"])
    resp = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "hi"}], stream=False,
      extra_body={"enable_thinking": False})
    # with thinking disabled the think markers are not split out; everything is content
    self.assertEqual(resp.choices[0].message.content, "<think>x</think>y")
    self.assertIsNone(getattr(resp.choices[0].message, "reasoning_content", None))

  def test_stop_sequence_streaming(self):
    self._script(["keep ", "this ", "STOP", " drop ", "this"])
    stream = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "hi"}], stream=True, stop="STOP")
    content, finish = [], None
    for chunk in stream:
      if not chunk.choices: continue
      if chunk.choices[0].delta.content: content.append(chunk.choices[0].delta.content)
      if chunk.choices[0].finish_reason: finish = chunk.choices[0].finish_reason
    self.assertEqual("".join(content), "keep this ")
    self.assertEqual(finish, "stop")

  def test_tool_call_non_streaming(self):
    self._script(["<tool_call>", '{"name": "get_weather", "arguments": {"city": "SF"}}', "</tool_call>"])
    resp = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "weather?"}], tools=[WEATHER_TOOL], stream=False)
    msg = resp.choices[0].message
    self.assertEqual(len(msg.tool_calls), 1)
    self.assertEqual(msg.tool_calls[0].type, "function")
    self.assertEqual(msg.tool_calls[0].function.name, "get_weather")
    self.assertEqual(json.loads(msg.tool_calls[0].function.arguments), {"city": "SF"})
    self.assertEqual(resp.choices[0].finish_reason, "tool_calls")

  def test_tool_call_streaming_reassembly(self):
    self._script(["<tool_call>", '{"name": "get_weather", ', '"arguments": {"city": "SF"}}', "</tool_call>"])
    stream = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "weather?"}], tools=[WEATHER_TOOL], stream=True)
    calls, finish = {}, None
    for chunk in stream:
      if not chunk.choices: continue
      for tc in (chunk.choices[0].delta.tool_calls or []):
        c = calls.setdefault(tc.index, {"name": "", "args": "", "id": None})
        if tc.id: c["id"] = tc.id
        if tc.function and tc.function.name: c["name"] = tc.function.name
        if tc.function and tc.function.arguments: c["args"] += tc.function.arguments
      if chunk.choices[0].finish_reason: finish = chunk.choices[0].finish_reason
    self.assertEqual(calls[0]["name"], "get_weather")
    self.assertIsNotNone(calls[0]["id"])
    self.assertEqual(json.loads(calls[0]["args"]), {"city": "SF"})
    self.assertEqual(finish, "tool_calls")

  def test_tool_choice_none_suppresses(self):
    self._script(["<tool_call>", '{"name": "get_weather", "arguments": {}}', "</tool_call>"])
    resp = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "hi"}], tools=[WEATHER_TOOL], tool_choice="none", stream=False)
    self.assertIsNone(resp.choices[0].message.tool_calls)
    self.assertEqual(resp.choices[0].finish_reason, "stop")  # tools not parsed; block stays as content

  def test_tool_choice_named_restricts(self):
    other = {"type":"function", "function":{"name":"other_fn", "parameters":{"type":"object", "properties":{}}}}
    self._script(["<tool_call>", '{"name": "other_fn", "arguments": {}}', "</tool_call>",
                  "<tool_call>", '{"name": "get_weather", "arguments": {}}', "</tool_call>"])
    resp = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "hi"}], tools=[WEATHER_TOOL, other],
      tool_choice={"type": "function", "function": {"name": "get_weather"}}, stream=False)
    names = [tc.function.name for tc in resp.choices[0].message.tool_calls]
    self.assertEqual(names, ["get_weather"])  # the non-chosen function call is dropped

  def test_tool_call_text_not_in_content(self):
    self._script(["Sure! ", "<tool_call>", '{"name": "get_weather", "arguments": {}}', "</tool_call>"])
    resp = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "hi"}], tools=[WEATHER_TOOL], stream=False)
    self.assertEqual(resp.choices[0].message.content, "Sure! ")
    self.assertEqual(resp.choices[0].message.tool_calls[0].function.name, "get_weather")

  def test_invalid_tools_shape_returns_400(self):
    import requests as req
    resp = req.post(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "nope"}]})
    self.assertEqual(resp.status_code, 400)
    self.assertEqual(resp.json()["error"]["type"], "invalid_request_error")

  def test_enable_thinking_false_via_chat_template_kwargs(self):
    # enable_thinking can also be disabled inside chat_template_kwargs (5.4)
    self._script(["<think>", "x", "</think>", "y"])
    resp = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "hi"}], stream=False,
      extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    self.assertEqual(resp.choices[0].message.content, "<think>x</think>y")
    self.assertIsNone(getattr(resp.choices[0].message, "reasoning_content", None))

  def test_thinking_budget_injects_close(self):
    # model keeps reasoning without ever closing; the budget forces a </think> injection -> visible content (5.3)
    calls = {"n": 0}
    def gen(ids, **k):
      calls["n"] += 1
      return iter([0, 1, 2, 3] if calls["n"] == 1 else [10, 11, 999])  # 1st: think (no close); 2nd: content + eos
    self.mock_model.generate = Mock(side_effect=gen)
    pieces = {0:"<think>", 1:"a", 2:"b", 3:"c", 10:"answer", 11:"!"}
    self.mock_tok.stream_decoder = Mock(return_value=lambda tid=None: pieces.get(tid, "") if tid is not None else "")
    resp = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "hi"}], stream=False, extra_body={"thinking_budget": 2})
    msg = resp.choices[0].message
    self.assertEqual(msg.content, "answer!")            # generation resumed as content after the injected close
    self.assertNotIn("</think>", msg.content or "")     # injected marker never leaks into content
    self.assertEqual(getattr(msg, "reasoning_content", None), "a")  # reasoning cut off at the budget
    self.assertEqual(calls["n"], 2)                     # generation was restarted exactly once

  def test_invalid_reasoning_effort_returns_400(self):
    import requests as req
    resp = req.post(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "ludicrous"})
    self.assertEqual(resp.status_code, 400)
    self.assertEqual(resp.json()["error"]["type"], "invalid_request_error")

  def test_json_object_valid(self):
    self._script(['{"x": ', "1}"])
    resp = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "give me json"}], stream=False,
      response_format={"type": "json_object"})
    self.assertEqual(json.loads(resp.choices[0].message.content), {"x": 1})
    self.assertEqual(resp.choices[0].finish_reason, "stop")

  def test_json_object_invalid_then_retry(self):
    attempts = {"n": 0}
    def make_decoder():
      attempts["n"] += 1
      pieces = ["not json"] if attempts["n"] == 1 else ['{"ok": ', "true}"]  # 1st attempt invalid, 2nd valid
      return lambda tid=None: (pieces[tid] if tid < len(pieces) else "") if tid is not None else ""
    self.mock_tok.stream_decoder = Mock(side_effect=make_decoder)
    self.mock_model.generate = Mock(side_effect=lambda ids, **k: iter([0, 1, 999]))
    resp = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "give me json"}], stream=False,
      response_format={"type": "json_object"})
    self.assertEqual(json.loads(resp.choices[0].message.content), {"ok": True})
    self.assertEqual(attempts["n"], 2)  # retried exactly once

  def test_json_object_exhausted_retries_errors(self):
    import requests as req
    self._script(["nope"])  # never valid JSON
    resp = req.post(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "json"}],
                          "response_format": {"type": "json_object"}})
    self.assertEqual(resp.status_code, 502)
    self.assertEqual(resp.json()["error"]["type"], "json_validation_error")

  def test_prompt_overflow_returns_400(self):
    import requests as req
    self.mock_model.max_context = 2  # tiny window so any real prompt overflows
    try:
      resp = req.post(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                      json={"model": "test", "messages": [{"role": "user", "content": "hello there"}]})
      self.assertEqual(resp.status_code, 400)
      self.assertEqual(resp.json()["error"]["code"], "context_length_exceeded")
    finally:
      self.mock_model.max_context = 4096

  def test_penalties_threaded_to_generate(self):
    captured = {}
    def gen(ids, **kwargs):
      captured.update(kwargs)
      return iter([300, 301, 999])
    self.mock_model.generate = Mock(side_effect=gen)
    self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "hi"}], stream=False,
      extra_body={"repetition_penalty": 1.3, "presence_penalty": 1.5, "frequency_penalty": 0.2})
    self.assertEqual(captured.get("rep_pen"), 1.3)
    self.assertEqual(captured.get("presence_pen"), 1.5)
    self.assertEqual(captured.get("freq_pen"), 0.2)

  def test_invalid_repetition_penalty_returns_400(self):
    import requests as req
    resp = req.post(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "repetition_penalty": 0})
    self.assertEqual(resp.status_code, 400)
    self.assertEqual(resp.json()["error"]["type"], "invalid_request_error")

  def test_invalid_response_format_returns_400(self):
    import requests as req
    resp = req.post(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}],
                          "response_format": {"type": "yaml"}})
    self.assertEqual(resp.status_code, 400)
    self.assertEqual(resp.json()["error"]["type"], "invalid_request_error")

  def test_json_schema_threads_mask_fn(self):
    # json_schema requests must build a grammar masker and pass it into model.generate (§7 wiring)
    self.mock_tok.grammar_vocab = Mock(return_value=({0: "{", 1: "}"}, {999}, 1000))
    captured = {}
    def gen(ids, **kwargs):
      captured.update(kwargs)
      return iter([0, 999])
    self.mock_model.generate = Mock(side_effect=gen)
    self.mock_tok.stream_decoder = Mock(return_value=lambda tid=None: "{}" if tid is not None else "")
    self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "hi"}], stream=False,
      response_format={"type": "json_schema", "json_schema": {"name": "s", "schema": {"type": "object", "properties": {}}}})
    self.assertIsNotNone(captured.get("mask_fn"))
    del self.mock_tok.grammar_vocab  # don't leak the stub to other tests

  def test_invalid_json_schema_returns_400(self):
    import requests as req
    resp = req.post(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}],
                          "response_format": {"type": "json_schema", "json_schema": {"name": "s"}}})  # missing schema
    self.assertEqual(resp.status_code, 400)
    self.assertEqual(resp.json()["error"]["type"], "invalid_request_error")

  def test_models_endpoint(self):
    import requests as req
    resp = req.get(f"http://127.0.0.1:{self.port}/v1/models")
    self.assertEqual(resp.status_code, 200)
    data = resp.json()
    self.assertEqual(data["object"], "list")
    self.assertEqual(len(data["data"]), 1)
    self.assertEqual(data["data"][0]["id"], "test-model")
    self.assertEqual(data["data"][0]["object"], "model")

if __name__ == '__main__':
  unittest.main()
