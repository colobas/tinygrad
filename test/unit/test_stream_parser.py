import unittest, json
from tinygrad.llm.parser import StreamParser, Segment, parse_tool_call_body, detect_reasoning_format, DEFAULT_REASONING

THINK = ("<think>", "</think>")
TOOL = ("<tool_call>", "</tool_call>")

def run(parser:StreamParser, chunks:list[str]) -> list[Segment]:
  out: list[Segment] = []
  for c in chunks: out += list(parser.feed(c))
  out += list(parser.flush())
  return out

def joined(segs:list[Segment], kind:str) -> str: return "".join(s.text for s in segs if s.kind == kind)

class TestStreamParserContent(unittest.TestCase):
  def test_identity_no_config(self):
    # with no reasoning/stop config every chunk passes straight through as content (prior behavior)
    segs = run(StreamParser(), ["Hello", " ", "world"])
    self.assertTrue(all(s.kind == "content" for s in segs))
    self.assertEqual(joined(segs, "content"), "Hello world")

  def test_empty_feeds_emit_nothing(self):
    segs = run(StreamParser(), ["", "", "hi", ""])
    self.assertEqual(joined(segs, "content"), "hi")

class TestStreamParserReasoning(unittest.TestCase):
  def test_think_split_single_feed(self):
    segs = run(StreamParser(reasoning=THINK), ["<think>reasoning here</think>the answer"])
    self.assertEqual(joined(segs, "reasoning"), "reasoning here")
    self.assertEqual(joined(segs, "content"), "the answer")

  def test_content_before_think(self):
    segs = run(StreamParser(reasoning=THINK), ["pre <think>r</think> post"])
    self.assertEqual(joined(segs, "content"), "pre  post")
    self.assertEqual(joined(segs, "reasoning"), "r")

  def test_markers_split_across_tokens(self):
    # open and close markers each arrive split across several feeds
    segs = run(StreamParser(reasoning=THINK), ["<th", "ink>", "abc", "</thi", "nk>", "done"])
    self.assertEqual(joined(segs, "reasoning"), "abc")
    self.assertEqual(joined(segs, "content"), "done")

  def test_reasoning_started_implicit_open(self):
    # generation begins inside reasoning (template/prefill opened <think> itself), only close emitted
    segs = run(StreamParser(reasoning=THINK, reasoning_started=True), ["thoughts</think>answer"])
    self.assertEqual(joined(segs, "reasoning"), "thoughts")
    self.assertEqual(joined(segs, "content"), "answer")

  def test_unclosed_reasoning_drains_on_flush(self):
    segs = run(StreamParser(reasoning=THINK), ["<think>still thinking"])
    self.assertEqual(joined(segs, "reasoning"), "still thinking")
    self.assertEqual(joined(segs, "content"), "")

  def test_no_marker_text_not_held_forever(self):
    # plain text that shares a prefix char with the marker still flushes
    segs = run(StreamParser(reasoning=THINK), ["a < b"])
    self.assertEqual(joined(segs, "content"), "a < b")

  def test_partial_marker_at_end_held_until_resolved(self):
    p = StreamParser(reasoning=THINK)
    # feeding text ending in "<" must not emit the "<" yet (could be start of "<think>")
    first = list(p.feed("hello<"))
    self.assertEqual(joined(first, "content"), "hello")
    # next feed proves it was ordinary text
    rest = list(p.feed("3 things")) + list(p.flush())
    self.assertEqual(joined(rest, "content"), "<3 things")

class TestStreamParserStop(unittest.TestCase):
  def test_stop_truncates_content(self):
    segs = run(StreamParser(stop=["STOP"]), ["keep this STOP drop this"])
    self.assertEqual(joined(segs, "content"), "keep this ")
    self.assertTrue(any(s.kind == "stop" and s.text == "STOP" for s in segs))

  def test_stop_split_across_tokens(self):
    segs = run(StreamParser(stop=["<|end|>"]), ["text <|en", "d|> after"])
    self.assertEqual(joined(segs, "content"), "text ")
    self.assertTrue(any(s.kind == "stop" for s in segs))

  def test_stop_inside_reasoning_ignored(self):
    segs = run(StreamParser(reasoning=THINK, stop=["STOP"]), ["<think>STOP in here</think>real STOP gone"])
    self.assertEqual(joined(segs, "reasoning"), "STOP in here")
    self.assertEqual(joined(segs, "content"), "real ")
    self.assertTrue(any(s.kind == "stop" for s in segs))

  def test_no_output_after_stop(self):
    p = StreamParser(stop=["END"])
    list(p.feed("a END b"))
    # further feeds produce nothing once stopped
    self.assertEqual(list(p.feed("more")), [])
    self.assertEqual(list(p.flush()), [])

def tool_calls(segs:list[Segment]) -> list[dict]: return [json.loads(s.text) for s in segs if s.kind == "tool_call"]

class TestParseToolCallBody(unittest.TestCase):
  def test_hermes_json(self):
    call = parse_tool_call_body('\n{"name": "get_weather", "arguments": {"city": "SF"}}\n')
    self.assertEqual(call, {"name": "get_weather", "arguments": '{"city": "SF"}'})
  def test_hermes_parameters_alias(self):
    call = parse_tool_call_body('{"name": "f", "parameters": {"x": 1}}')
    self.assertEqual(call, {"name": "f", "arguments": '{"x": 1}'})
  def test_qwen_xml(self):
    body = "\n<function=get_weather>\n<parameter=city>\nSF\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n"
    call = parse_tool_call_body(body)
    self.assertEqual(call["name"], "get_weather")
    self.assertEqual(json.loads(call["arguments"]), {"city": "SF", "days": 3})  # numeric value coerced
  def test_unparseable_returns_none(self):
    self.assertIsNone(parse_tool_call_body("just some text"))
    self.assertIsNone(parse_tool_call_body('{"no_name": 1}'))

class TestStreamParserToolCalls(unittest.TestCase):
  def test_single_hermes_call(self):
    segs = run(StreamParser(tool_call=TOOL), ['<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>'])
    self.assertEqual(tool_calls(segs), [{"name": "f", "arguments": '{"a": 1}'}])
    self.assertEqual(joined(segs, "content"), "")  # block text not duplicated into content

  def test_content_then_tool_call(self):
    segs = run(StreamParser(tool_call=TOOL), ['Sure! <tool_call>{"name": "f", "arguments": {}}</tool_call>'])
    self.assertEqual(joined(segs, "content"), "Sure! ")
    self.assertEqual(tool_calls(segs), [{"name": "f", "arguments": "{}"}])

  def test_two_calls(self):
    segs = run(StreamParser(tool_call=TOOL),
               ['<tool_call>{"name": "a", "arguments": {}}</tool_call>\n<tool_call>{"name": "b", "arguments": {}}</tool_call>'])
    self.assertEqual([c["name"] for c in tool_calls(segs)], ["a", "b"])

  def test_call_split_across_tokens(self):
    segs = run(StreamParser(tool_call=TOOL), ["<tool", "_call>", '{"name": "f", ', '"arguments": {"k": "v"}}', "</tool", "_call>"])
    self.assertEqual(tool_calls(segs), [{"name": "f", "arguments": '{"k": "v"}'}])
    self.assertEqual(joined(segs, "content"), "")

  def test_qwen_xml_call(self):
    segs = run(StreamParser(tool_call=TOOL), ["<tool_call>\n<function=f>\n<parameter=x>\nhi\n</parameter>\n</function>\n</tool_call>"])
    call = tool_calls(segs)[0]
    self.assertEqual(call["name"], "f")
    self.assertEqual(json.loads(call["arguments"]), {"x": "hi"})

  def test_reasoning_then_tool_call(self):
    segs = run(StreamParser(reasoning=THINK, tool_call=TOOL),
               ['<think>let me call it</think><tool_call>{"name": "f", "arguments": {}}</tool_call>'])
    self.assertEqual(joined(segs, "reasoning"), "let me call it")
    self.assertEqual([c["name"] for c in tool_calls(segs)], ["f"])
    self.assertEqual(joined(segs, "content"), "")

  def test_unclosed_tool_call_dropped_on_flush(self):
    segs = run(StreamParser(tool_call=TOOL), ['<tool_call>{"name": "f"'])
    self.assertEqual(tool_calls(segs), [])
    self.assertEqual(joined(segs, "content"), "")  # partial block never leaks as content

class TestReasoningFormat(unittest.TestCase):
  def test_detects_think_from_template(self):
    self.assertEqual(detect_reasoning_format("...{% if enable_thinking %}<think>\n</think>{% endif %}..."), ("<think>", "</think>"))
  def test_detects_think_from_special_tokens(self):
    self.assertEqual(detect_reasoning_format(None, ["<think>", "</think>", "<tool_call>"]), ("<think>", "</think>"))
  def test_defaults_to_think_when_unknown(self):
    # no metadata -> default think-tags (harmless: non-thinking models never emit the marker)
    self.assertEqual(detect_reasoning_format(None, []), DEFAULT_REASONING)

if __name__ == "__main__":
  unittest.main()
