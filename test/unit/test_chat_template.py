import unittest, pathlib, datetime, json
from tinygrad.llm.jinja import render, JinjaError
from tinygrad.llm.cli import strip_reasoning

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
LLAMA = (FIXTURES / "llama3.2.jinja").read_text()

# a small ChatML-style template that exercises for/filters/if and the trim_blocks/lstrip_blocks semantics
CHATML = (
  "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content | trim }}<|im_end|>\n{% endfor %}"
  "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

class TestJinjaEngine(unittest.TestCase):
  def r(self, t, **kw): return render(t, **kw)
  def test_output_and_text(self): self.assertEqual(self.r("a{{ x }}b", x="Z"), "aZb")
  def test_if_elif_else(self):
    t = "{% if n == 1 %}one{% elif n == 2 %}two{% else %}many{% endif %}"
    self.assertEqual([self.r(t, n=n) for n in (1, 2, 3)], ["one", "two", "many"])
  def test_for_and_loop(self):
    self.assertEqual(self.r("{% for x in xs %}{{ loop.index0 }}:{{ x }}{% if not loop.last %},{% endif %}{% endfor %}",
                            xs=["a", "b", "c"]), "0:a,1:b,2:c")
  def test_for_else(self): self.assertEqual(self.r("{% for x in xs %}{{x}}{% else %}empty{% endfor %}", xs=[]), "empty")
  def test_set_and_concat(self): self.assertEqual(self.r("{% set g = 'hi ' + name %}{{ g }}", name="bob"), "hi bob")
  def test_namespace_mutation(self):
    t = "{% set ns = namespace(c=0) %}{% for x in xs %}{% set ns.c = ns.c + 1 %}{% endfor %}{{ ns.c }}"
    self.assertEqual(self.r(t, xs=[1, 2, 3, 4]), "4")
  def test_slice_and_index(self):
    self.assertEqual(self.r("{{ xs[1:] | join(',') }}", xs=["a", "b", "c"]), "b,c")
    self.assertEqual(self.r("{{ m['k'] }}", m={"k": "v"}), "v")
  def test_attr_vs_item(self): self.assertEqual(self.r("{{ m.k }}-{{ m['k'] }}", m={"k": "v"}), "v-v")
  def test_tests(self):
    self.assertEqual(self.r("{% if x is defined %}D{% else %}U{% endif %}", x=1), "D")
    self.assertEqual(self.r("{% if x is defined %}D{% else %}U{% endif %}"), "U")
    self.assertEqual(self.r("{% if x is none %}N{% endif %}", x=None), "N")
    self.assertEqual(self.r("{% if x is not none %}NN{% endif %}", x=5), "NN")
    self.assertEqual(self.r("{% if m is mapping %}M{% endif %}", m={}), "M")
    self.assertEqual(self.r("{% if s is string %}S{% endif %}", s="x"), "S")
  def test_ternary_and_bool(self):
    self.assertEqual(self.r("{{ 'yes' if a and not b else 'no' }}", a=True, b=False), "yes")
    self.assertEqual(self.r("{{ 'yes' if a or b else 'no' }}", a=False, b=False), "no")
  def test_in_operator(self):
    self.assertEqual(self.r("{% if 'k' in m %}Y{% endif %}", m={"k": 1}), "Y")
    self.assertEqual(self.r("{% if 'z' not in m %}N{% endif %}", m={"k": 1}), "N")
  def test_tojson(self):
    self.assertEqual(self.r("{{ d | tojson }}", d={"a": 1}), '{"a": 1}')
    self.assertEqual(self.r("{{ d | tojson(indent=2) }}", d={"a": 1}), '{\n  "a": 1\n}')
  def test_filters(self):
    self.assertEqual(self.r("{{ s | trim }}", s="  hi  "), "hi")
    self.assertEqual(self.r("{{ xs | length }}", xs=[1, 2, 3]), "3")
    self.assertEqual(self.r("{{ s | replace('a','b') }}", s="aa"), "bb")
    self.assertEqual(self.r("{{ x | default('fallback') }}"), "fallback")
  def test_whitespace_trim_lstrip_blocks(self):
    # block tags on their own indented lines leave no stray whitespace (lstrip_blocks + trim_blocks)
    t = "x\n  {% if true %}\n  y\n  {% endif %}\nz"
    self.assertEqual(self.r(t), "x\n  y\nz")
  def test_dash_whitespace_control(self):
    self.assertEqual(self.r("a  {{- x -}}  b", x="Z"), "aZb")
  def test_raise_exception_is_jinja_error(self):
    with self.assertRaises(JinjaError): self.r("{{ raise_exception('boom') }}")
  def test_unsupported_block_raises(self):
    with self.assertRaises(JinjaError): self.r("{% macro f() %}{% endmacro %}")
  def test_chatml_template(self):
    out = self.r(CHATML, messages=[{"role": "user", "content": " hi "}], add_generation_prompt=True)
    self.assertEqual(out, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")

class TestLlamaTemplate(unittest.TestCase):
  def render(self, **kw):
    return render(LLAMA, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>", **kw)
  def test_basic_conversation(self):
    date = datetime.datetime.now().strftime("%d %b %Y")
    out = self.render(messages=[{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello"}],
                      add_generation_prompt=True)
    expected = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"Cutting Knowledge Date: December 2023\nToday Date: {date}\n\n"
                f"You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    self.assertEqual(out, expected)
  def test_tools_rendered_into_prompt(self):
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    out = self.render(messages=[{"role": "user", "content": "weather?"}], tools=tools, add_generation_prompt=True)
    self.assertIn("Environment: ipython", out)
    self.assertIn('"name": "get_weather"', out)  # tool schema (tojson indent=4) is in the prompt

class TestStripReasoning(unittest.TestCase):
  def test_drops_reasoning_content_field(self):
    msgs = [{"role": "assistant", "content": "hi", "reasoning_content": "secret"}, {"role": "user", "content": "q"}]
    out = strip_reasoning(msgs)
    self.assertNotIn("reasoning_content", out[0])
    self.assertEqual(out[0]["content"], "hi")
  def test_strips_think_from_prior_assistant_turns(self):
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>hidden</think>visible"},
            {"role": "user", "content": "q2"}]
    out = strip_reasoning(msgs)
    self.assertEqual(out[1]["content"], "visible")
  def test_keeps_final_assistant_content_for_prefill(self):
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<think>x</think>prefill"}]
    out = strip_reasoning(msgs)
    self.assertEqual(out[-1]["content"], "<think>x</think>prefill")  # final (prefill) turn untouched

class TestTemplatePath(unittest.TestCase):
  """The server's prompt builder: template path, prefill trimming, and preset fallback."""
  def _self(self, **tokattrs):
    from types import SimpleNamespace
    tok = SimpleNamespace(chat_template=CHATML, bos_token="", eos_token="", encode=lambda s: list(s.encode()))
    tok.__dict__.update(tokattrs)
    return SimpleNamespace(server=SimpleNamespace(tok=tok))
  def _text(self, fake, body):
    from tinygrad.llm.cli import Handler
    ids = Handler._template_ids(fake, body)
    return None if ids is None else bytes(ids).decode()
  def test_template_path_renders_prompt(self):
    text = self._text(self._self(), {"messages": [{"role": "user", "content": "hi"}]})
    self.assertEqual(text, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
  def test_prefill_trims_after_final_content(self):
    body = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "partial"}]}
    text = self._text(self._self(), body)
    self.assertTrue(text.endswith("partial"))  # nothing appended after the prefill content
    self.assertNotIn("<|im_start|>assistant\n<|im_end|>", text)
  def test_no_template_falls_back_to_preset(self):
    from tinygrad.llm.cli import Handler
    self.assertIsNone(Handler._template_ids(self._self(chat_template=None), {"messages": [{"role": "user", "content": "hi"}]}))
  def test_unsupported_template_falls_back_to_preset(self):
    from tinygrad.llm.cli import Handler
    fake = self._self(chat_template="{% macro f() %}{% endmacro %}")
    self.assertIsNone(Handler._template_ids(fake, {"messages": [{"role": "user", "content": "hi"}]}))

WEATHER = {"type": "function", "function": {"name": "get_weather",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}

class TestPresetToolInjection(unittest.TestCase):
  """When the template is unsupported, the preset fallback must still put tool signatures in the prompt."""
  def _preset_text(self, body):
    from types import SimpleNamespace
    from tinygrad.llm.cli import Handler
    # fake tokenizer whose ids are just the utf-8 bytes of every emitted string, so we can read the prompt back
    tok = SimpleNamespace(prefix=lambda: [], end_turn=lambda: [], encode=lambda s: list(s.encode()),
                          role=lambda r: list(f"<{r}>".encode()))
    return bytes(Handler._preset_ids(SimpleNamespace(server=SimpleNamespace(tok=tok)), body)).decode()

  def test_tools_injected_into_preset_prompt(self):
    text = self._preset_text({"messages": [{"role": "user", "content": "weather in SF?"}], "tools": [WEATHER]})
    self.assertIn("<tools>", text)
    self.assertIn("get_weather", text)                 # the signature reached the prompt
    self.assertIn("<tool_call>", text)                 # and the model is told the call format our parser reads
    self.assertIn("weather in SF?", text)              # original user message preserved

  def test_tools_folded_into_existing_system_message(self):
    text = self._preset_text({"messages": [{"role": "system", "content": "You are helpful."},
                                           {"role": "user", "content": "hi"}], "tools": [WEATHER]})
    self.assertEqual(text.count("<system>"), 1)        # folded into the existing system turn, not a second one
    self.assertIn("You are helpful.", text)
    self.assertIn("get_weather", text)

  def test_no_tools_no_injection(self):
    text = self._preset_text({"messages": [{"role": "user", "content": "hi"}]})
    self.assertNotIn("<tools>", text)

  def test_tools_system_text_serializes_each_tool(self):
    from tinygrad.llm.cli import _tools_system_text
    other = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}
    txt = _tools_system_text([WEATHER, other])
    self.assertIn("get_weather", txt)
    self.assertIn("lookup", txt)

@unittest.skipUnless(__import__("importlib").util.find_spec("jinja2"), "jinja2 not installed (golden reference)")
class TestGoldenVsJinja2(unittest.TestCase):
  """Golden: rendered bytes match jinja2 — the engine transformers.apply_chat_template uses — configured
  with the same trim_blocks/lstrip_blocks settings and globals. Skipped when jinja2 is unavailable."""
  def reference(self, template, **ctx):
    import jinja2
    def raise_exception(m=""): raise RuntimeError(m)
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, undefined=jinja2.Undefined)
    env.filters["tojson"] = lambda v, indent=None: json.dumps(v, indent=indent, ensure_ascii=False)
    env.globals.update(raise_exception=raise_exception,
                       strftime_now=lambda fmt: datetime.datetime.now().strftime(fmt))
    return env.from_string(template).render(**ctx)
  def assertMatches(self, template, **ctx):
    self.assertEqual(render(template, **ctx), self.reference(template, **ctx))
  def test_chatml_matches(self):
    self.assertMatches(CHATML, messages=[{"role": "system", "content": "s"}, {"role": "user", "content": " hi "}],
                       add_generation_prompt=True)
  def test_llama_no_tools_matches(self):
    self.assertMatches(LLAMA, messages=[{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}],
                       add_generation_prompt=True, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>")
  def test_llama_with_tools_matches(self):
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    self.assertMatches(LLAMA, messages=[{"role": "user", "content": "go"}], tools=tools,
                       add_generation_prompt=True, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>")

if __name__ == "__main__":
  unittest.main()
