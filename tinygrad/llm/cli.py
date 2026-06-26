from __future__ import annotations
import sys, argparse, codecs, typing, re, unicodedata, json, uuid, time, pathlib
from tinygrad import nn
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored, Context, fetch, profile_marker, getenv
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler
from tinygrad.llm.model import Transformer
from tinygrad.llm.parser import StreamParser, detect_reasoning_format, DEFAULT_REASONING
from tinygrad.llm.grammar import GrammarMasker, build_trie
from tinygrad.llm.jinja import render, JinjaError

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
def strip_reasoning(messages:list[dict]) -> list[dict]:
  # ephemeral prior-turn reasoning: never re-encode earlier reasoning. Drop `reasoning_content` everywhere
  # and strip `<think>...</think>` blocks from the content of prior (non-final) assistant turns.
  out = []
  for i, m in enumerate(messages):
    m = {k: v for k, v in m.items() if k != "reasoning_content"}
    if m.get("role") == "assistant" and i != len(messages)-1 and isinstance(m.get("content"), str):
      m["content"] = _THINK_RE.sub("", m["content"])
    out.append(m)
  return out

# reasoning_effort -> thinking-token budget (5.2). A single default mapping today; the dict is the seam
# for per-family overrides (a family with a tighter context would key its own budgets here).
REASONING_EFFORT_BUDGETS = {"low": 256, "medium": 1024, "high": 4096}

def _reasoning_markers(tok) -> tuple[str, str]:
  # the model's configured reasoning markers; defensively defaults to think-tags when metadata is absent.
  fmt = getattr(tok, "reasoning_format", DEFAULT_REASONING)
  return fmt if isinstance(fmt, tuple) and len(fmt) == 2 else DEFAULT_REASONING

def _resolve_reasoning(body:dict, tok) -> tuple[tuple[str,str]|None, int|None]:
  # decide (reasoning markers, thinking-token budget) for a request. enable_thinking (top-level or in
  # chat_template_kwargs, 5.4) disables the split entirely; reasoning_effort/thinking_budget set the
  # host-side budget (5.2). Returns (None, None) when thinking is disabled.
  ckw = body.get("chat_template_kwargs") or {}
  if not body.get("enable_thinking", ckw.get("enable_thinking", True)): return None, None
  budget = body.get("thinking_budget", ckw.get("thinking_budget"))
  if budget is None and (effort := body.get("reasoning_effort")) is not None:
    if effort not in REASONING_EFFORT_BUDGETS: raise ValueError(f"invalid reasoning_effort '{effort}'")
    budget = REASONING_EFFORT_BUDGETS[effort]
  if budget is not None and (not isinstance(budget, int) or isinstance(budget, bool) or budget < 0):
    raise ValueError("thinking_budget must be a non-negative integer")
  return _reasoning_markers(tok), budget

JSON_MODE_RETRIES = 2  # attempts = retries + 1; configurable ceiling for json_object validation (6.2)

def _guide_json(messages:list[dict]) -> list[dict]:
  # guide the model to emit JSON (6.1): fold a JSON-only instruction into the system message (or prepend one).
  instr = "You must respond with a single valid JSON object and nothing else."
  msgs = [dict(m) for m in messages]
  for m in msgs:
    if m.get("role") == "system" and isinstance(m.get("content"), str):
      m["content"] = (m["content"] + "\n\n" + instr).strip()
      return msgs
  return [{"role":"system", "content":instr}, *msgs]

def _valid_json(s:str|None) -> bool:
  if not s: return False
  try: return isinstance(json.loads(s), (dict, list))
  except (json.JSONDecodeError, TypeError): return False

def _resolve_response_format(body:dict) -> tuple[str|None, dict|None]:
  # returns (mode, schema): mode is "json_object" (prompt-guided + validate/retry), "json_schema"
  # (grammar-constrained decoding, schema returned), or None (unconstrained). Raises ValueError
  # (-> HTTP 400) on a malformed shape.
  rf = body.get("response_format")
  if rf is None: return None, None
  if not isinstance(rf, dict) or not isinstance(rf.get("type"), str): raise ValueError("response_format must be {type: ...}")
  if rf["type"] == "json_object":
    body["messages"] = _guide_json(body.get("messages") or [])
    return "json_object", None
  if rf["type"] == "json_schema":
    js = rf.get("json_schema")
    schema = js.get("schema") if isinstance(js, dict) else None
    if not isinstance(schema, dict): raise ValueError("json_schema requires a 'json_schema.schema' object")
    return "json_schema", schema
  if rf["type"] == "text": return None, None
  raise ValueError(f"unsupported response_format type '{rf['type']}'")

_TRIE_CACHE: dict[int, tuple] = {}  # vocabulary trie is fixed per model; build once, reuse every request
def _grammar_mask_fn(tok, schema:dict):
  # build the per-step logit-bias callback that constrains decoding to JSON conforming to `schema` (§7).
  texts, eos_ids, vocab_size = tok.grammar_vocab()
  if (ent := _TRIE_CACHE.get(id(texts))) is None:
    ent = _TRIE_CACHE[id(texts)] = (texts, build_trie(texts))  # keep a ref to texts so its id can't be reused
  return GrammarMasker(schema, texts, eos_ids, vocab_size, trie=ent[1]).mask

def _resolve_tools(body:dict) -> tuple[bool, str|None]:
  # validate the `tools`/`tool_choice` request shape and apply `tool_choice` semantics. Mutates
  # body["tools"] to the effective set rendered into the prompt (None to suppress). Returns
  # (enable_tool_parsing, tool_choice_name). Raises ValueError (-> HTTP 400) on a malformed shape.
  tools = body.get("tools")
  if tools is not None:
    if not isinstance(tools, list): raise ValueError("'tools' must be an array")
    for t in tools:
      if not isinstance(t, dict) or t.get("type") != "function" or not isinstance(t.get("function"), dict) \
          or not isinstance(t["function"].get("name"), str): raise ValueError("each tool must be {type:'function', function:{name:...}}")
  choice = body.get("tool_choice", "auto" if tools else "none")
  name = None
  if isinstance(choice, str):
    if choice not in ("auto", "none", "required"): raise ValueError(f"invalid tool_choice '{choice}'")
  elif isinstance(choice, dict) and choice.get("type") == "function" and isinstance(choice.get("function"), dict) \
      and isinstance(choice["function"].get("name"), str):
    name = choice["function"]["name"]
  else: raise ValueError("tool_choice must be 'auto'/'none'/'required' or {type:'function', function:{name:...}}")
  if not tools or choice == "none":  # nothing to offer / explicitly suppressed
    body["tools"] = None
    return False, None
  if name is not None:  # named choice: only render the chosen tool into the prompt, and restrict parsing to it
    body["tools"] = [t for t in tools if t["function"]["name"] == name] or tools
  return True, name

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3",
               bos_id:int|None=None, eos_id:int=0, eot_id:int|None=None,
               chat_template:str|None=None, bos_token:str="", eos_token:str=""):
    preset = {"qwen35":"qwen2","qwen35moe":"qwen2"}.get(preset, preset)
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo","kimi-k2","tekken","glm4"):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # 0x323b0 is one past the max codepoint in unicode categories L/N/Z (0x323af is max L)
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset
    self.bos_id, self.eos_id, self.eot_id = bos_id, eos_id, eot_id
    # chat-template metadata (3.1): the model's embedded Jinja template plus the literal bos/eos token
    # strings it interpolates. `chat_template` is None when the model ships no template (preset fallback).
    self.chat_template, self.bos_token, self.eos_token = chat_template, bos_token, eos_token
    # per-model reasoning format (5.1): the (open, close) marker pair this model uses for chain-of-thought,
    # selected from metadata (template + special tokens). Consumed by the server's StreamParser.
    self.reasoning_format = detect_reasoning_format(chat_template, special_tokens.keys())

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    toks = kv["tokenizer.ggml.tokens"]
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(toks))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    bos_id, eos_id = kv.get('tokenizer.ggml.bos_token_id'), kv.get('tokenizer.ggml.eos_token_id', 0)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv["tokenizer.ggml.pre"],
      bos_id=bos_id if kv.get('tokenizer.ggml.add_bos_token', True) else None,
      eos_id=eos_id, eot_id=kv.get('tokenizer.ggml.eot_token_id'),
      chat_template=kv.get('tokenizer.chat_template'),
      bos_token=toks[bos_id] if bos_id is not None else "", eos_token=toks[eos_id] if eos_id is not None else "")

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def grammar_vocab(self) -> tuple[dict[int, str], set[int], int]:
    # for constrained decoding (§7): decodable normal-token texts (id -> str), end-of-turn ids, and vocab
    # size. Special tokens and byte-fragment tokens that aren't valid standalone UTF-8 are excluded — the
    # grammar acceptor operates on characters, and JSON output is text. Cached: the vocabulary is fixed.
    if (cached := getattr(self, "_grammar_vocab_cache", None)) is not None: return cached
    special = set(self._special_tokens.values())
    texts: dict[int, str] = {}
    for tid, b in self._tok2bytes.items():
      if tid in special: continue
      try: texts[tid] = b.decode("utf-8")
      except UnicodeDecodeError: continue
    self._grammar_vocab_cache = (texts, {i for i in (self.eos_id, self.eot_id) if i is not None}, max(self._tok2bytes) + 1)
    return self._grammar_vocab_cache

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def stream_decoder(self) -> typing.Callable[..., str]:
    dec = codecs.getincrementaldecoder('utf-8')('replace')
    def _decode(tid:int|None=None) -> str: return dec.decode(self._tok2bytes[tid]) if tid is not None else dec.decode(b'', final=True)
    return _decode
  def role(self, role:str):
    if self.preset == 'olmo': return self.encode("<|" + role + "|>\n")  # OLMoE Instruct format
    if self.preset == 'kimi-k2': return self.encode("<|im_" + role + "|>" + role + "<|im_middle|>")
    if self.preset == 'qwen2': return self.encode("<|im_start|>" + role + "\n")
    if self.preset == 'glm4': return self.encode("<|" + role + "|>")
    if self.preset == 'tekken':
      if role == 'user': return self.encode("[INST]")
      if role == 'assistant': return []
      raise ValueError(f"Unsupported role '{role}' for tokenizer preset '{self.preset}'")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def end_turn(self):
    if self.preset == 'olmo': return self.encode("\n")
    if self.preset == 'kimi-k2': return [self.eos_id]
    if self.preset == 'qwen2': return [self.eos_id] + self.encode("\n")
    if self.preset == 'glm4': return []
    if self.preset == 'tekken': return self.encode("[/INST]")
    return [self.eos_id]
  def prefix(self) -> list[int]:
    return ([] if self.bos_id is None else [self.bos_id]) + (self.encode("<sop>") if self.preset == 'glm4' else [])
  def is_end(self, token_id:int) -> bool: return token_id in (self.eos_id, self.eot_id)

models = {
  "llama3.2:1b": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
  "llama3.2:1b-q4": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "llama3.2:3b": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf",
  "llama3.2:3b-f16": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf",
  "llama3.1:8b": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
  "qwen3:0.6b": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
  "qwen3:1.7b": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
  "qwen3:8b": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
  "qwen3:30b-a3b": "https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf",
  "qwen3.5:0.8b": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf",
  "qwen3.5:4b": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
  "qwen3.5:9b": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
  "qwen3.5:27b": "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q4_K_M.gguf",
  "qwen3.5:35b-a3b": "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
  "moonlight": "https://huggingface.co/gabriellarson/Moonlight-16B-A3B-Instruct-GGUF/resolve/main/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf",
  "glm-4.7-flash": "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf",
}

# *** simple OpenAI API compatible server with web interface on http://localhost:8000/ ***

class Handler(HTTPRequestHandler):
  server: LLMServer
  def log_request(self, code='-', size='-'): pass
  def do_GET(self):
    if self.path == "/v1/models": self.send_data(json.dumps({"object":"list","data":[{"id":self.server.model_name,"object":"model"}]}).encode())
    else: self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False, max_tokens:int|None=None, temperature:float=0.0,
                stop:list[str]|None=None, reasoning:tuple[str,str]|None=None, tool_call:tuple[str,str]|None=None,
                tool_choice_name:str|None=None, reasoning_budget:int|None=None, mask_fn=None):
    model, tok = self.server.model, self.server.tok
    cache_start_pos = model.get_start_pos(ids)
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  "
               f"in:{colored(f'{cache_start_pos:5d}', 'green')} +{len(ids)-cache_start_pos:5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    yield {"choices": [{"index":0, "delta":{"role":"assistant","content":""}, "finish_reason":None}], **tmpl}
    out: list[int] = []
    finish_reason = "stop"
    tool_idx = 0  # number of tool calls emitted so far (also the OpenAI tool_calls index)
    st = time.perf_counter()
    dec = tok.stream_decoder()
    parser = StreamParser(reasoning=reasoning, stop=stop, tool_call=tool_call)
    def deltas(segs):
      nonlocal tool_idx
      for seg in segs:
        if seg.kind == "stop": return True
        if seg.kind == "tool_call":
          call = json.loads(seg.text)
          if tool_choice_name is not None and call["name"] != tool_choice_name: continue  # named choice restricts
          # stream the function name + id once, then the arguments as one fragment (block is buffered until its close)
          tc = {"index":tool_idx, "id":f"call_{uuid.uuid4().hex[:24]}", "type":"function", "function":{"name":call["name"], "arguments":""}}
          yield {"choices": [{"index":0, "delta":{"tool_calls":[tc]}, "finish_reason":None}], **tmpl}
          yield {"choices": [{"index":0, "delta":{"tool_calls":[{"index":tool_idx, "function":{"arguments":call["arguments"]}}]},
                             "finish_reason":None}], **tmpl}
          tool_idx += 1
          continue
        key = "reasoning_content" if seg.kind == "reasoning" else "content"
        if seg.text: yield {"choices": [{"index":0, "delta":{key:seg.text}, "finish_reason":None}], **tmpl}
      return False
    # generation runs in an outer restart loop so the reasoning-token budget can be enforced host-side
    # (5.3): when the budget is hit mid-`<think>`, we inject the model's reasoning-close sequence into the
    # context and resume generation, forcing the transition to visible content. No kernel-graph change —
    # gen_ids carries the prompt + already-generated tokens (generate() appends), so the KV cache is reused.
    pt, gen_ids, reasoning_count, injected, done = st, ids, 0, False, False
    while not done:
      done = True
      for next_id in model.generate(gen_ids, temperature=temperature, mask_fn=mask_fn):
        if len(out) == 0: stderr_log(f"prefill:{(len(ids)-cache_start_pos)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
        if tok.is_end(next_id): break
        out.append(next_id)
        stop_hit = yield from deltas(parser.feed(dec(next_id)))
        if stop_hit: break
        if reasoning_budget is not None and not injected and reasoning is not None and parser.state == "reasoning":
          reasoning_count += 1
          if reasoning_count >= reasoning_budget:  # budget reached: inject close marker, resume as content
            gen_ids = gen_ids + tok.encode(reasoning[1])
            yield from deltas(parser.feed(reasoning[1]))  # transition the parser out of reasoning
            injected, done = True, False
            break
        if max_tokens is not None and len(out) >= max_tokens:
          finish_reason = "length"
          break
    yield from deltas(parser.feed(dec()))
    yield from deltas(parser.flush())
    if tool_idx > 0: finish_reason = "tool_calls"  # tool calls take precedence over the default stop
    yield {"choices": [{"index":0, "delta":{},"finish_reason":finish_reason}], **tmpl}
    if include_usage:
      yield {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}, **tmpl}
    et = time.perf_counter()
    stderr_log(f"gen:{len(out)/(et-pt) if len(out) > 1 else 0:4.0f} tok/s  {colored('--', 'BLACK')}  "
               f"out:{len(out):5d}  {colored('--', 'BLACK')}  total:{et-st:6.2f}s\n")

  def _preset_ids(self, body) -> list[int]:
    # fallback prompt assembly from hardcoded per-preset role strings (last assistant message = prefill)
    tok = self.server.tok
    ids: list[int] = tok.prefix()
    for i, msg in enumerate(body["messages"]):
      ids += tok.role(msg["role"])
      content = msg["content"]
      if isinstance(content, str): ids += tok.encode(content)
      elif isinstance(content, list):
        for c in content:
          if c["type"] == "text": ids += tok.encode(c["text"])
          else: raise RuntimeError(f"unhandled type: {c['type']}")
      else: raise RuntimeError(f"unknown content type: {type(content)}")
      if msg["role"] == "assistant" and i == len(body["messages"]) - 1: break
      ids += tok.end_turn()
    else: ids += tok.role("assistant")
    return ids

  def _template_ids(self, body) -> list[int]|None:
    # render the model's Jinja chat template; returns None (-> preset fallback) when there is no usable template
    tok = self.server.tok
    if not tok.chat_template: return None
    messages = strip_reasoning(body["messages"])
    prefill = bool(messages) and messages[-1].get("role") == "assistant"
    kwargs = dict(body.get("chat_template_kwargs") or {})
    if "enable_thinking" in body: kwargs.setdefault("enable_thinking", body["enable_thinking"])
    try:
      text = render(tok.chat_template, messages=messages, tools=(body.get("tools") or None),
                    add_generation_prompt=not prefill, bos_token=tok.bos_token, eos_token=tok.eos_token, **kwargs)
    except JinjaError as e:
      if DEBUG >= 1: stderr_log(f"chat template render failed, using preset: {e}\n")
      return None
    if prefill and isinstance(c := messages[-1].get("content"), str) and (c := c.strip()) and (idx := text.rfind(c)) != -1:
      text = text[:idx+len(c)]  # continue_final_message: trim everything the template added after the prefill content
    return tok.encode(text)

  def build_ids(self, body) -> list[int]:
    ids = self._template_ids(body)
    return ids if ids is not None else self._preset_ids(body)

  def _collect(self, chunks):
    # drain a run_model chunk stream into (content, reasoning, tool_calls, finish_reason, last_chunk),
    # reassembling the streamed name-once + arguments-fragment tool-call deltas.
    content, thinking, tool_calls, finish_reason, last = [], [], [], "stop", None
    for c in chunks:
      last = c
      if not c["choices"]: continue
      delta = c["choices"][0].get("delta", {})
      if delta.get("content"): content.append(delta["content"])
      if delta.get("reasoning_content"): thinking.append(delta["reasoning_content"])
      for tc in delta.get("tool_calls", []):
        if "id" in tc: tool_calls.append({"id":tc["id"], "type":"function", "function":{"name":tc["function"]["name"], "arguments":""}})
        else: tool_calls[tc["index"]]["function"]["arguments"] += tc["function"]["arguments"]
      if c["choices"][0].get("finish_reason"): finish_reason = c["choices"][0]["finish_reason"]
    return "".join(content), "".join(thinking), tool_calls, finish_reason, last

  def _send_message(self, content, reasoning_str, tool_calls, finish_reason, last):
    message: dict[str, typing.Any] = {"role":"assistant", "content":content or None}
    if reasoning_str: message["reasoning_content"] = reasoning_str
    if tool_calls: message["tool_calls"] = tool_calls
    self.send_data(json.dumps({**last, "object":"chat.completion",
      "choices":[{"index":0, "message":message, "finish_reason":finish_reason}]}).encode())

  def _synth_stream(self, content, reasoning_str, finish_reason, last):
    # re-emit a buffered (json-mode) result as a stream: role, optional reasoning, content, finish, usage.
    tmpl = {k: last[k] for k in ("id", "object", "created", "model") if k in last}
    yield {"choices": [{"index":0, "delta":{"role":"assistant","content":""}, "finish_reason":None}], **tmpl}
    if reasoning_str: yield {"choices": [{"index":0, "delta":{"reasoning_content":reasoning_str}, "finish_reason":None}], **tmpl}
    if content: yield {"choices": [{"index":0, "delta":{"content":content}, "finish_reason":None}], **tmpl}
    yield {"choices": [{"index":0, "delta":{}, "finish_reason":finish_reason}], **tmpl}
    if last.get("usage"): yield {"choices": [], "usage":last["usage"], **tmpl}

  def do_POST(self):
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      try:
        enable_tools, tool_choice_name = _resolve_tools(body)  # validate shape + honor tool_choice (mutates body["tools"])
        reasoning, reasoning_budget = _resolve_reasoning(body, self.server.tok)  # markers + host-side think budget
        rf_mode, schema = _resolve_response_format(body)  # json_object: guide+retry; json_schema: constrained decode
      except ValueError as e:
        self.send_data(json.dumps({"error": {"message": str(e), "type": "invalid_request_error"}}).encode(), status_code=400)
        return

      max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
      stop = body.get("stop")
      if isinstance(stop, str): stop = [stop]
      tool_call = ("<tool_call>", "</tool_call>") if enable_tools else None  # Qwen/Hermes delimiter (see scoping task 1.1)
      # json_schema: a fresh stateful grammar masker per request constrains decoding to a conforming value
      mask_fn = _grammar_mask_fn(self.server.tok, schema) if rf_mode == "json_schema" else None
      base_temp = float(body.get("temperature", 0.0))
      def make_chunks(temperature, include_usage):
        # build the prompt from the model's chat template (tools rendered in); fall back to preset roles
        return self.run_model(self.build_ids(body), body["model"], include_usage, max_tokens=max_tokens, temperature=temperature,
                              stop=stop, reasoning=reasoning, tool_call=tool_call, tool_choice_name=tool_choice_name,
                              reasoning_budget=reasoning_budget, mask_fn=mask_fn)

      if rf_mode == "json_object":
        # buffer + validate + retry (6.2). Greedy first; later attempts vary temperature so they can differ.
        content = reasoning_str = ""; tool_calls = []; last = None
        for attempt in range(JSON_MODE_RETRIES + 1):
          temp = base_temp if attempt == 0 else max(base_temp, 0.7)
          content, reasoning_str, tool_calls, _, last = self._collect(make_chunks(temp, True))
          if _valid_json(content): break
        if not _valid_json(content):
          self.send_data(json.dumps({"error": {"message": f"model did not produce valid JSON in {JSON_MODE_RETRIES+1} attempts",
            "type": "json_validation_error"}}).encode(), status_code=502)
          return
        if body.get("stream"): self.stream_json(self._synth_stream(content, reasoning_str, "stop", last))
        else: self._send_message(content, reasoning_str, tool_calls, "stop", last)
        return

      include_usage = not body.get("stream") or (body.get("stream_options") or {}).get("include_usage", False)
      chunks = make_chunks(base_temp, include_usage)
      if body.get("stream"): self.stream_json(chunks)
      else: self._send_message(*self._collect(chunks))
    else:
      raise RuntimeError(f"unhandled path {self.path}")

class LLMServer(TCPServerWithReuse):
  def __init__(self, server_address:tuple, model:Transformer, model_name:str, tok:SimpleTokenizer):
    self.model, self.model_name, self.tok = model, model_name, tok
    super().__init__(server_address, Handler)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", default=list(models.keys())[0], help=f"Model choice ({', '.join(models.keys())}) or path to a local GGUF file")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=8000, metavar="PORT",
                      help="Run OpenAI-compatible API (chat templates, tools/tool_choice, reasoning_effort, "
                           "response_format json_object/json_schema; optional port, default 8000)")
  parser.add_argument("--warmup", action="store_true", help="warmup the JIT")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  args = parser.parse_args()

  # load the model
  model, kv = Transformer.from_gguf(fetch(models.get(args.model, args.model)), args.max_context)
  model_name = kv.get('general.name') or kv.get('general.basename') or args.model
  file_sizes = [y.nbytes() for y in UOp.sink(*[x.uop for x in nn.state.get_parameters(model)]).toposort() if y.op is Ops.BUFFER]
  print(f"using model \"{model_name}\" with {sum(file_sizes):,} bytes and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params")

  # get tokenizer
  tok = SimpleTokenizer.from_gguf_kv(kv)

  # warmup the JIT
  if args.warmup or args.serve:
    # run 2 tokens through the model twice to capture the JIT before serving
    with Context(DEBUG=max(DEBUG.value, 1)):
      for _ in range(2): list(zip(range(2), model.generate([0])))

  # start server
  if args.serve: LLMServer(('', args.serve), model, model_name, tok).serve_forever()

  # do benchmark
  if args.benchmark is not None:
    gen = model.generate(toks:=[tok.bos_id or 0])
    for i in range(args.benchmark):
      profile_marker(f"decode @ {i}")
      GlobalCounters.reset()
      if (log:=getenv("BENCHMARK_LOG", "")): from extra.bench_log import WallTimeEvent, BenchEvent
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+\
                  tok.decode(toks).replace("\n", "\\n")):
        if log:
          with WallTimeEvent(BenchEvent.STEP): next(gen)
        else: next(gen)
    exit(0)

  # interactive chat
  ids: list[int] = tok.prefix()
  while 1:
    try:
      ids += tok.role("user") + tok.encode(input('>>> ')) + tok.end_turn() + tok.role("assistant")
    except EOFError:
      break
    dec = tok.stream_decoder()
    for next_id in model.generate(ids):
      sys.stdout.write(dec(next_id) if not tok.is_end(next_id) else dec() + "\n\n")
      sys.stdout.flush()
      if tok.is_end(next_id): break

if __name__ == "__main__": main()
