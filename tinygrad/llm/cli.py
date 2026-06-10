from __future__ import annotations
import sys, argparse, codecs, typing, re, unicodedata, json, uuid, time, pathlib
from tinygrad import nn
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored, Context, fetch, profile_marker
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler
from tinygrad.llm.model import Transformer

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3",
               bos_id:int|None=None, eos_id:int=0, eot_id:int|None=None):
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

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv["tokenizer.ggml.pre"],
      bos_id=kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None,
      eos_id=kv.get('tokenizer.ggml.eos_token_id', 0), eot_id=kv.get('tokenizer.ggml.eot_token_id'))

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

class StopChecker:
  """Streaming detector for OpenAI-style `stop` sequences.

  feed(text) yields the prefix that is safe to emit (i.e. text we've ruled out as part of any
  in-progress stop match). When a stop sequence completes, sets .stopped=True; the matched
  sequence itself is excluded from emitted output. flush() drains any held tail at the end.
  """
  def __init__(self, stops:list[str]):
    self.stops = [s for s in stops if s]
    self.maxlen = max((len(s) for s in self.stops), default=0)
    self.buf = ""
    self.stopped = False
  def feed(self, text:str):
    if self.stopped: return
    if not self.stops: yield text; return
    self.buf += text
    # full match anywhere in buffer wins
    earliest = -1
    for s in self.stops:
      i = self.buf.find(s)
      if i != -1 and (earliest == -1 or i < earliest): earliest = i
    if earliest != -1:
      if earliest > 0: yield self.buf[:earliest]
      self.buf = ""
      self.stopped = True
      return
    safe = len(self.buf) - (self.maxlen - 1)
    if safe > 0:
      yield self.buf[:safe]
      self.buf = self.buf[safe:]
  def flush(self):
    if self.buf and not self.stopped:
      yield self.buf
      self.buf = ""

def strip_think_blocks(text:str) -> str:
  """Remove <think>...</think> sections from prior-turn assistant content.

  Qwen3-family chat templates expect thinking traces to be ephemeral — they appear in the live
  turn's output but are stripped from history when re-encoding prompts for the next turn.
  """
  import re
  return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

class ThinkParser:
  """Streaming parser that splits model output around <think>...</think> blocks.

  Feed text incrementally via .feed(text) — yields (mode, text) tuples where mode is
  'reasoning' (inside a think block) or 'content' (outside). Tags that straddle token
  boundaries are handled by buffering up to len(tag)-1 trailing characters until disambiguated.
  Tag characters themselves are stripped from both outputs.
  """
  OPEN, CLOSE = "<think>", "</think>"
  def __init__(self, start_mode:str="content"):
    self.buf, self.mode = "", start_mode
  def feed(self, text:str):
    self.buf += text
    while True:
      target = self.CLOSE if self.mode == "reasoning" else self.OPEN
      if (idx := self.buf.find(target)) != -1:
        if idx > 0: yield (self.mode, self.buf[:idx])
        self.buf = self.buf[idx + len(target):]
        self.mode = "content" if self.mode == "reasoning" else "reasoning"
      else:
        # hold the last len(target)-1 chars in case a tag is being formed
        safe = len(self.buf) - (len(target) - 1)
        if safe > 0:
          yield (self.mode, self.buf[:safe])
          self.buf = self.buf[safe:]
        break
  def flush(self):
    if self.buf:
      yield (self.mode, self.buf)
      self.buf = ""

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
                stop:list[str]|None=None):
    model, tok = self.server.model, self.server.tok
    cache_start_pos = model.get_start_pos(ids)
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  "
               f"in:{colored(f'{cache_start_pos:5d}', 'green')} +{len(ids)-cache_start_pos:5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    yield {"choices": [{"index":0, "delta":{"role":"assistant","content":""}, "finish_reason":None}], **tmpl}
    out: list[int] = []
    finish_reason = "stop"
    st = time.perf_counter()
    dec = tok.stream_decoder()
    parser = ThinkParser()
    stopper = StopChecker(stop or [])
    def _emit(mode:str, text:str):
      if not text: return None
      field = "reasoning_content" if mode == "reasoning" else "content"
      return {"choices": [{"index":0, "delta":{field:text}, "finish_reason":None}], **tmpl}
    gen = model.generate_mtp(ids, k=self.server.mtp_k, temperature=temperature) if self.server.mtp_k and model.mtp_heads \
          else model.generate(ids, temperature=temperature)
    for next_id in gen:
      if len(out) == 0: stderr_log(f"prefill:{(len(ids)-cache_start_pos)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
      if tok.is_end(next_id): break
      out.append(next_id)
      for mode, text in parser.feed(dec(next_id)):
        if mode == "reasoning":
          # stop sequences only apply to user-visible content, not reasoning traces
          if (chunk := _emit(mode, text)) is not None: yield chunk
        else:
          for safe in stopper.feed(text):
            if (chunk := _emit(mode, safe)) is not None: yield chunk
          if stopper.stopped: break
      if stopper.stopped: break
      if max_tokens is not None and len(out) >= max_tokens:
        finish_reason = "length"
        break
    if not stopper.stopped:
      if (tail := dec()):
        for mode, text in parser.feed(tail):
          if mode == "reasoning":
            if (chunk := _emit(mode, text)) is not None: yield chunk
          else:
            for safe in stopper.feed(text):
              if (chunk := _emit(mode, safe)) is not None: yield chunk
      for mode, text in parser.flush():
        if mode == "reasoning":
          if (chunk := _emit(mode, text)) is not None: yield chunk
        else:
          for safe in stopper.feed(text):
            if (chunk := _emit(mode, safe)) is not None: yield chunk
      for safe in stopper.flush():
        if (chunk := _emit("content", safe)) is not None: yield chunk
    yield {"choices": [{"index":0, "delta":{},"finish_reason":finish_reason}], **tmpl}
    if include_usage:
      yield {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}, **tmpl}
    et = time.perf_counter()
    stderr_log(f"gen:{len(out)/(et-pt) if len(out) > 1 else 0:4.0f} tok/s  {colored('--', 'BLACK')}  "
               f"out:{len(out):5d}  {colored('--', 'BLACK')}  total:{et-st:6.2f}s\n")

  def do_POST(self):
    tok = self.server.tok
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      # determine thinking mode: enable_thinking may live at the top level or inside chat_template_kwargs
      enable_thinking = body.get("enable_thinking",
        body.get("chat_template_kwargs", {}).get("enable_thinking", True))

      # extract tokens, last assistant message is treated as prefill
      ids: list[int] = tok.prefix()
      n_msgs = len(body["messages"])
      for i, msg in enumerate(body["messages"]):
        ids += tok.role(msg["role"])
        is_last = (i == n_msgs - 1)
        # reasoning_content from prior turns is ephemeral — never re-encoded.
        # for prior assistant turns we also strip any <think>...</think> blocks left in `content`.
        content = msg["content"]
        if isinstance(content, str):
          if msg["role"] == "assistant" and not is_last: content = strip_think_blocks(content)
          ids += tok.encode(content)
        elif isinstance(content, list):
          for c in content:
            if c["type"] == "text":
              txt = c["text"]
              if msg["role"] == "assistant" and not is_last: txt = strip_think_blocks(txt)
              ids += tok.encode(txt)
            else: raise RuntimeError(f"unhandled type: {c['type']}")
        else: raise RuntimeError(f"unknown content type: {type(content)}")
        if msg["role"] == "assistant" and is_last: break
        ids += tok.end_turn()
      else:
        ids += tok.role("assistant")
        # If the client opts out of thinking, prefill the empty think block so the model skips
        # straight to user-visible content (matches the standard Qwen3 chat template behavior).
        if not enable_thinking: ids += tok.encode("<think>\n\n</think>\n\n")

      # normalize the optional `stop` field (OpenAI allows string or list)
      stop_val = body.get("stop")
      stop_list = [stop_val] if isinstance(stop_val, str) else (list(stop_val) if stop_val else None)

      # reply
      max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False),
                              max_tokens=max_tokens, temperature=float(body.get("temperature", 0.0)), stop=stop_list)
      if body.get("stream"): self.stream_json(chunks)
      else:
        content_parts, reasoning_parts, finish_reason = [], [], "stop"
        for c in chunks:
          if c["choices"]:
            delta = c["choices"][0].get("delta", {})
            if (txt := delta.get("content")): content_parts.append(txt)
            if (txt := delta.get("reasoning_content")): reasoning_parts.append(txt)
            if c["choices"][0].get("finish_reason"): finish_reason = c["choices"][0]["finish_reason"]
        msg: dict = {"role":"assistant", "content":"".join(content_parts)}
        if reasoning_parts: msg["reasoning_content"] = "".join(reasoning_parts)
        self.send_data(json.dumps({**c, "object":"chat.completion",
          "choices":[{"index":0, "message":msg, "finish_reason":finish_reason}]}).encode())
    else:
      raise RuntimeError(f"unhandled path {self.path}")

class LLMServer(TCPServerWithReuse):
  def __init__(self, server_address:tuple, model:Transformer, model_name:str, tok:SimpleTokenizer, mtp_k:int=0):
    self.model, self.model_name, self.tok, self.mtp_k = model, model_name, tok, mtp_k
    super().__init__(server_address, Handler)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", default=list(models.keys())[0], help=f"Model choice ({', '.join(models.keys())}) or path to a local GGUF file")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=8000, metavar="PORT", help="Run OpenAI compatible API (optional port, default 8000)")
  parser.add_argument("--warmup", action="store_true", help="warmup the JIT")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  parser.add_argument("--prompt", type=str, default=None, help="Benchmark prompt (default: BOS only)")
  parser.add_argument("--mtp", type=int, default=0, metavar="K",
                      help="Speculative decoding via MTP heads, drafting K tokens per main step (requires an MTP-enabled GGUF). "
                           "NOTE: currently slower than baseline on hybrid SSM models — needs kernel-layer tuning to deliver speedup.")
  args = parser.parse_args()

  # load the model
  model, kv = Transformer.from_gguf(fetch(models.get(args.model, args.model)), args.max_context)
  model_name = kv.get('general.name') or kv.get('general.basename') or args.model
  file_sizes = [y.nbytes() for y in UOp.sink(*[x.uop for x in nn.state.get_parameters(model)]).toposort() if y.op is Ops.BUFFER]
  print(f"using model \"{model_name}\" with {sum(file_sizes):,} bytes and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params")

  # get tokenizer
  tok = SimpleTokenizer.from_gguf_kv(kv)

  # warmup the JIT (compile + capture once, hit cache thereafter)
  if args.warmup or args.serve:
    with Context(DEBUG=max(DEBUG.value, 1)):
      for _ in range(2):
        if args.mtp and model.mtp_heads:
          # warm the MTP-specific JITs (verify T=K+1, mtp_main, mtp_draft) at the requested K
          list(zip(range(2), model.generate_mtp([0], k=args.mtp)))
        else:
          list(zip(range(2), model.generate([0])))

  if args.mtp and not model.mtp_heads:
    print(f"warning: --mtp {args.mtp} requested but model has no MTP heads; falling back to standard decode")
    args.mtp = 0
  if args.mtp and model.has_recurrent_block:
    print(f"warning: --mtp on a hybrid SSM/attention model currently runs SLOWER than baseline. "
          f"The MTP algorithm is correct, but the T>{1} verify forward generates new kernel shapes "
          f"that tinygrad's JIT scheduler has not tuned as well as the baseline T=1 path. "
          f"Closing the gap is a kernel-layer task. Use --mtp 0 for fastest inference.")

  # start server
  if args.serve: LLMServer(('', args.serve), model, model_name, tok, mtp_k=args.mtp).serve_forever()

  # do benchmark
  if args.benchmark is not None:
    toks = (tok.prefix() + tok.encode(args.prompt)) if args.prompt else [tok.bos_id or 0]
    gen = (model.generate_mtp(toks, k=args.mtp) if args.mtp else model.generate(toks))
    for i in range(args.benchmark):
      profile_marker(f"decode @ {i}")
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+\
                  tok.decode(toks).replace("\n", "\\n")): next(gen)
    exit(0)

  # interactive chat
  ids: list[int] = tok.prefix()
  while 1:
    try:
      ids += tok.role("user") + tok.encode(input('>>> ')) + tok.end_turn() + tok.role("assistant")
    except EOFError:
      break
    dec = tok.stream_decoder()
    chat_gen = model.generate_mtp(ids, k=args.mtp) if args.mtp else model.generate(ids)
    for next_id in chat_gen:
      sys.stdout.write(dec(next_id) if not tok.is_end(next_id) else dec() + "\n\n")
      sys.stdout.flush()
      if tok.is_end(next_id): break

if __name__ == "__main__": main()
