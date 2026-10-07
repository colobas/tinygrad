from __future__ import annotations
import json, math, pathlib
from typing import Callable, cast
from dataclasses import dataclass
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.llm.gguf import ggml_data_to_tensor, ggml_nbytes
from tinygrad.llm.model import Transformer

# Cloudflare Clef: a Qwen3.8-27B backbone plus a "joint schema head". one forward pass over [state, schema] gives one logit per allowed option of
# every question, there is no decoding. reference: https://huggingface.co/Cloudflare/clef (joint_schema_model.py)

SYSTEM_PROMPT = ("Read the complete state and schema. Decide every field jointly. Each answer "
                 "must be exactly one of that field's allowed options.")
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}

def render(value) -> str:
  return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

def question_options(question:dict) -> list[tuple[str, object]]:
  if (qt:=str(question["type"])) == "noul":
    criteria = {"true": "The proposition is true or the answer is yes.", "false": "The proposition is false or the answer is no."}
    criteria.update(question.get("criteria") or {})
    return [(key, criteria[key]) for key in ("true", "false")]
  if qt == "choice": return sorted((str(key), value) for key, value in question["criteria"].items())
  return [(str(i), value) for i, value in enumerate(question["criteria"])]

@dataclass(frozen=True)
class EncodedQuestion:
  question_id: str
  question_type: int
  question_span: tuple[int, int]
  option_spans: tuple[tuple[int, int], ...]
  option_ids: tuple[str, ...]

@dataclass(frozen=True)
class EncodedRecord:
  input_ids: tuple[int, ...]
  questions: tuple[EncodedQuestion, ...]

def encode_record(tok:SimpleTokenizer, record:dict, max_length:int=16384) -> EncodedRecord:
  if record.get("images") or record.get("videos"): raise NotImplementedError("clef images/videos are not supported")
  schema_ids = tok.encode("\n\nSCHEMA FIELDS:\n")
  questions: list[EncodedQuestion] = []
  for qi, (qid, question) in enumerate(record["questions"].items()):
    schema_ids += tok.encode(f"\nFIELD {qi+1}\nID: {qid}\nTYPE: {question['type']}\nINSTRUCTION: ")
    q_start = len(schema_ids)
    instructions = question.get("instructions")
    if instructions is None or instructions == "": instructions = str(qid)
    schema_ids += tok.encode(render(instructions))
    q_end = len(schema_ids)
    schema_ids += tok.encode("\nALLOWED OPTIONS:\n")
    spans, option_ids = [], []
    for oi, (option_id, description) in enumerate(question_options(question)):
      schema_ids += tok.encode(f"OPTION {oi+1}: ")
      o_start = len(schema_ids)
      semantics: dict[str, object] = {"option_id": option_id}
      if description is not None: semantics["description"] = description
      schema_ids += tok.encode(render(semantics))
      spans.append((o_start, len(schema_ids)))
      option_ids.append(option_id)
      schema_ids += tok.encode("\n")
    schema_ids += tok.encode("END FIELD\n")
    questions.append(EncodedQuestion(str(qid), QUESTION_TYPES[str(question["type"])], (q_start, q_end), tuple(spans), tuple(option_ids)))
  prefix_ids = tok.encode(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
  suffix_ids = tok.encode("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
  state_ids = tok.encode(render(record["state"]))
  if (fixed:=len(prefix_ids) + len(schema_ids) + len(suffix_ids)) > max_length:
    raise ValueError(f"schema requires {fixed} tokens before state; maximum is {max_length}")
  state_ids = state_ids[:max_length - fixed]
  off = len(prefix_ids) + len(state_ids)
  questions = [EncodedQuestion(q.question_id, q.question_type, (q.question_span[0]+off, q.question_span[1]+off),
                               tuple((s+off, e+off) for s, e in q.option_spans), q.option_ids) for q in questions]
  if not questions: raise ValueError("record produced no questions")
  return EncodedRecord(tuple(prefix_ids + state_ids + schema_ids + suffix_ids), tuple(questions))

def systemone_answer(question:dict, probs:dict[str, float]) -> dict:
  if question["type"] == "noul": return {"type": "noul", "noul": round(probs["true"], 4)}
  if question["type"] == "choice":
    options = [str(o) for o in question["criteria"]]
    choice = max(options, key=probs.__getitem__)
    return {"type": "choice", "choice": choice, "confidence": round(probs[choice], 4), "probabilities": {o: round(probs[o], 4) for o in options}}
  levels = [str(i) for i in range(len(question["criteria"]))]
  return {"type": "score", "score": round(sum(i * probs[l] for i, l in enumerate(levels)), 4), "confidence": round(max(probs[l] for l in levels), 4),
          "legend": dict(zip(levels, question["criteria"])), "probabilities": {l: round(probs[l], 4) for l in levels}}

def _softmax(x:list[float]) -> list[float]:
  m = max(x)
  e = [math.exp(v - m) for v in x]
  return [v / sum(e) for v in e]
def _gelu(x:Tensor) -> Tensor: return x * 0.5 * (1 + (x * (1 / math.sqrt(2))).erf())  # torch's exact gelu, not tinygrad's tanh approximation
def _normalize(x:Tensor, eps:float=1e-12) -> Tensor: return x / x.square().sum(-1, keepdim=True).sqrt().maximum(eps)

class JointSchemaHead:
  """port of JointSchemaHead from the reference. weights come from the gguf dec.blk.* / decision.* tensors, computed in float32"""
  def __init__(self, w:dict[str, Tensor], heads:int, routing_layers:int, layers:int, eps:float):
    self.w, self.heads, self.routing_layers, self.layers, self.eps = w, heads, routing_layers, layers, eps
    scales = cast(list[float], w["decision.scales"].tolist())  # prior_logit_scale, joint_logit_scale, residual_gate (parameter declaration order)
    self.prior_scale, self.joint_scale = math.exp(min(float(scales[0]), math.log(100.0))), math.exp(min(float(scales[1]), math.log(100.0)))
    self.gate = 1 / (1 + math.exp(-float(scales[2])))
    self._jits: dict[tuple, Callable] = {}

  def _ln(self, x:Tensor, name:str) -> Tensor: return x.layernorm(eps=self.eps) * self.w[name+".weight"] + self.w[name+".bias"]
  def _lin(self, x:Tensor, name:str) -> Tensor: return x @ self.w[name+".weight"].T + self.w[name+".bias"]
  def _mha(self, q_in:Tensor, kv_in:Tensor, pre:str, mask:Tensor) -> Tensor:  # nn.MultiheadAttention; q_in (Lq, W), kv_in (Lk, W), mask: True = keep
    def split(x:Tensor) -> Tensor: return x.reshape(x.shape[0], self.heads, -1).transpose(0, 1).unsqueeze(0)
    q, k, v = split(self._lin(q_in, pre+"_q")), split(self._lin(kv_in, pre+"_k")), split(self._lin(kv_in, pre+"_v"))
    o = q.scaled_dot_product_attention(k, v, attn_mask=mask)[0].transpose(0, 1).reshape(q_in.shape[0], -1)
    return self._lin(o, pre+"_o")
  def _ffn(self, x:Tensor, pre:str) -> Tensor: return self._lin(_gelu(self._lin(x, pre+".ffn_up")), pre+".ffn_down")

  def forward(self, hidden:Tensor, mem_mask:Tensor, self_mask:Tensor, glob_oh:Tensor, q_span:Tensor, o_span:Tensor, lex_m:Tensor,
              lex_rows:Tensor, owner_oh:Tensor, type_oh:Tensor) -> Tensor:
    """static-shape head (so it can be JIT'd). P = padded tokens, Q = padded questions, N = padded options. padded keys are masked, padded
    options belong to no question. returns one logit per (padded) option"""
    w = self.w
    hn = self._ln(hidden.float(), "decision.hidden_norm")
    memory = hn @ w["decision.proj_memory.weight"].T                                    # (P, W)
    glob = (glob_oh @ hn)[0]                                                           # last real token
    q_vec, ctx = q_span @ hn, o_span @ hn                                              # (Q, H), (N, H): mean over spans
    lex = lex_m @ lex_rows                                                             # (N, H): mean output embedding of the option tokens
    routed = ctx @ w["decision.proj_option_context.weight"].T + lex @ w["decision.proj_option_lexical.weight"].T \
      + owner_oh @ (q_vec @ w["decision.proj_option_question.weight"].T)
    for i in range(self.routing_layers):
      p = f"dec.blk.{i}"
      routed = routed + self._mha(self._ln(routed, p+".cross_attn_norm"), self._ln(memory, p+".cross_attn_norm_kv"), p+".cross_attn", mem_mask)
      routed = routed + self._ffn(self._ln(routed, p+".ffn_norm"), p)
    base = q_vec @ w["decision.proj_question.weight"].T                                 # (Q, W)
    scores = (base @ routed.T) / math.sqrt(routed.shape[-1]) + (owner_oh.T - 1) * 1e9  # per-question softmax over its own options
    summaries = scores.softmax(-1) @ routed
    fields = base + self._ln(summaries, "decision.option_summary_norm") + (glob @ w["decision.proj_global.weight"].T).unsqueeze(0) \
      + type_oh @ w["token_types.weight"]
    for i in range(self.routing_layers, self.routing_layers + self.layers):
      p = f"dec.blk.{i}"
      fields = fields + self._mha(self._ln(fields, p+".attn_norm"), self._ln(fields, p+".attn_norm"), p+".attn", self_mask)
      fields = fields + self._mha(self._ln(fields, p+".cross_attn_norm"), memory, p+".cross_attn", mem_mask)
      fields = fields + self._ffn(self._ln(fields, p+".ffn_norm"), p)
    fields = self._ln(fields, "decision.field_norm")
    prior = self.prior_scale * (_normalize(lex) * (owner_oh @ _normalize(q_vec + glob))).sum(-1)
    opts, rf = self._ln(routed, "decision.option_norm"), owner_oh @ fields
    cosine = (rf * opts).sum(-1) / (rf.square().sum(-1).sqrt().maximum(1e-8) * opts.square().sum(-1).sqrt().maximum(1e-8))
    feats = Tensor.cat(rf, opts, rf * opts, (rf - opts).abs(), dim=-1)
    residual = (_gelu(self._lin(feats, "decision.scorer")) @ w["decision.scorer_out.weight"]) + w["decision.scorer_out.bias"]
    return prior + self.gate * (self.joint_scale * cosine + residual)

  def __call__(self, hidden:Tensor, n_tokens:int, output_rows:dict[int, Tensor], rec:EncodedRecord) -> list[list[float]]:
    """hidden: (P>=n_tokens, hidden_size) final-norm backbone states, rows past n_tokens are padding. returns per-question logits"""
    P, qs = hidden.shape[0], rec.questions
    Q, all_spans = -(-len(qs) // 4) * 4, [s for q in qs for s in q.option_spans]
    N = -(-len(all_spans) // 8) * 8
    def span_mean(spans:list[tuple[int, int]], rows:int) -> Tensor:
      m = [[1.0 / (e - s) if s <= j < e else 0.0 for j in range(P)] for s, e in spans] + [[0.0] * P] * (rows - len(spans))
      return Tensor(m, dtype=dtypes.float32)
    uniq = sorted({t for s, e in all_spans for t in rec.input_ids[s:e]})
    U, col = -(-len(uniq) // 64) * 64, {t: i for i, t in enumerate(uniq)}
    lex_m = [[0.0] * U for _ in range(N)]
    for i, (s, e) in enumerate(all_spans):
      for t in rec.input_ids[s:e]: lex_m[i][col[t]] += 1.0 / (e - s)
    lex_rows = Tensor.stack(*[output_rows[t] for t in uniq]).float().pad(((0, U - len(uniq)), (0, 0)))
    counts, owner = [len(q.option_spans) for q in qs], [i for i, q in enumerate(qs) for _ in q.option_spans]
    args = (hidden, Tensor([[[[j < n_tokens for j in range(P)]]]]), Tensor([[[[j < len(qs) for j in range(Q)]] * Q]]),
            Tensor([[1.0 if j == n_tokens - 1 else 0.0 for j in range(P)]], dtype=dtypes.float32), span_mean([q.question_span for q in qs], Q),
            span_mean(all_spans, N), Tensor(lex_m, dtype=dtypes.float32), lex_rows,
            Tensor([[1.0 if owner[i] == j else 0.0 for j in range(Q)] if i < len(owner) else [0.0] * Q for i in range(N)], dtype=dtypes.float32),
            Tensor([[1.0 if i < len(qs) and qs[i].question_type == j else 0.0 for j in range(3)] for i in range(Q)], dtype=dtypes.float32))
    if (jit:=self._jits.get(key:=(P, Q, N, U))) is None: jit = self._jits[key] = TinyJit(self.forward)
    flat, off, ret = cast(list[float], jit(*args).tolist()), 0, []
    for c in counts:
      ret.append(flat[off:off+c])
      off += c
    return ret

class Clef:
  def __init__(self, gguf:str|pathlib.Path, max_context:int=16384):
    self.model, kv = Transformer.from_gguf(gguf, max_context)
    assert kv["general.architecture"] == "clef", "not a clef gguf"
    self.kv, self.tok, self.max_context = kv, SimpleTokenizer.from_gguf_kv(kv), max_context
    ex = self.model.extras
    self.out_raw, self.out_type = ex.pop("output.raw"), int(ex.pop("output.ggml_type").item())
    self.dim = kv["clef.embedding_length"]
    w = {k: v.float().contiguous() for k, v in ex.items()}
    Tensor.realize(*w.values())
    self.head = JointSchemaHead(w, kv["clef.decision.head_count"], kv["clef.decision.routing_block_count"], kv["clef.decision.block_count"],
                                kv["clef.attention.layer_norm_epsilon"])
    self.chunk = getenv("CLEF_CHUNK", 64)
    self._hidden_jit = TinyJit(self.model.forward_hidden)

  def hidden_states(self, ids:list[int]) -> Tensor:
    """final-norm hidden states of every token, (T padded up to a multiple of the chunk, dim). full static chunks go through one JIT.
    the last chunk is zero-padded: padding follows the real tokens, so the causal outputs of the real ones are unaffected"""
    T, C = len(ids), self.chunk
    assert T <= self.max_context
    padded = ids + [0] * (-T % C)
    t = Tensor(padded, dtype="int32").reshape(1, -1).contiguous()
    v_sp, outs = UOp.variable("start_pos", 0, self.max_context - 1), []
    for s in range(0, len(padded), C):
      sp = v_sp.bind(s)
      outs.append(self._hidden_jit(t[:, sp:sp+C].contiguous(), sp)[0])
    return Tensor.cat(*outs, dim=0)

  def output_rows(self, token_ids:list[int]) -> dict[int, Tensor]:
    ids = sorted(set(token_ids))
    rows = Tensor.stack(*[self.out_raw[i] for i in ids]).reshape(-1)  # int-indexed rows are views of the packed bytes
    deq = ggml_data_to_tensor(rows, len(ids) * self.dim, self.out_type).reshape(len(ids), self.dim).float().realize()
    assert ggml_nbytes(self.dim, self.out_type) * len(ids) == rows.shape[0]
    return {t: deq[i] for i, t in enumerate(ids)}

  def logits(self, rec:EncodedRecord) -> list[list[float]]:
    hidden = self.hidden_states(list(rec.input_ids))
    rows = self.output_rows([t for q in rec.questions for s, e in q.option_spans for t in rec.input_ids[s:e]])
    return self.head(hidden, len(rec.input_ids), rows, rec)

  def systemone(self, request:dict) -> dict:
    questions = request.get("questions")
    if not isinstance(request.get("model"), str) or "state" not in request: raise ValueError("model and state are required")
    if not isinstance(questions, dict) or not questions: raise ValueError("at least one question is required")
    for qid, q in questions.items():
      if q.get("type") not in QUESTION_TYPES: raise ValueError(f"{qid}: type must be noul, choice, or score")
      if q["type"] != "noul" and not q.get("criteria"): raise ValueError(f"{qid}: criteria must not be empty")
    rec = encode_record(self.tok, request, self.max_context)
    answers = {q.question_id: systemone_answer(questions[q.question_id], dict(zip(q.option_ids, _softmax(lg))))
               for q, lg in zip(rec.questions, self.logits(rec))}
    return {"model": request["model"], "answers": answers, "usage": {"input_tokens": len(rec.input_ids), "output_tokens": 0}}

def main():
  import argparse
  from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
  from tinygrad.helpers import fetch
  parser = argparse.ArgumentParser(description="serve a Clef decision model on /v1/systemone")
  parser.add_argument("model", help="clef gguf path or url")
  parser.add_argument("--serve", type=int, default=8080, metavar="PORT")
  parser.add_argument("--max_context", type=int, default=16384)
  args = parser.parse_args()
  clef = Clef(fetch(args.model) if args.model.startswith("http") else args.model, args.max_context)
  class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
      if self.path != "/v1/systemone": return self.send_error(404)
      try: code, body = 200, clef.systemone(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
      except (ValueError, KeyError, NotImplementedError) as e: code, body = 400, {"error": str(e)}
      data = json.dumps(body).encode()
      self.send_response(code)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(data)))
      self.end_headers()
      self.wfile.write(data)
  ThreadingHTTPServer(("", args.serve), Handler).serve_forever()

if __name__ == "__main__": main()
