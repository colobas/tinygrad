from __future__ import annotations
import json
from typing import Any

# A minimal incremental JSON-Schema grammar for constrained decoding (§7). It is a pure-functional
# pushdown acceptor: `initial(schema)` builds a stack, `step(stack, ch)` consumes one character returning
# the new stack (or None on a grammar violation), and `can_eof(stack)` reports whether the value is
# complete. Because the state is an immutable tuple, the token-mask builder can cheaply trial-feed every
# vocabulary token from the committed state to decide which tokens keep the output grammar-viable.
#
# Supported schema subset (documented restrictions, chosen to keep the acceptor deterministic):
#   - emits COMPACT JSON (no insignificant whitespace).
#   - object: declared `properties` are emitted in declaration order and all are treated as required;
#     keys are assumed to need no escaping. `enum` constrains to the listed JSON values.
#   - types: object, array (with `items`), string, integer, number, boolean, null; a schema with no
#     `type`/`enum` accepts any JSON value.

_DIGITS = set("0123456789")
_HEX = set("0123456789abcdefABCDEF")

def _val_frame(schema:dict|None): return ("val", schema if isinstance(schema, dict) else {})

def initial(schema:dict|None) -> tuple: return (_val_frame(schema),)

def _start_value(schema:dict, ch:str):
  # given a fresh value position and the first char, return (frames_to_push, consumed_ch) or None.
  # frames are returned top-last (so the returned tuple is appended to the stack as-is).
  if (enum := schema.get("enum")) is not None:
    cands = tuple(s for v in enum if (s := json.dumps(v, separators=(",", ":"))) and s[0] == ch)
    if not cands: return None
    return (("enum", tuple(c[1:] for c in cands)),), True
  t = schema.get("type")
  if t == "object" or (t is None and ch == "{"):
    if ch != "{": return None
    props = schema.get("properties")
    if props: return (("obj", tuple(props.items()), 0, "first"),), True  # closed object: declared props in order
    return (("objfree", "first"),), True  # free-form object: arbitrary keys -> any values
  if t == "array" or (t is None and ch == "["):
    if ch != "[": return None
    return (("arr", schema.get("items") or {}, "first"),), True
  if t == "string" or (t is None and ch == '"'):
    if ch != '"': return None
    return (("str", "normal"),), True
  if t == "boolean" or (t is None and ch in "tf"):
    if ch == "t": return (("lit", "rue"),), True
    if ch == "f": return (("lit", "alse"),), True
    return None
  if t == "null" or (t is None and ch == "n"):
    if ch != "n": return None
    return (("lit", "ull"),), True
  if t in ("integer", "number") or (t is None and (ch == "-" or ch in _DIGITS)):
    is_float = t != "integer"
    if ch == "-": return (("num", is_float, "sign"),), True
    if ch == "0": return (("num", is_float, "int0"),), True
    if ch in _DIGITS: return (("num", is_float, "int"),), True
    return None
  return None

def _num_consume(is_float:str, state:str, ch:str):
  d = ch in _DIGITS
  if state == "sign": return "int0" if ch == "0" else "int" if d else None
  if state == "int": return "int" if d else "dot" if (is_float and ch == ".") else "exp" if (is_float and ch in "eE") else None
  if state == "int0": return "dot" if (is_float and ch == ".") else "exp" if (is_float and ch in "eE") else None
  if state == "dot": return "frac" if d else None
  if state == "frac": return "frac" if d else "exp" if ch in "eE" else None
  if state == "exp": return "expsign" if ch in "+-" else "expdig" if d else None
  if state == "expsign": return "expdig" if d else None
  if state == "expdig": return "expdig" if d else None
  return None

_NUM_EOF = ("int", "int0", "frac", "expdig")  # number states that form a complete number

def step(stack:tuple, ch:str) -> tuple | None:
  # consume one character; returns the new stack or None if `ch` is not grammar-permitted here.
  while True:
    if not stack: return None  # a complete value was parsed; no more input is allowed
    top = stack[-1]
    tag = top[0]
    if tag == "lit":
      s = top[1]
      if s and s[0] == ch:
        rest = s[1:]
        return stack[:-1] + ((("lit", rest),) if rest else ())
      return None
    if tag == "val":
      r = _start_value(top[1], ch)
      if r is None: return None
      frames, _ = r
      return stack[:-1] + frames
    if tag == "enum":
      cands = tuple(c[1:] for c in top[1] if c and c[0] == ch)
      if not cands: return None  # a "" suffix (fully-matched candidate) cannot consume more chars
      return stack[:-1] + (("enum", cands),)
    if tag == "str":
      esc = top[1]
      if esc == "normal":
        if ch == '"': return stack[:-1]
        if ch == "\\": return stack[:-1] + (("str", "esc"),)
        if ord(ch) < 0x20: return None
        return stack
      if esc == "esc":
        if ch in '"\\/bfnrt': return stack[:-1] + (("str", "normal"),)
        if ch == "u": return stack[:-1] + (("str", "u0"),)
        return None
      if esc.startswith("u"):
        if ch not in _HEX: return None
        n = int(esc[1:])
        return stack[:-1] + ((("str", "normal"),) if n == 3 else (("str", f"u{n+1}"),))
      return None
    if tag == "num":
      ns = _num_consume(top[1], top[2], ch)
      if ns is not None: return stack[:-1] + (("num", top[1], ns),)
      if top[2] in _NUM_EOF: stack = stack[:-1]; continue  # number ended; reprocess ch against the parent
      return None
    if tag == "obj":
      props, i, need = top[1], top[2], top[3]
      base = stack[:-1]
      if need == "first":
        if not props: return base if ch == "}" else None  # empty object: '}' closes it

        if ch == '"':  # start key 0
          key, sub = props[0]
          return base + (("obj", props, 1, "sep"), _val_frame(sub), ("lit", key + '":'))
        return None
      if need == "sep":
        if i >= len(props): return base if ch == "}" else None  # all properties emitted -> close
        if ch == ",": return base + (("obj", props, i, "key"),)
        return None
      if need == "key":
        if ch == '"':
          key, sub = props[i]
          return base + (("obj", props, i+1, "sep"), _val_frame(sub), ("lit", key + '":'))
        return None
      return None
    if tag == "objfree":
      need, base = top[1], stack[:-1]
      pair = base + (("objfree", "sep"), _val_frame({}), ("lit", ":"), ("str", "normal"))  # key body -> ':' -> any value
      if need == "first": return base if ch == "}" else pair if ch == '"' else None
      if need == "sep": return base if ch == "}" else base + (("objfree", "key"),) if ch == "," else None
      if need == "key": return pair if ch == '"' else None
      return None
    if tag == "arr":
      item, need = top[1], top[2]
      base = stack[:-1]
      if need == "first":
        if ch == "]": return base  # empty array
        # epsilon: every non-']' char begins the first item; push a value frame and reprocess ch
        stack = base + (("arr", item, "sep"), _val_frame(item)); continue
      if need == "sep":
        if ch == "]": return base
        if ch == ",": return base + (("arr", item, "sep"), _val_frame(item))
        return None
      return None
    return None

def can_eof(stack:tuple) -> bool:
  # True when the stack represents a fully-parsed value (only complete-number frames may epsilon-pop).
  while stack:
    top = stack[-1]
    if top[0] == "num" and top[2] in _NUM_EOF: stack = stack[:-1]
    elif top[0] == "enum" and "" in top[1]: stack = stack[:-1]  # a candidate is fully matched
    else: return False
  return True

class _TrieNode:
  # one node of the vocabulary trie: child chars -> nodes, plus token ids whose text ends exactly here.
  __slots__ = ("children", "ids")
  def __init__(self): self.children: dict[str, _TrieNode] = {}; self.ids: list[int] = []

def build_trie(token_texts:dict[int, str]) -> _TrieNode:
  # one-time index of the whole vocabulary by shared character prefix. Tokens sharing a prefix share a
  # path, so the grammar only has to validate each distinct prefix once when masking (not each token).
  root = _TrieNode()
  for tid, text in token_texts.items():
    if not text: continue
    node = root
    for ch in text:
      if (nxt := node.children.get(ch)) is None: nxt = node.children[ch] = _TrieNode()
      node = nxt
    node.ids.append(tid)
  return root

class GrammarMasker:
  """Drives a JSON-Schema grammar over a model's token stream, producing a per-step additive logit bias.

  Construct with the decodable vocabulary (`token_texts`: id -> text for normal tokens), the set of
  end-of-turn token ids, and the vocab size. `mask(completion_ids)` advances the committed grammar state
  by any newly-generated tokens, then returns a length-`vocab_size` list: 0.0 for tokens whose text keeps
  the output grammar-viable (and EOS only once the value can complete), a large negative for the rest.

  Masking walks a prefix trie of the vocabulary guided by the grammar: descending into a child only when
  the grammar accepts that character means a single rejected character prunes every token under that
  prefix in one `step`. Pass a prebuilt `trie` (shared across requests for a fixed vocabulary) to skip
  the one-time build.
  """
  def __init__(self, schema:dict, token_texts:dict[int, str], eos_ids:set[int], vocab_size:int, neg:float=-1e9,
               trie:_TrieNode|None=None):
    self.stack: tuple | None = initial(schema)
    self.token_texts, self.eos_ids, self.vocab_size, self.neg = token_texts, eos_ids, vocab_size, neg
    self.trie = trie if trie is not None else build_trie(token_texts)
    self._consumed = 0

  def _feed_text(self, stack:tuple|None, text:str) -> tuple | None:
    for ch in text:
      if stack is None: return None
      stack = step(stack, ch)
    return stack

  def mask(self, completion_ids:list[int]) -> list[float]:
    for tid in completion_ids[self._consumed:]:  # fold newly-committed tokens into the grammar state
      self.stack = self._feed_text(self.stack, self.token_texts.get(tid, ""))
    self._consumed = len(completion_ids)
    bias = [self.neg] * self.vocab_size
    base = self.stack
    if base is not None:
      # DFS the trie carrying the grammar state reached along each path; reaching a node means its prefix
      # is grammar-viable, so its terminal token ids are permitted. Rejected chars prune whole subtrees.
      # `step` is memoized on (id(state), char): permissive positions (e.g. a string body) self-loop on one
      # state object, so the same transition recurs across the whole subtree and is computed once. Keying by
      # identity is safe — every state used as a key is `base` or a memo value, so it stays referenced (no
      # id reuse) for the call; grammar states embed schema dicts and aren't hashable by value.
      smemo: dict[tuple, tuple | None] = {}
      frontier = [(self.trie, base)]
      while frontier:
        node, gstate = frontier.pop()
        for tid in node.ids: bias[tid] = 0.0
        gid = id(gstate)
        for ch, child in node.children.items():
          if (ns := smemo.get(k := (gid, ch), 0)) == 0: ns = smemo[k] = step(gstate, ch)
          if ns is not None: frontier.append((child, ns))
      if can_eof(base):
        for e in self.eos_ids:
          if 0 <= e < self.vocab_size: bias[e] = 0.0
    return bias
