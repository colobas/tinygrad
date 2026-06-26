from __future__ import annotations
import re, json, datetime
from typing import Any

# A minimal Jinja engine — a minja-style subset (à la llama.cpp) sufficient to render the chat templates
# embedded in the catalog's GGUFs, without taking a dependency on jinja2/transformers. It implements the
# whitespace semantics transformers uses for `apply_chat_template` (trim_blocks=True, lstrip_blocks=True
# plus the explicit `-` markers) so rendered bytes match the reference. Constructs outside this subset
# raise `JinjaError`, which the caller treats as "template unsupported" and falls back to preset role
# formatting. This keeps the engine small while staying byte-faithful for what it does support.

class JinjaError(Exception): pass

class Undefined:
  # Jinja undefined: falsy, compares unequal to everything, only meaningful with the `defined`/`none` tests.
  __slots__ = ()
  def __bool__(self): return False
  def __eq__(self, o): return isinstance(o, Undefined)
  def __ne__(self, o): return not isinstance(o, Undefined)
  def __hash__(self): return 0
  def __iter__(self): return iter(())
  def __len__(self): return 0
UNDEFINED = Undefined()

class Namespace:
  def __init__(self, **kw): self.__dict__.update(kw)

# ---- lexer: split template into text / {{ var }} / {% block %} / {# comment #} tokens ----
class Tok:
  __slots__ = ("kind", "val", "lstrip", "rstrip")
  def __init__(self, kind, val, lstrip=False, rstrip=False): self.kind, self.val, self.lstrip, self.rstrip = kind, val, lstrip, rstrip

_OPEN = {"{{": ("var", "}}"), "{%": ("block", "%}"), "{#": ("comment", "#}")}
def _lex(src:str) -> list[Tok]:
  toks: list[Tok] = []
  i, n, text_start = 0, len(src), 0
  while i < n:
    if src[i] == "{" and i+1 < n and src[i+1] in "{%#":
      if i > text_start: toks.append(Tok("text", src[text_start:i]))
      kind, close = _OPEN["{" + src[i+1]]
      j = i + 2
      lstrip = src[j:j+1] == "-"
      if lstrip: j += 1
      if (end := src.find(close, j)) == -1: raise JinjaError(f"unclosed {kind} tag")
      rstrip = src[end-1] == "-"
      toks.append(Tok(kind, src[j:end-1 if rstrip else end].strip(), lstrip, rstrip))
      i = end + len(close)
      text_start = i
    else: i += 1
  if text_start < n: toks.append(Tok("text", src[text_start:]))
  return toks

def _apply_whitespace(toks:list[Tok]) -> list[Tok]:
  # trim_blocks + lstrip_blocks (block/comment tags only) plus explicit `-` markers (all tags).
  for idx, t in enumerate(toks):
    if t.kind == "text": continue
    prev, nxt = toks[idx-1] if idx > 0 else None, toks[idx+1] if idx+1 < len(toks) else None
    if prev is not None and prev.kind == "text":
      if t.lstrip: prev.val = prev.val.rstrip()
      elif t.kind in ("block", "comment"):  # lstrip_blocks: drop spaces/tabs from line start up to the tag
        prev.val = re.sub(r"[ \t]*\Z", "", prev.val) if "\n" not in prev.val else re.sub(r"(?<=\n)[ \t]*\Z", "", prev.val)
    if nxt is not None and nxt.kind == "text":
      if t.rstrip: nxt.val = nxt.val.lstrip()
      elif t.kind in ("block", "comment") and nxt.val[:1] == "\n": nxt.val = nxt.val[1:]  # trim_blocks: drop first newline after tag
      elif t.kind in ("block", "comment") and nxt.val[:2] == "\r\n": nxt.val = nxt.val[2:]
  return toks

# ---- expression tokenizer ----
_E_RE = re.compile(r"""\s*(?:
    (?P<str>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')
  | (?P<num>\d+\.\d+|\d+)
  | (?P<op><=|>=|==|!=|//|\*\*|[-+*/%()\[\]{}.,:|~]|<|>|=)
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  )""", re.VERBOSE)
_KEYWORDS = {"and", "or", "not", "in", "is", "if", "else", "true", "false", "none", "True", "False", "None"}

def _etok(s:str) -> list[tuple[str, str]]:
  out, pos = [], 0
  while pos < len(s):
    if not s[pos:].strip(): break
    if not (m := _E_RE.match(s, pos)): raise JinjaError(f"bad expression token at {s[pos:]!r}")
    pos, val = m.end(), m.group(m.lastgroup)
    if m.lastgroup == "name" and val in _KEYWORDS: out.append(("kw", val))
    else: out.append((m.lastgroup, val))
  out.append(("end", ""))
  return out

# ---- expression parser (recursive descent, Pratt-ish) into evaluable AST tuples ----
class _EP:
  def __init__(self, toks): self.toks, self.i = toks, 0
  def peek(self): return self.toks[self.i]
  def next(self): self.i += 1; return self.toks[self.i-1]
  def expect(self, val):
    if self.peek()[1] != val: raise JinjaError(f"expected {val!r}, got {self.peek()[1]!r}")
    return self.next()

  def parse(self):
    node = self.ternary()
    if self.peek()[0] != "end": raise JinjaError(f"trailing expression tokens: {self.peek()[1]!r}")
    return node
  def ternary(self):
    node = self.or_()
    if self.peek() == ("kw", "if"):
      self.next(); cond = self.or_(); self.expect("else"); other = self.ternary()
      return ("cond", cond, node, other)
    return node
  def or_(self):
    node = self.and_()
    while self.peek() == ("kw", "or"): self.next(); node = ("or", node, self.and_())
    return node
  def and_(self):
    node = self.not_()
    while self.peek() == ("kw", "and"): self.next(); node = ("and", node, self.not_())
    return node
  def not_(self):
    if self.peek() == ("kw", "not"): self.next(); return ("not", self.not_())
    return self.compare()
  def compare(self):
    node = self.add()
    while True:
      k, v = self.peek()
      if k == "op" and v in ("==", "!=", "<", ">", "<=", ">="): self.next(); node = ("cmp", v, node, self.add())
      elif v == "in": self.next(); node = ("in", node, self.add())
      elif self.peek() == ("kw", "not") and self.toks[self.i+1] == ("kw", "in"):
        self.next(); self.next(); node = ("not", ("in", node, self.add()))
      elif v == "is": node = self.is_test(node)
      else: break
    return node
  def is_test(self, node):
    self.next()  # 'is'
    negate = self.peek() == ("kw", "not")
    if negate: self.next()
    name = self.next()[1]  # test name (defined/none/mapping/iterable/string/...)
    if name in ("none", "None"): name = "none"
    t = ("test", name, node)
    return ("not", t) if negate else t
  def add(self):
    node = self.mul()
    while self.peek()[0] == "op" and self.peek()[1] in ("+", "-"):
      op = self.next()[1]; node = ("bin", op, node, self.mul())
    return node
  def mul(self):
    node = self.filter_()
    while self.peek()[0] == "op" and self.peek()[1] in ("*", "/", "//", "%"):
      op = self.next()[1]; node = ("bin", op, node, self.filter_())
    return node
  def filter_(self):
    node = self.unary()
    while self.peek() == ("op", "|"):
      self.next(); fname = self.next()[1]; args, kwargs = [], {}
      if self.peek() == ("op", "("): args, kwargs = self.call_args()
      node = ("filter", fname, node, args, kwargs)
    return node
  def unary(self):
    if self.peek() == ("op", "-"): self.next(); return ("neg", self.unary())
    return self.postfix()
  def postfix(self):
    node = self.atom()
    while True:
      k, v = self.peek()
      if v == ".": self.next(); node = ("attr", node, self.next()[1])
      elif v == "[": self.next(); node = self.subscript(node)
      elif v == "(": args, kwargs = self.call_args(); node = ("call", node, args, kwargs)
      else: break
    return node
  def subscript(self, node):
    # index or slice a[i] / a[i:j]
    lo = None if self.peek()[1] == ":" else self.ternary()
    if self.peek()[1] == ":":
      self.next(); hi = None if self.peek()[1] == "]" else self.ternary()
      self.expect("]"); return ("slice", node, lo, hi)
    self.expect("]"); return ("item", node, lo)
  def call_args(self):
    self.expect("("); args, kwargs = [], {}
    while self.peek()[1] != ")":
      if self.peek()[0] == "name" and self.toks[self.i+1] == ("op", "="):
        key = self.next()[1]; self.next(); kwargs[key] = self.ternary()
      else: args.append(self.ternary())
      if self.peek()[1] == ",": self.next()
    self.expect(")"); return args, kwargs
  def atom(self):
    k, v = self.next()
    if k == "str": return ("const", _unquote(v))
    if k == "num": return ("const", float(v) if "." in v else int(v))
    if k == "kw" and v in ("true", "True"): return ("const", True)
    if k == "kw" and v in ("false", "False"): return ("const", False)
    if k == "kw" and v in ("none", "None"): return ("const", None)
    if v == "(": node = self.ternary(); self.expect(")"); return node
    if v == "[":
      items = []
      while self.peek()[1] != "]":
        items.append(self.ternary())
        if self.peek()[1] == ",": self.next()
      self.expect("]"); return ("list", items)
    if v == "{":
      pairs = []
      while self.peek()[1] != "}":
        key = self.ternary(); self.expect(":"); pairs.append((key, self.ternary()))
        if self.peek()[1] == ",": self.next()
      self.expect("}"); return ("dict", pairs)
    if k == "name": return ("name", v)
    raise JinjaError(f"unexpected token {v!r}")

def _unquote(s:str) -> str: return json.loads('"' + s[1:-1].replace('\\"','"').replace('"','\\"') + '"') if s[0] == '"' else \
  s[1:-1].encode().decode("unicode_escape")

def _parse_expr(s:str): return _EP(_etok(s)).parse()

# ---- statement parser: turn the token stream into a block AST ----
def _parse_template(toks:list[Tok]):
  body, stack = [], []  # stack of (kind, node, body) for open blocks
  cur = body
  def push(node, child): stack.append((cur, node)); return child
  for t in toks:
    if t.kind == "comment": continue
    if t.kind == "text":
      if t.val: cur.append(("text", t.val))
      continue
    if t.kind == "var": cur.append(("out", _parse_expr(t.val))); continue
    # block statement
    parts = t.val.split(None, 1)
    kw = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    if kw == "if":
      node = ["if", [(_parse_expr(rest), (nb := []))]]; cur.append(node); stack.append((cur, node)); cur = nb
    elif kw == "elif":
      cur_parent, node = stack[-1]
      if node[0] != "if": raise JinjaError("elif outside if")
      node[1].append((_parse_expr(rest), (nb := []))); cur = nb
    elif kw == "else":
      cur_parent, node = stack[-1]
      if node[0] == "if": node[1].append((("const", True), (nb := []))); cur = nb
      elif node[0] == "for": node.append((nb := [])); cur = nb  # for-else
      else: raise JinjaError("else outside if/for")
    elif kw == "endif":
      cur_parent, node = stack.pop(); cur = cur_parent
    elif kw == "for":
      m = re.match(r"(.+?)\s+in\s+(.+)$", rest)
      if not m: raise JinjaError(f"bad for: {rest!r}")
      targets = [x.strip() for x in m.group(1).split(",")]
      node = ["for", targets, _parse_expr(m.group(2)), (nb := [])]; cur.append(node); stack.append((cur, node)); cur = nb
    elif kw == "endfor":
      cur_parent, node = stack.pop(); cur = cur_parent
    elif kw == "set":
      if "=" not in rest: raise JinjaError(f"unsupported set: {rest!r}")
      tgt, expr = rest.split("=", 1)
      cur.append(("set", tgt.strip(), _parse_expr(expr.strip())))
    elif kw in ("macro", "endmacro", "filter", "endfilter", "block", "endblock", "raw", "endraw", "generation", "endgeneration"):
      raise JinjaError(f"unsupported block: {kw}")
    else: raise JinjaError(f"unknown block: {kw}")
  if stack: raise JinjaError("unclosed block")
  return body

# ---- evaluator ----
def _getattr(obj, name):
  if isinstance(obj, dict): return obj.get(name, UNDEFINED)
  if isinstance(obj, Namespace): return obj.__dict__.get(name, UNDEFINED)
  if isinstance(obj, Undefined): return UNDEFINED
  # bound string/list methods used by templates
  if hasattr(obj, name) and not name.startswith("_"): return getattr(obj, name)
  return UNDEFINED

class _Ctx:
  def __init__(self, scopes): self.scopes = scopes
  def get(self, name):
    for s in reversed(self.scopes):
      if name in s: return s[name]
    return UNDEFINED
  def set(self, name, val): self.scopes[-1][name] = val
  def set_global(self, name, val):
    for s in reversed(self.scopes):
      if name in s: s[name] = val; return
    self.scopes[0][name] = val

class _Renderer:
  def __init__(self): self.out: list[str] = []
  def render(self, body, ctx:_Ctx):
    for node in body: self.exec(node, ctx)
  def exec(self, node, ctx):
    op = node[0]
    if op == "text": self.out.append(node[1])
    elif op == "out":
      v = self.eval(node[1], ctx)
      self.out.append(v if isinstance(v, str) else _stringify(v))
    elif op == "if":
      for cond, blk in node[1]:
        if self.eval(cond, ctx): self.render(blk, ctx); break
    elif op == "for": self.exec_for(node, ctx)
    elif op == "set": self.exec_set(node[1], node[2], ctx)
  def exec_set(self, target, expr, ctx):
    val = self.eval(expr, ctx)
    if "." in target or "[" in target:  # assignment to namespace attr / item
      base, _, attr = target.rpartition(".")
      obj = self.eval(_parse_expr(base), ctx)
      if isinstance(obj, Namespace): obj.__dict__[attr] = val
      elif isinstance(obj, dict): obj[attr] = val
      else: raise JinjaError(f"cannot assign to {target}")
    else: ctx.set_global(target, val)
  def exec_for(self, node, ctx):
    _, targets, it_expr, *rest = node
    body = rest[0]
    else_blk = rest[1] if len(rest) > 1 else None
    seq = self.eval(it_expr, ctx)
    if isinstance(seq, Undefined) or seq is None: seq = []
    if isinstance(seq, dict): seq = list(seq.keys())
    seq = list(seq)
    if not seq and else_blk is not None: self.render(else_blk, ctx); return
    ctx.scopes.append({})
    try:
      for i, item in enumerate(seq):
        if len(targets) == 1: ctx.scopes[-1][targets[0]] = item
        else:
          for t, v in zip(targets, item): ctx.scopes[-1][t] = v
        ctx.scopes[-1]["loop"] = Namespace(index=i+1, index0=i, first=(i == 0), last=(i == len(seq)-1), length=len(seq))
        self.render(body, ctx)
    finally: ctx.scopes.pop()

  def eval(self, node, ctx):
    op = node[0]
    if op == "const": return node[1]
    if op == "name":
      n = node[1]
      if n in _GLOBALS: return _GLOBALS[n]
      return ctx.get(n)
    if op == "out": return self.eval(node[1], ctx)
    if op == "attr": return _getattr(self.eval(node[1], ctx), node[2])
    if op == "item":
      base, key = self.eval(node[1], ctx), self.eval(node[2], ctx)
      try:
        if isinstance(base, dict): return base.get(key, UNDEFINED)
        return base[key]
      except (KeyError, IndexError, TypeError): return UNDEFINED
    if op == "slice":
      base = self.eval(node[1], ctx)
      lo = self.eval(node[2], ctx) if node[2] else None
      hi = self.eval(node[3], ctx) if node[3] else None
      return base[lo:hi]
    if op == "list": return [self.eval(x, ctx) for x in node[1]]
    if op == "dict": return {self.eval(k, ctx): self.eval(v, ctx) for k, v in node[1]}
    if op == "neg": return -self.eval(node[1], ctx)
    if op == "not": return not self.eval(node[1], ctx)
    if op == "and": return self.eval(node[1], ctx) and self.eval(node[2], ctx)
    if op == "or": return self.eval(node[1], ctx) or self.eval(node[2], ctx)
    if op == "cond": return self.eval(node[2], ctx) if self.eval(node[1], ctx) else self.eval(node[3], ctx)
    if op == "cmp": return _compare(node[1], self.eval(node[2], ctx), self.eval(node[3], ctx))
    if op == "in":
      a, b = self.eval(node[1], ctx), self.eval(node[2], ctx)
      try: return a in b
      except TypeError: return False
    if op == "bin": return _binop(node[1], self.eval(node[2], ctx), self.eval(node[3], ctx))
    if op == "test": return _test(node[1], self.eval(node[2], ctx))
    if op == "filter": return _filter(node[1], self.eval(node[2], ctx), [self.eval(a, ctx) for a in node[3]],
                                       {k: self.eval(v, ctx) for k, v in node[4].items()})
    if op == "call":
      fn = self.eval(node[1], ctx)
      args = [self.eval(a, ctx) for a in node[2]]
      kwargs = {k: self.eval(v, ctx) for k, v in node[3].items()}
      if not callable(fn): raise JinjaError(f"not callable: {node[1]}")
      return fn(*args, **kwargs)
    raise JinjaError(f"cannot eval {op}")

def _stringify(v):
  if v is None or isinstance(v, Undefined): return ""
  if isinstance(v, bool): return "true" if v else "false"
  if isinstance(v, float) and v.is_integer(): return str(int(v))
  return str(v)
def _compare(op, a, b):
  return {"==": a == b, "!=": a != b, "<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[op]
def _binop(op, a, b):
  if op == "+":
    if isinstance(a, str) or isinstance(b, str): return _stringify(a) + _stringify(b) if not (isinstance(a, str) and isinstance(b, str)) else a + b
    if isinstance(a, list) and isinstance(b, list): return a + b
    return a + b
  return {"-": lambda: a-b, "*": lambda: a*b, "/": lambda: a/b, "//": lambda: a//b, "%": lambda: a % b}[op]()
def _test(name, v):
  if name == "defined": return not isinstance(v, Undefined)
  if name == "undefined": return isinstance(v, Undefined)
  if name == "none": return v is None
  if name == "mapping": return isinstance(v, dict)
  if name == "iterable": return isinstance(v, (list, tuple, dict, str))
  if name in ("string",): return isinstance(v, str)
  if name == "number": return isinstance(v, (int, float)) and not isinstance(v, bool)
  if name in ("sequence",): return isinstance(v, (list, tuple, str))
  if name == "true": return v is True
  if name == "false": return v is False
  raise JinjaError(f"unsupported test: {name}")
def _tojson(v, indent=None):
  return json.dumps(v, indent=indent, ensure_ascii=False)
def _filter(name, v, args, kwargs):
  if name == "trim": return v.strip(args[0]) if args else v.strip()
  if name == "tojson": return _tojson(v, kwargs.get("indent", args[0] if args else None))
  if name == "length" or name == "count": return len(v)
  if name == "lower": return v.lower()
  if name == "upper": return v.upper()
  if name == "capitalize": return v.capitalize()
  if name == "title": return v.title()
  if name == "string": return _stringify(v)
  if name == "int": return int(v)
  if name == "float": return float(v)
  if name == "list": return list(v)
  if name == "first": return v[0]
  if name == "last": return v[-1]
  if name == "replace": return v.replace(args[0], args[1])
  if name == "join": return (args[0] if args else "").join(_stringify(x) for x in v)
  if name == "default" or name == "d": return args[0] if (isinstance(v, Undefined) or (args[1:] and args[1] and not v)) else v
  if name == "trim_start" or name == "lstrip": return v.lstrip()
  raise JinjaError(f"unsupported filter: {name}")

def _raise_exception(msg=""): raise JinjaError(str(msg))
_GLOBALS = {
  "namespace": lambda **kw: Namespace(**kw),
  "raise_exception": _raise_exception,
  "strftime_now": lambda fmt: datetime.datetime.now().strftime(fmt),
  "none": None, "true": True, "false": False,
}

def render(template:str, **context) -> str:
  """Render a Jinja chat template to a string. Raises JinjaError on any unsupported construct."""
  ast = _parse_template(_apply_whitespace(_lex(template)))
  r = _Renderer()
  r.render(ast, _Ctx([dict(context)]))
  return "".join(r.out)
