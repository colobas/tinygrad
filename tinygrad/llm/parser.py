from __future__ import annotations
import json, re
from typing import NamedTuple, Iterator, Iterable

# Per-model reasoning formats. The catalog's thinking models (Qwen3/3.5, DeepSeek-R1) all delimit their
# chain-of-thought with <think>...</think>; channel-based (harmony) or other family-specific formats would
# register as new entries here. Selection is data-driven (detect_reasoning_format) so adding a family is a
# table entry, not a parser change — this is the per-model abstraction the StreamParser consumes.
REASONING_FORMATS: dict[str, tuple[str, str]] = {
  "think": ("<think>", "</think>"),     # Qwen3 / Qwen3.5 / DeepSeek-R1
}
DEFAULT_REASONING = REASONING_FORMATS["think"]

def detect_reasoning_format(chat_template:str|None=None, special_tokens:Iterable[str]=()) -> tuple[str, str]:
  # pick the reasoning marker pair from model metadata: a model whose template or special-token set
  # references a known close marker uses that format. Falls back to think-tags, which are harmless for
  # non-thinking models (they never emit the marker, so nothing is ever split into reasoning_content).
  hay = (chat_template or "") + " " + " ".join(special_tokens)
  for fmt in REASONING_FORMATS.values():
    if fmt[1] in hay: return fmt
  return DEFAULT_REASONING

# A StreamParser consumes the host-side decoded token-text stream and yields tagged segments.
# It is the single seam the server uses to split a model's raw output into the channels the OpenAI
# contract distinguishes: visible answer text (content), chain-of-thought (reasoning), tool calls,
# and stop-sequence termination. Today the server fed decoded tokens straight to `delta.content`;
# this layer gives that path a parser so reasoning/stop/tool handling can plug in per model.

class Segment(NamedTuple):
  kind: str   # one of: "content" | "reasoning" | "tool_call" | "stop"
  text: str   # for "stop", the matched stop string (never emitted to the client);
              # for "tool_call", a JSON string {"name", "arguments"} (arguments itself a JSON string)

def _holdback_len(buf:str, markers:list[str]) -> int:
  # longest k>0 such that buf's k-char suffix is a *proper* prefix of some marker, so we can hold it
  # back instead of emitting text that might turn out to be the start of a marker split across tokens.
  best = 0
  for marker in markers:
    for k in range(min(len(buf), len(marker)-1), 0, -1):
      if marker.startswith(buf[-k:]):
        best = max(best, k)
        break
  return best

_FUNC_RE = re.compile(r"<function=([^>\n]+)>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
def parse_tool_call_body(body:str) -> dict | None:
  # parse the text inside a <tool_call>...</tool_call> block into {"name", "arguments"} where arguments
  # is a JSON string (the OpenAI shape). Handles two catalog formats: Hermes JSON (Qwen3) and the
  # XML <function=..>/<parameter=..> form (Qwen3.5). Returns None if the body parses as neither.
  s = body.strip()
  if s.startswith("{"):  # Hermes JSON: {"name": ..., "arguments": {...}}
    try: obj = json.loads(s)
    except json.JSONDecodeError: return None
    if not isinstance(obj, dict) or "name" not in obj: return None
    args = obj.get("arguments", obj.get("parameters", {}))
    return {"name": obj["name"], "arguments": args if isinstance(args, str) else json.dumps(args)}
  if (fn := _FUNC_RE.search(s)) is not None:  # Qwen3.5 XML: <function=name><parameter=k>v</parameter>...
    args: dict = {}
    for k, v in _PARAM_RE.findall(s):
      try: args[k] = json.loads(v)  # coerce to typed JSON when possible (numbers/bools/objects)
      except json.JSONDecodeError: args[k] = v
    return {"name": fn.group(1), "arguments": json.dumps(args)}
  return None

class StreamParser:
  """Incrementally splits a model's decoded text into tagged `Segment`s.

  - `reasoning`: an (open, close) marker pair (e.g. `("<think>", "</think>")`). Text between the markers
    is emitted as `reasoning`; text outside as `content`. `None` disables reasoning splitting, in which
    case all text is emitted verbatim as `content` (the server's prior behavior).
  - `stop`: stop strings matched on the *content* stream only (never inside reasoning). On a match the
    parser emits a `Segment("stop", <matched>)` and discards all subsequent buffered text.
  - `reasoning_started`: start already inside reasoning (for templates/prefills that open `<think>`
    themselves so the model only emits the close marker).
  - `tool_call`: an (open, close) marker pair (e.g. `("<tool_call>", "</tool_call>")`). Text between
    the markers is buffered as a tool-call block, parsed via `parse_tool_call_body`, and emitted as a
    `tool_call` Segment whose text is the JSON `{"name", "arguments"}`. The raw block text is never
    emitted as `content`. `None` disables tool-call parsing.

  Markers and stop strings are matched across `feed()` calls, so they are detected even when split
  across token boundaries. `feed(text)` returns segments decodable so far; `flush()` drains the tail.
  """
  def __init__(self, reasoning:tuple[str,str]|None=None, stop:list[str]|None=None, reasoning_started:bool=False,
               tool_call:tuple[str,str]|None=None):
    self.reasoning, self.stop, self.tool_call = reasoning, stop or [], tool_call
    self.buf = ""
    self.state = "reasoning" if (reasoning is not None and reasoning_started) else "content"
    self.stopped = False

  def _kind(self) -> str: return "reasoning" if self.state == "reasoning" else "content"
  def _triggers(self) -> list[tuple[str, str]]:
    if self.state == "reasoning": return [(self.reasoning[1], "close")] if self.reasoning is not None else []
    if self.state == "tool_call": return [(self.tool_call[1], "endtool")] if self.tool_call is not None else []
    t: list[tuple[str, str]] = []
    if self.reasoning is not None: t.append((self.reasoning[0], "open"))
    if self.tool_call is not None: t.append((self.tool_call[0], "opentool"))
    return t + [(s, "stop") for s in self.stop if s]

  def _run(self, final:bool) -> list[Segment]:
    out: list[Segment] = []
    if self.stopped: return out
    while True:
      triggers = self._triggers()
      best: tuple[int, str, str] | None = None  # (index, action, marker)
      for marker, act in triggers:
        if (i := self.buf.find(marker)) != -1 and (best is None or i < best[0] or (i == best[0] and len(marker) > len(best[2]))):
          best = (i, act, marker)
      if best is not None:
        i, act, marker = best
        if i > 0 and self.state != "tool_call": out.append(Segment(self._kind(), self.buf[:i]))
        body = self.buf[:i]
        self.buf = self.buf[i+len(marker):]
        if act == "open": self.state = "reasoning"
        elif act == "close": self.state = "content"
        elif act == "opentool": self.state = "tool_call"
        elif act == "endtool":  # parse the buffered block; emit a tool_call (drop if unparseable)
          if (call := parse_tool_call_body(body)) is not None: out.append(Segment("tool_call", json.dumps(call)))
          self.state = "content"
        else:  # stop: drop the marker and everything after it
          out.append(Segment("stop", marker))
          self.buf, self.stopped = "", True
          return out
        continue
      if self.state == "tool_call" and not final: return out  # buffer the whole block until its close marker
      if final:
        if self.buf and self.state != "tool_call": out.append(Segment(self._kind(), self.buf))
        self.buf = ""
        return out
      hold = _holdback_len(self.buf, [m for m, _ in triggers])
      if (emit := self.buf[:len(self.buf)-hold] if hold else self.buf): out.append(Segment(self._kind(), emit))
      self.buf = self.buf[len(self.buf)-hold:] if hold else ""
      return out

  def feed(self, text:str) -> Iterator[Segment]:
    self.buf += text
    yield from self._run(final=False)
  def flush(self) -> Iterator[Segment]:
    yield from self._run(final=True)
