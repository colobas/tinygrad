import unittest, json
from tinygrad.llm.grammar import initial, step, can_eof, GrammarMasker, build_trie

def accepts(schema, s):
  # feed the whole string; return True iff every char is permitted and the value ends complete
  st = initial(schema)
  for ch in s:
    st = step(st, ch)
    if st is None: return False
  return can_eof(st)

def viable_prefix(schema, s):
  # True if `s` is a grammar-viable prefix (not necessarily complete)
  st = initial(schema)
  for ch in s:
    st = step(st, ch)
    if st is None: return False
  return True

class TestScalars(unittest.TestCase):
  def test_string(self):
    self.assertTrue(accepts({"type": "string"}, '"hello"'))
    self.assertTrue(accepts({"type": "string"}, '"with \\"escape\\" and \\u00e9"'))
    self.assertFalse(accepts({"type": "string"}, '"unterminated'))
    self.assertFalse(accepts({"type": "string"}, '123'))
  def test_integer(self):
    for s in ("0", "-5", "42", "1000"):
      self.assertTrue(accepts({"type": "integer"}, s), s)
    self.assertFalse(accepts({"type": "integer"}, "3.14"))  # no float for integer
    self.assertFalse(accepts({"type": "integer"}, "01"))     # no leading zero
    self.assertFalse(accepts({"type": "integer"}, "-"))      # incomplete
  def test_number(self):
    for s in ("0", "-5", "3.14", "1e10", "-2.5E-3", "0.5"):
      self.assertTrue(accepts({"type": "number"}, s), s)
    self.assertFalse(accepts({"type": "number"}, "."))
    self.assertFalse(accepts({"type": "number"}, "1."))
  def test_boolean_null(self):
    self.assertTrue(accepts({"type": "boolean"}, "true"))
    self.assertTrue(accepts({"type": "boolean"}, "false"))
    self.assertFalse(accepts({"type": "boolean"}, "tru"))
    self.assertTrue(accepts({"type": "null"}, "null"))
  def test_enum(self):
    sc = {"enum": ["red", "green", 3]}
    self.assertTrue(accepts(sc, '"red"'))
    self.assertTrue(accepts(sc, '"green"'))
    self.assertTrue(accepts(sc, '3'))
    self.assertFalse(accepts(sc, '"blue"'))
    self.assertTrue(viable_prefix(sc, '"gr'))   # still viable toward "green"
    self.assertFalse(accepts(sc, '"gr'))         # but not complete

class TestComposite(unittest.TestCase):
  OBJ = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
  def test_object(self):
    self.assertTrue(accepts(self.OBJ, '{"name":"bob","age":42}'))
    self.assertFalse(accepts(self.OBJ, '{"age":42,"name":"bob"}'))   # declared order enforced
    self.assertFalse(accepts(self.OBJ, '{"name":"bob"}'))            # all declared props required
    self.assertFalse(accepts(self.OBJ, '{"name":42,"age":42}'))      # wrong value type
  def test_empty_object(self):
    self.assertTrue(accepts({"type": "object", "properties": {}}, "{}"))
  def test_array(self):
    sc = {"type": "array", "items": {"type": "integer"}}
    self.assertTrue(accepts(sc, "[]"))
    self.assertTrue(accepts(sc, "[1,2,3]"))
    self.assertFalse(accepts(sc, "[1,2,]"))     # trailing comma
    self.assertFalse(accepts(sc, '[1,"x"]'))    # wrong item type
  def test_nested(self):
    sc = {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    self.assertTrue(accepts(sc, '{"tags":["a","b"]}'))
    self.assertFalse(accepts(sc, '{"tags":[1]}'))
  def test_any_value(self):
    self.assertTrue(accepts({}, '{"a":[1,true,null,"x"]}'))

class TestGrammarMasker(unittest.TestCase):
  def _greedy(self, schema, vocab):
    # vocab: list of token strings; token id == index. eos is the last id. Greedily pick the lowest-id
    # allowed token each step (deterministic) and assemble the output until eos is chosen.
    texts = {i: t for i, t in enumerate(vocab)}
    eos = len(vocab)
    m = GrammarMasker(schema, texts, {eos}, vocab_size=eos+1, neg=-1e9)
    out_ids, out = [], ""
    for _ in range(200):
      bias = m.mask(out_ids)
      allowed = [i for i, b in enumerate(bias) if b == 0.0]
      self.assertTrue(allowed, "grammar left no legal token")
      pick = min(allowed)
      if pick == eos: break
      out_ids.append(pick); out += texts[pick]
    return out

  def test_forces_valid_object(self):
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    vocab = ['{', '}', '"', 'ok', ':', 'true', 'false', ',']
    out = self._greedy(schema, vocab)
    self.assertEqual(json.loads(out), {"ok": True})  # greedy picks 'true' (lower id than 'false')

  def test_eos_only_when_complete(self):
    schema = {"type": "string"}
    texts = {0: '"', 1: 'a'}
    m = GrammarMasker(schema, texts, {2}, vocab_size=3)
    self.assertEqual(m.mask([])[2], m.neg)        # cannot stop before the string starts
    self.assertEqual(m.mask([0, 1])[2], m.neg)    # cannot stop mid-string (unclosed quote)
    self.assertEqual(m.mask([0, 1, 0])[2], 0.0)   # "a" complete -> eos now allowed

class TestTrieMaskEquivalence(unittest.TestCase):
  # the trie walk is an optimization; it must produce exactly the same allowed set as a brute-force
  # per-token feed from the committed state, at every position along a generation.
  VOCAB = ['{', '}', '"', ':', ',', '[', ']', 'na', 'me', 'name', 'ok', 'true', 'false',
           'null', '1', '2', '42', '-', 'x', 'ab', 'abc', '\\n', 't', 'r', 'u', 'e']

  def _brute_allowed(self, base, vocab):
    allowed = set()
    for tid, text in enumerate(vocab):
      st = base
      for ch in text:
        st = step(st, ch)
        if st is None: break
      if st is not None: allowed.add(tid)
    return allowed

  def _check(self, schema, completion):
    texts = {i: t for i, t in enumerate(self.VOCAB)}
    eos = len(self.VOCAB)
    m = GrammarMasker(schema, texts, {eos}, vocab_size=eos+1, trie=build_trie(texts))
    # commit `completion` tokens (by id) and compare the trie mask to the brute-force reference
    bias = m.mask(completion)
    trie_allowed = {i for i, b in enumerate(bias[:eos]) if b == 0.0}
    self.assertEqual(trie_allowed, self._brute_allowed(m.stack, self.VOCAB))

  def test_equivalence_object(self):
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "ok": {"type": "boolean"}}}
    self._check(schema, [])             # at '{'
    self._check(schema, [0])            # after '{', expecting first key
    self._check(schema, [0, 2])         # inside the key string
  def test_equivalence_string_and_array(self):
    self._check({"type": "string"}, [2])                                  # inside a string (permissive)
    self._check({"type": "array", "items": {"type": "integer"}}, [5])     # after '['

  def test_prebuilt_trie_used(self):
    texts = {0: "true", 1: "false"}
    trie = build_trie(texts)
    m = GrammarMasker({"type": "boolean"}, texts, {2}, 3, trie=trie)
    self.assertIs(m.trie, trie)
    self.assertEqual([i for i, b in enumerate(m.mask([])) if b == 0.0], [0, 1])  # both booleans allowed at start

if __name__ == "__main__":
  unittest.main()
