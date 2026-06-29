import unittest
from unittest.mock import patch
from tinygrad import Tensor, UOp
from tinygrad.schedule import schedule_cache
from tinygrad.llm.model import Transformer, TransformerConfig

TEST_CONFIG = TransformerConfig(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                           norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, rope_dim=32, v_head_dim=32, max_context=32)

class TestTransformerGenerate(unittest.TestCase):
  def test_kv_cache_reuse(self):
    """Test that generate reuses the KV cache when tokens extend the cached prefix."""
    model = Transformer(TEST_CONFIG)

    captured_inputs = []
    def mock_call(self, tokens, start_pos, temperature):
      captured_inputs.append((tokens.shape, start_pos if isinstance(start_pos, int) else start_pos.val))
      return Tensor([[42]])

    with patch.object(Transformer, '__call__', mock_call):
      # first conversation: prefill 5 tokens + 1 decode
      tokens = [1, 2, 3, 4, 5]
      gen = model.generate(tokens)
      next(gen)  # prefill
      next(gen)  # decode

      # second call extends the conversation — cached prefix should be reused
      captured_inputs.clear()
      tokens = [1, 2, 3, 4, 5, 42, 42, 10, 11, 12]
      gen = model.generate(tokens)
      next(gen)

    # should process tokens[6:] = [42, 10, 11, 12] since first 6 have cached k/v
    toks_shape = captured_inputs[0][0][-1]
    self.assertEqual(toks_shape.val if isinstance(toks_shape, UOp) else toks_shape, 4)
    self.assertEqual(captured_inputs[0][1], 6)

  def test_kv_cache_invalidation(self):
    """Test that generate invalidates the KV cache when tokens diverge from the cached prefix."""
    model = Transformer(TEST_CONFIG)

    captured_inputs = []
    def mock_call(self, tokens, start_pos, temperature):
      captured_inputs.append((tokens.shape, start_pos if isinstance(start_pos, int) else start_pos.val))
      return Tensor([[42]])

    with patch.object(Transformer, '__call__', mock_call):
      # first conversation
      gen = model.generate([1, 2, 3, 4, 5])
      next(gen)

      # completely different prompt — KV cache should be invalidated
      captured_inputs.clear()
      gen = model.generate([10, 20, 30])
      next(gen)

    # should process all 3 tokens from start
    toks_shape = captured_inputs[0][0][-1]
    self.assertEqual(toks_shape.val if isinstance(toks_shape, UOp) else toks_shape, 3)
    self.assertEqual(captured_inputs[0][1], 0)

  def test_two_prompts_schedule_cache(self):
    """Third prompt should hit the schedule cache, not miss (first two warm up both jits: prefill + decode)."""
    from dataclasses import replace
    model = Transformer(replace(TEST_CONFIG, max_context=64))

    # first two prompts warm up both jits (prefill + decode)
    ids = list(range(1, 6))
    gen = model.generate(ids)
    for _ in range(3): next(gen)

    ids += list(range(10, 15))
    gen = model.generate(ids)
    for _ in range(3): next(gen)
    cache_size_after_warmup = len(schedule_cache)

    # third prompt should reuse the same schedule cache entries, not create new ones
    ids += list(range(20, 25))
    gen = model.generate(ids)
    for _ in range(3): next(gen)

    self.assertEqual(cache_size_after_warmup, len(schedule_cache),
      f"third prompt added {len(schedule_cache) - cache_size_after_warmup} new schedule cache entries (expected 0)")

  def test_chunked_prefill(self):
    """When prompt > chunk_size, all chunks should be prefill"""
    from tinygrad.uop.ops import resolve
    from dataclasses import replace
    model = Transformer(replace(TEST_CONFIG, max_context=64))

    def get_prefill_flags(tokens, chunk_size):
      is_prefill = []
      def mock_call(self, tokens, start_pos, temperature):
        is_prefill.append(resolve(tokens.shape[1] != 1))
        return Tensor([[42]])
      with patch.object(Transformer, '__call__', mock_call):
        gen = model.generate(tokens, chunk_size=chunk_size)
        for _ in range(3): next(gen)
      model._cached_tokens = []
      return is_prefill

    # 8 tokens, chunk_size=4 -> 2 prefill chunks
    self.assertEqual(get_prefill_flags(list(range(8)), 4), [True, True, False, False])
    # 9 tokens, chunk_size=4 -> 3 prefill chunks (4+4+1)
    self.assertEqual(get_prefill_flags(list(range(9)), 4), [True, True, True, False, False])
    # 4 tokens, chunk_size=4 -> 1 prefill chunk
    self.assertEqual(get_prefill_flags(list(range(4)), 4), [True, False, False])

  def test_kv_cache_resume_matches_fresh(self):
    model = Transformer(TEST_CONFIG)

    # generate 2 tokens, then abandon
    prompt = list(range(1, 6))
    gen = model.generate(list(prompt))
    out1, out2 = next(gen), next(gen)

    # resume with conversation history + new user tokens appended
    extended = prompt + [out1, out2, 10, 11, 12]
    gen = model.generate(list(extended))
    resumed_out = [next(gen) for _ in range(3)]

    # compare against fresh generation (no cache) of the same prompt
    model._cached_tokens = []
    gen = model.generate(list(extended))
    fresh_out = [next(gen) for _ in range(3)]

    self.assertEqual(fresh_out, resumed_out)

  def test_temperature_zero_is_greedy(self):
    """Temperature 0 (or near 0) should produce deterministic output."""
    model = Transformer(TEST_CONFIG)
    tokens = list(range(1, 6))
    results = [list(zip(range(5), model.generate(list(tokens)))) for _ in range(3)]
    # all runs should produce the same tokens
    self.assertEqual(results[0], results[1])
    self.assertEqual(results[1], results[2])

  def test_temperature_high_produces_variety(self):
    """High temperature should produce different outputs across runs."""
    model = Transformer(TEST_CONFIG)
    tokens = list(range(1, 6))
    runs = set()
    for _ in range(5):
      gen = model.generate(list(tokens), temperature=2.0)
      out = tuple(next(gen) for _ in range(10))
      runs.add(out)
    # with temperature=2.0, we should see at least 2 distinct outputs across 5 runs
    self.assertGreater(len(runs), 1, "high temperature should produce varied outputs")

  def test_temperature_passed_to_forward(self):
    """Temperature from generate should be passed through to __call__."""
    model = Transformer(TEST_CONFIG)
    captured_temps = []
    def mock_call(self, tokens, start_pos, temperature):
      captured_temps.append(float(temperature.item()))
      return Tensor([[42]])
    with patch.object(Transformer, '__call__', mock_call):
      gen = model.generate([1, 2, 3], temperature=0.6)
      next(gen)
    self.assertAlmostEqual(captured_temps[-1], 0.6, places=5)

class TestConstrainedDecoding(unittest.TestCase):
  def test_masked_generate_forces_conforming_json(self):
    """A grammar mask threaded into the real decode graph forces output that conforms to the schema."""
    import json
    from tinygrad.llm.grammar import GrammarMasker
    model = Transformer(TEST_CONFIG)  # vocab_size=100, random weights -> arbitrary logits
    # synthetic char vocab; every token the grammar might need is present, eos is id 99
    chars = ['{', '}', '"', ':', 'o', 'k', 't', 'r', 'u', 'e', 'f', 'a', 'l', 's']
    token_texts = {i: c for i, c in enumerate(chars)}
    eos = 99
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    masker = GrammarMasker(schema, token_texts, {eos}, vocab_size=100)

    out_ids = []
    for tid in model.generate([1, 2, 3], temperature=0.0, mask_fn=masker.mask):
      if tid == eos: break
      out_ids.append(tid)
      self.assertIn(tid, token_texts, "model sampled a grammar-forbidden token")
      if len(out_ids) > 50: self.fail("generation did not terminate")
    text = "".join(token_texts[t] for t in out_ids)
    self.assertIn(json.loads(text), ({"ok": True}, {"ok": False}))  # whichever boolean the weights favored

  def test_unmasked_path_unchanged(self):
    """Generation without mask_fn must be byte-identical to before (opt-in gate)."""
    model = Transformer(TEST_CONFIG)
    a = [t for _, t in zip(range(6), model.generate([1, 2, 3], temperature=0.0))]
    model._cached_tokens = []
    b = [t for _, t in zip(range(6), model.generate([1, 2, 3], temperature=0.0, mask_fn=None))]
    self.assertEqual(a, b)

class TestSamplingPenalties(unittest.TestCase):
  def setUp(self): self.m = Transformer(TEST_CONFIG)

  def test_repetition_penalty_is_sign_dependent(self):
    # seen tokens: positive logits divided by rep_pen, negative logits multiplied; unseen untouched
    logits, counts = Tensor([[2.0, 1.0, -1.0]]), Tensor([1.0, 0.0, 1.0])
    out = self.m._penalize(logits, counts, Tensor([2.0]), Tensor([0.0]), Tensor([0.0])).tolist()[0]
    self.assertAlmostEqual(out[0], 1.0, places=4)   # 2.0 / 2 (seen, positive)
    self.assertAlmostEqual(out[1], 1.0, places=4)   # unseen, unchanged
    self.assertAlmostEqual(out[2], -2.0, places=4)  # -1.0 * 2 (seen, negative)

  def test_presence_and_frequency_are_additive(self):
    logits, counts = Tensor([[2.0, 2.0, 2.0]]), Tensor([0.0, 1.0, 3.0])
    out = self.m._penalize(logits, counts, Tensor([1.0]), Tensor([0.5]), Tensor([0.1])).tolist()[0]
    self.assertAlmostEqual(out[0], 2.0, places=4)   # unseen
    self.assertAlmostEqual(out[1], 1.4, places=4)   # 2 - 0.5(presence) - 0.1(freq*1)
    self.assertAlmostEqual(out[2], 1.2, places=4)   # 2 - 0.5(presence) - 0.3(freq*3)

  def test_neutral_penalties_are_identity(self):
    logits, counts = Tensor([[2.0, -1.0, 0.5]]), Tensor([5.0, 5.0, 5.0])  # all seen, but neutral params
    out = self.m._penalize(logits, counts, Tensor([1.0]), Tensor([0.0]), Tensor([0.0])).tolist()[0]
    self.assertEqual([round(x, 4) for x in out], [2.0, -1.0, 0.5])

  def test_penalty_changes_decode_end_to_end(self):
    # fixed logits (token 0 highest) -> greedy repeats token 0; a presence penalty bigger than the 0-vs-1 gap
    # must push the second pick to token 1, proving counts accumulate and feed the sampler through the JIT.
    fixed = Tensor([[10.0, 9.0] + [0.0] * (TEST_CONFIG.vocab_size - 2)])
    self.m._logits = lambda tokens, start_pos: fixed
    base = [t for _, t in zip(range(3), self.m.generate([1, 2, 3], temperature=0.0))]
    self.m._cached_tokens = []
    pen = [t for _, t in zip(range(3), self.m.generate([1, 2, 3], temperature=0.0, presence_pen=5.0))]
    self.assertEqual(base, [0, 0, 0])      # greedy loops on the highest-logit token
    self.assertEqual(pen[:2], [0, 1])      # after token 0 is penalized (10-5 < 9), token 1 wins

if __name__ == '__main__':
  unittest.main()
