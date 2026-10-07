from __future__ import annotations
import enum, functools, itertools, math, pathlib, re
from typing import cast
from math import prod
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Device
from tinygrad.helpers import DEBUG
from tinygrad.llm.kernels.amd import Linear, gated_delta_prefill, flash_attention, amd_custom_kernels_supported, clear_activation_memos, \
  gated_delta_kernel_supported
from tinygrad.llm.gguf import gguf_parse, gguf_shard, ggml_nbytes, ggml_data_to_tensor
from tinygrad.uop.ops import resolve, Ops, KernelInfo

def _embedding_rows_kernel(out:UOp, table:UOp, idx:UOp) -> UOp:
  # gather the packed rows of a quantized embedding table: only the looked up rows get dequantized, never the whole table
  t, j = UOp.range(out.shape[0], 0), UOp.range(out.shape[1], 1)
  return out[t, j].store(table[idx[t].load().cast(dtypes.weakint), j].load()).end(t, j).sink(arg=KernelInfo(name="embedding_rows", opts_to_apply=()))

PREFILL_TAIL = 128 # max tokens per call of the prefill tail jit (Apple GPUs: 32-token calls ran at ~35 tok/s, 128 at ~95)

class ExpertGating(enum.IntEnum):
  SOFTMAX = 1
  SIGMOID = 2
  SOFTMAX_WEIGHT = 3  # softmax over the top-k selected logits
  SQRT_SOFTPLUS = 4

@dataclass(frozen=True)
class YaRNConfig:
  factor: float
  orig_ctx_len: int
  beta_fast: float
  beta_slow: float

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|tuple[str, ...]|None=None, yarn:YaRNConfig|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2) / dim))
  concentration = 1.0
  if yarn is not None and yarn.factor > 1.0:
    concentration = 0.1 * math.log(yarn.factor) + 1.0
    d_half = dim // 2
    low = max(0, math.floor(d_half * math.log(yarn.orig_ctx_len / (yarn.beta_fast * 2 * math.pi)) / math.log(theta)))
    high = min(dim-1, math.ceil(d_half * math.log(yarn.orig_ctx_len / (yarn.beta_slow * 2 * math.pi)) / math.log(theta)))
    interp = 1 - ((Tensor.arange(d_half).float() - low) / max(0.001, high-low)).clamp(0, 1)
    freqs = freqs * interp + (freqs / yarn.factor) * (1 - interp)
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return (freqs.cos() * concentration).cat(freqs.sin() * concentration, dim=-1).clone(device)

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int, bias:bool=False):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
    if bias: self.bias = Tensor.zeros(num_experts, out_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    ret = (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)
    return ret + self.bias[sel] if hasattr(self, 'bias') else ret

def gated_activation(gate:Tensor, up:Tensor, *, alpha:float=1.0, limit:float|None=None, up_bias:float=0.0) -> Tensor:
  if limit is not None and limit > 0: gate, up = gate.clamp(max_=limit), up.clamp(-limit, limit)
  return gate * (gate * alpha).sigmoid() * (up + up_bias)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  yarn: YaRNConfig|None = None
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  expert_gating_func: ExpertGating = ExpertGating.SOFTMAX
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  ssm_layers: tuple[bool, ...] = ()
  attn_output_gate: bool = False
  attn_output_bias: bool = False
  attn_sinks: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False
  expert_proj_bias: bool = False
  swiglu_alpha: float = 1.0
  swiglu_clamp_exp: float|None = None
  swiglu_up_bias: float = 0.0
  sliding_window: int = 0
  sliding_window_pattern: int = 0
  num_mtp_heads: int = 0  # trailing MTP (nextn) blocks, for speculative decoding (generate_mtp)
  mtp_ssm_layer: bool = False  # the MTP block is a GatedDeltaNetBlock (vs an attention block)

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=config.expert_proj_bias)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim, bias=config.expert_proj_bias)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim, bias=config.expert_proj_bias)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim, bias=config.expert_proj_bias)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      bias = self.exp_probs_b["bias"] if hasattr(self, 'exp_probs_b') else None
      gating, normalize_topk = self.config.expert_gating_func, self.config.norm_topk_prob
      # fast path: without selection bias, normalized SOFTMAX is equivalent to SOFTMAX_WEIGHT
      if gating == ExpertGating.SOFTMAX and bias is None and normalize_topk:
        gating, normalize_topk = ExpertGating.SOFTMAX_WEIGHT, False
      if   gating == ExpertGating.SOFTMAX_WEIGHT: scores = logits
      elif gating == ExpertGating.SOFTMAX:        scores = logits.softmax(-1)
      elif gating == ExpertGating.SIGMOID:        scores = logits.sigmoid()
      elif gating == ExpertGating.SQRT_SOFTPLUS:  scores = logits.softplus().sqrt()

      _, sel = pairwise_topk(scores if bias is None else scores + bias, self.config.num_experts_per_tok)
      probs = scores.gather(-1, sel)
      # SOFTMAX_WEIGHT applies softmax after top-k selection
      if gating == ExpertGating.SOFTMAX_WEIGHT: probs = probs.softmax(-1)
      if normalize_topk: probs = probs / probs.sum(axis=-1, keepdim=True)
      probs = probs * self.config.routed_scaling_factor
      gate, up = self.ffn_gate_exps(sel, h), self.ffn_up_exps(sel, h)
      act = gated_activation(gate, up, alpha=self.config.swiglu_alpha, limit=self.config.swiglu_clamp_exp, up_bias=self.config.swiglu_up_bias)
      x_down = self.ffn_down_exps(sel, act.contiguous())  # (B, T, k, D)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x).silu().contiguous() * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError
  # MTP speculative decoding (Transformer.generate_mtp). a KV cache block verifies like it decodes: a rejected draft's keys sit past the
  # committed length and are overwritten by the next round. recurrent blocks override these to keep their state untouched until commit
  def _attention_verify(self, x:Tensor, start_pos:int|UOp) -> Tensor: return self._attention(x, start_pos)
  def _init_verify_state(self, x:Tensor) -> None: pass # eager, before the traced region (like _init_state)
  def commit_verify(self, accept:int) -> list[UOp]: return [] # the state writes that land the accepted prefix

  def __call__(self, x: Tensor, start_pos: int|UOp, verify:bool=False):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    if verify: self._init_verify_state(x)
    attn_fn = self._attention_verify if verify else self._attention
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp):
      h =     x + attn_fn(self.attn_norm(x), start_pos)
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return _run(x, start_pos)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=config.attn_output_bias)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)
    if config.attn_sinks: self.attn_sinks = {"weight": Tensor.zeros(config.n_heads)}

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    store = self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).cast(self.cache_kv.dtype).uop)
    assigned_kv = Tensor(self.cache_kv.uop.after(store))
    # on RDNA3/4, hybrid models use custom flash attention kernels on the KV cache
    if amd_custom_kernels_supported(x.device) and self.config.ssm is not None:
      attn = flash_attention(q, assigned_kv, start_pos+T)
      attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
      return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))
    k = assigned_kv[0, :, :, 0:start_pos+T, :]
    v = assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask, window = None, self.config.sliding_window
    if resolve(T != 1) or window:
      mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False)
      mask = mask.triu(start_pos+1) + mask.tril(start_pos-window) if window else mask.triu(start_pos+1)
    if hasattr(self, 'attn_sinks'):
      k, v = k.cat(k[..., :1, :].const_like(0), dim=-2), v.cat(v[..., :1, :].const_like(0), dim=-2)
      sink_col = self.attn_sinks["weight"].reshape(1, -1, 1, 1).expand(1, self.config.n_heads, T, 1)
      if mask is None: mask = Tensor.zeros(1, 1, T, start_pos+T, dtype=x.dtype, buffer=False)
      mask = mask.expand(1, self.config.n_heads, T, start_pos+T).cat(sink_col, dim=-1)
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # zeroed so the flash kernels can safely read whole tiles past the valid region (masked lanes multiply by 0)
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim,
                                   dtype=dtypes.half, device=x.device if isinstance(x.device, str) else None)
      if isinstance(x.device, tuple): self.cache_kv = self.cache_kv.shard(x.device, 2).realize()
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta,
                                            device=x.device, yarn=self.config.yarn)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    if not self.config.ssm or not self.config.ssm.kda: q_rope = apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T])
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(q_rope, dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2)
    if not self.config.ssm or not self.config.ssm.kda: k_rope = apply_rope(k_rope, self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta,
                                            device=x.device, yarn=self.config.yarn)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
    if ssm.kda:
      self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _project(self, x:Tensor, start_pos:int|UOp):
    """conv window and q/k/v/beta/alpha/out_gate, shared by the scan (_attention) and the MTP verify window (_attention_verify)"""
    B, T, _ = x.shape
    # bind ints to a variable so the reset flag stays a runtime value (it toggles when generation restarts at position 0)
    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    initial = Tensor(start_pos).eq(0)
    is_kda = hasattr(self, "ssm_g_a")
    symbolic = isinstance(T, UOp)
    T_pad = x.max_shape[1]  # symbolic chunks are padded to their max size: one graph serves every size

    # input processing
    x = x.half()
    out_gate = self.ssm_g_b(self.ssm_g_a(x)) if is_kda else self.attn_gate(x)
    out_gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, T, self.num_v_heads)
    alpha = self.ssm_f_b(self.ssm_f_a(x)) if is_kda else self.ssm_alpha(x)
    log_alpha = ((alpha.float() + self.ssm_dt["bias"]).softplus().reshape(B, T, self.num_v_heads, -1) *
                 self.ssm_a.reshape(self.num_v_heads, -1))

    # qkv conv, conv_state is reset when starting from position 0
    conv_state = initial.where(0, self.conv_state)
    # assemble the conv window in a static-size buffer: [conv_state | qkv rows | zero-pad].
    # padded steps are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1 after exp)
    conv_window = conv_state.cat(self.attn_qkv(x).cast(conv_state.dtype), dim=1)
    conv_window = conv_window.pad_to((B, self.ssm_conv_kernel-1 + T_pad, self.conv_channels)).contiguous()
    # the last conv_kernel-1 columns of the window become the next conv state
    conv_state_store = self.conv_state.uop.store(conv_window[:, T:T+self.ssm_conv_kernel-1].cast(self.conv_state.dtype).uop)

    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T_pad] * self.ssm_conv1d["weight"][:, i] for i in range(self.ssm_conv_kernel))).silu()
    if symbolic:
      out_gate = out_gate.pad_to((B, T_pad, self.num_v_heads, self.head_v_dim))
      beta, log_alpha = beta.pad_to((B, T_pad, self.num_v_heads)), log_alpha.pad_to((B, T_pad, *log_alpha.shape[2:]))
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    qk_eps = 1e-12 if is_kda else 1e-6
    q, k = (z.reshape(B, T_pad, self.num_k_heads, self.head_k_dim).normalize(dim=-1, eps=qk_eps)
            .repeat(1, 1, self.num_v_heads//self.num_k_heads, 1) for z in (q, k))
    v = v.reshape(B, T_pad, self.num_v_heads, self.head_v_dim)
    # layout the per-step operands to broadcast against the (B, H, V, K) state
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q = q * self.head_k_dim**-0.5
    alpha = log_alpha.transpose(1, 2).exp()  # per-channel decay for kda, per-head otherwise (B, H, T, K|1)

    return q, k, v, beta, alpha, out_gate, conv_state_store, conv_window, initial, is_kda, symbolic, T_pad, start_pos

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q, k, v, beta, alpha, out_gate, conv_state_store, _, initial, is_kda, symbolic, T_pad, start_pos = self._project(x, start_pos)

    # recurrent: scan over the (padded) tokens, updating the recurrent state. collect the per-step outputs
    state = Tensor(self.recurrent_state.uop.after(conv_state_store))  # carry the conv write into this graph
    if self.head_k_dim % 32 == 0 and self.head_v_dim % 4 == 0 and gated_delta_kernel_supported(x.device):
      # one fused kernel for the whole scan; it resets and updates the recurrent state in place (RDNA3/4)
      core = gated_delta_prefill(q, k, v, beta, alpha, state, Tensor(start_pos)).transpose(1, 2)
    else:
      q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
      alpha = alpha.unsqueeze(-2)
      state = initial.where(0, state.float())
      outs = []
      for t in range(T_pad):
        s1 = state * alpha[:, :, t]  # decay the state
        delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]  # the delta rule update
        state = s1 + delta * k[:, :, t]
        outs.append((state * q[:, :, t]).sum(-1))

      # store the updated recurrent state in place, then read the stacked outputs after the write
      state_store = self.recurrent_state.uop.store(state.cast(self.recurrent_state.dtype).uop)
      core = Tensor(outs[0].stack(*outs[1:], dim=1).contiguous().uop.after(state_store))

    # output; undo the padding before the output projection
    z = (self.ssm_norm(core) * (out_gate.sigmoid() if is_kda else out_gate.silu())).cast(dtypes.half).contiguous()
    if symbolic: z = z[:, :T]
    return self.ssm_out(z.reshape(B, T, -1))

  # MTP verify window stashes (see _init_verify_state)
  _verify_N: int
  _verify_q: Tensor
  _verify_k: Tensor
  _verify_v: Tensor
  _verify_beta: Tensor
  _verify_alpha: Tensor
  _verify_conv_window: Tensor

  def _attention_verify(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    """the N=K+1 token verify window, on a COPY of the recurrent state: the committed state must not move before the accept length is
    known. the projected operands and the conv window are stashed so commit_verify can land the accepted prefix on the real state"""
    B, N, _ = x.shape
    q, k, v, beta, alpha, out_gate, _, conv_window, initial, is_kda, symbolic, T_pad, start_pos = self._project(x, start_pos)
    assert isinstance(N, int) and not symbolic and T_pad == N and not is_kda, "verify: a static window, scalar-alpha gated delta rule"
    # the conv_state store of _project is not threaded in: verify must be free of side effects, only commit_verify writes state
    core = gated_delta_prefill(q, k, v, beta, alpha, initial.where(0, self.recurrent_state.float()).contiguous()).transpose(1, 2)
    stores = self._verify_q.uop.store(q.contiguous().uop)
    for buf, val in ((self._verify_k, k), (self._verify_v, v), (self._verify_beta, beta), (self._verify_alpha, alpha),
                     (self._verify_conv_window, conv_window)):
      stores = buf.uop.after(stores).store(val.contiguous().uop)
    z = Tensor((self.ssm_norm(core) * out_gate.silu()).cast(dtypes.half).contiguous().uop.after(stores)) # the output fires the stashes
    return self.ssm_out(z.reshape(B, N, -1))

  def commit_verify(self, accept:int) -> list[UOp]:
    """land the state after the accepted prefix (window positions 0..accept): rerun the fused scan on the real state with the rejected
    tail as no-ops (beta=0: no delta update, alpha=1: no decay). returns reads after the writes, which is what makes a realize fire them"""
    if not hasattr(self, "_verify_q"): return []
    N, keep = self._verify_N, (Tensor.arange(self._verify_N) <= accept).float()
    beta = self._verify_beta * keep.reshape(1, 1, N)
    alpha = self._verify_alpha * keep.reshape(1, 1, N, 1) + (1 - keep).reshape(1, 1, N, 1)
    core = gated_delta_prefill(self._verify_q, self._verify_k, self._verify_v, beta, alpha, Tensor(self.recurrent_state.uop))
    # the conv state of the prefix is the kernel-1 window columns ending at accept+1 (a static slice: accept is a python int)
    conv_store = self.conv_state.uop.store(self._verify_conv_window[:, accept+1:accept+self.ssm_conv_kernel].cast(self.conv_state.dtype)
                                           .contiguous().uop)
    return [core.uop, self.conv_state.uop.after(conv_store)]

  def _init_verify_state(self, x:Tensor) -> None:
    B, N = x.shape[0], x.shape[1]
    assert isinstance(N, int), "the verify window is static (K is fixed for a generate_mtp run)"
    if hasattr(self, "_verify_q") and self._verify_N == N: return
    H, Dk, Dv, dev = self.num_v_heads, self.head_k_dim, self.head_v_dim, x.device
    # zeros, never empty: commit_verify reads them, a stray NaN would poison every later state
    def z(*shape) -> Tensor: return Tensor.zeros(*shape, device=dev).contiguous().realize()
    self._verify_N, self._verify_q, self._verify_k, self._verify_v = N, z(B, H, N, Dk), z(B, H, N, Dk), z(B, H, N, Dv)
    self._verify_beta, self._verify_alpha = z(B, H, N), z(B, H, N, 1)
    self._verify_conv_window = z(B, self.ssm_conv_kernel-1+N, self.conv_channels)

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, device=x.device).clone()

class MTPHead:
  """the trailing multi-token-prediction ("nextn") block: from the main model's pre-norm hidden state at t and the embedding of the token
  at t+1 it predicts the token at t+2. it reuses token_embd and output, with its own norms, eh_proj and final norm (shared_head_norm is not
  tied to output_norm)"""
  def __init__(self, config:TransformerConfig):
    self.enorm, self.hnorm = nn.RMSNorm(config.dim, config.norm_eps), nn.RMSNorm(config.dim, config.norm_eps)
    self.eh_proj = Linear(2*config.dim, config.dim, bias=False)
    self.shared_head_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.block:FFNBlock = GatedDeltaNetBlock(config, config.ssm) if config.mtp_ssm_layer and config.ssm else \
      (MLATransformerBlock(config) if config.kv_lora_rank > 0 else TransformerBlock(config))
  def __call__(self, h_prev:Tensor, tok_embed:Tensor, start_pos:int|UOp) -> Tensor:
    # [enorm(embed); hnorm(h)]: the Qwen3.5+ family concatenates in the opposite order from DeepSeek-V3
    return self.block(self.eh_proj(self.enorm(tok_embed).cat(self.hnorm(h_prev), dim=-1)), start_pos)

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = []
    for i in range(config.num_blocks):
      c = dense_config if i < config.leading_dense_blocks else config
      if config.sliding_window_pattern != 0 and (i+1) % config.sliding_window_pattern == 0: c = replace(c, sliding_window=0)
      self.blk.append(GatedDeltaNetBlock(c, config.ssm) if config.ssm and config.ssm_layers[i] else block_cls(c))
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.embd_packed:tuple[Tensor, int]|None = None # (packed token_embd rows as u32 words, ggml type), set by from_gguf for quantized tables
    self.prefill_chunk = 32 # prompt tokens per prefill forward (more amortizes the weight streaming, the jit binds it as the toks range)
    self.extras: dict[str, Tensor] = {}  # tensors outside the LM (the clef decision head), see from_gguf
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    # a prefill chunk's matmuls run at its static max size: a prompt's last partial chunk goes through a smaller tail jit instead of paying
    # for a whole padded chunk (when the chunk is bigger than PREFILL_TAIL)
    self.prefill_tail_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)
    # MTP speculative decoding: one jit per draft step and one for the verify window (generate_mtp)
    self.mtp_heads:list[MTPHead] = [MTPHead(config) for _ in range(config.num_mtp_heads)]
    self._mtp_cache:tuple|None = None # (K, commit jits per accept length, draft jits per step, verify jit)

  def embed(self, tokens:Tensor) -> Tensor:
    if self.embd_packed is None: return self.token_embd(tokens.to(self.token_embd.weight.device)).float()
    (table, ggml_type), dim = self.embd_packed, int(self.token_embd.weight.shape[1])
    flat = tokens.to(table.device).reshape(-1).cast(dtypes.int32)
    n, padded = flat.shape[0], flat.pad_to(flat.max_shape).contiguous() # symbolic token count: gather the max, slice the garbage off
    m = int(padded.shape[0])
    rows = Tensor.empty(m, table.shape[1], dtype=dtypes.uint32, device=table.device)
    rows = Tensor.custom_kernel(rows, table, padded, fxn=_embedding_rows_kernel)[0]
    x = ggml_data_to_tensor(rows.bitcast(dtypes.uint8).flatten(), m * dim, ggml_type).reshape(m, dim)
    return x[:n].reshape(*tokens.shape, dim).float()

  def forward_hidden(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    """final-norm hidden states of every token (B, T, D), for models that consume the hidden states instead of sampling (clef)"""
    clear_activation_memos()
    x = self.embed(tokens)  # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    return self.output_norm(x)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    x = self._run_blocks(tokens, start_pos)
    # only run the output projection on the last token
    logits = self.output(self.output_norm(x[:, -1:]))[:, -1, :].to(tokens.device)
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def _run_blocks(self, tokens:Tensor, start_pos:int|UOp, verify:bool=False) -> Tensor:
    clear_activation_memos()  # per-forward memo of shared q8 activation quantizations (kernels/amd.py q8_quantize)
    x = self.embed(tokens)  # (B, T, D)
    for block in self.blk: x = block(x, start_pos, verify=verify)
    return x

  @staticmethod
  def _sample(logits:Tensor, temperature:Tensor) -> Tensor: # Gumbel-max
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1)

  def forward_verify(self, start_pos:int|UOp, temperature:Tensor, *window:Tensor) -> tuple[Tensor, Tensor]:
    """the MTP verify window [last committed, draft_0..draft_{K-1}] (each (1, 1) on the device): the main model's token at every position,
    and the pre-norm hidden states (they seed the next round's drafts). the outputs are copies: an intermediate buffer would not survive
    the next replay"""
    x = self._run_blocks(Tensor.cat(*window, dim=1).cast(dtypes.int32).contiguous(), start_pos, verify=True)
    return self._sample(self.output(self.output_norm(x)), temperature).cast(dtypes.int32).clone(), x.clone()

  def _mtp_draft_step(self, tok:Tensor, h_prev:Tensor, start_pos:int|UOp, temperature:Tensor) -> tuple[Tensor, Tensor]:
    clear_activation_memos()
    h = self.mtp_heads[0](h_prev.contiguous(), self.embed(tok.contiguous()), start_pos)
    return self._sample(self.output(self.mtp_heads[0].shared_head_norm(h))[:, -1:, :], temperature).cast(dtypes.int32).clone(), h.clone()

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0)), shard:int=1) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, entries = gguf_parse(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)
    arch = kv['general.architecture']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']
    assert shard >= 1, f"shard must be at least 1, got {shard}"
    shard_map:dict[str, int] = {}
    if shard > 1:
      heads = n_heads if kv.get(f'{arch}.attention.kv_lora_rank') else n_kv_heads
      assert heads % shard == 0, f"tensor parallel needs the attention heads to split over {shard} devices"
      # shard MLA heads and routed experts while replicating latent projections, KV cache and shared experts
      rules = {**{w: 0 for w in ('token_embd.weight', 'output.weight', 'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'ffn_gate.weight',
        'ffn_up.weight', 'attn_q_b.weight', 'attn_k_b.weight', 'attn_v_b.weight')},
        **{w: 1 for w in ('attn_output.weight', 'ffn_down.weight', 'ffn_gate_exps.weight', 'ffn_up_exps.weight')}, 'ffn_down_exps.weight':2}
      shard_map = {name: rules[k] for name in entries if (k:=re.sub(r"^blk\.\d+\.", "", name)) in rules}
    devices = tuple(Device.canonicalize(f'{Device.DEFAULT}:{i}') for i in range(shard))
    embd_type = entries['token_embd.weight'][2]
    # clef: the decision head and the output embedding are not LM weights. the head is decoded to float32 on the default device, the output
    # embedding stays packed: only the rows of a request's option tokens are read (and dequantized)
    extras: dict[str, Tensor] = {}
    if arch == 'clef':
      assert shard == 1, "clef does not support tensor parallel"
      for k in [k for k in entries if k.startswith(('dec.', 'decision.', 'token_types'))]:
        data, shape, typ = entries.pop(k)
        extras[k] = ggml_data_to_tensor(data.to(Device.DEFAULT), prod(shape), typ).reshape(shape).float()
      data, (vocab, dim), typ = entries.pop('output.weight')
      extras['output.raw'], extras['output.ggml_type'] = data.to(Device.DEFAULT).reshape(vocab, ggml_nbytes(dim, typ)).contiguous(), Tensor([typ])
      entries['output.weight'] = (Tensor.zeros(2, dtype=dtypes.uint8), (1, 1), 24)  # stub: nothing samples tokens
    state_dict = gguf_shard(entries, devices, shard_map)
    # a quantized embedding table: keep a view of its packed rows, the lookup gathers and dequantizes only those (the whole dequantized
    # table would otherwise be materialized in every JIT)
    embd_raw = next((u for u in state_dict['token_embd.weight'].uop.toposort() if u.op is Ops.BUFFER and u.dtype == dtypes.uint8), None)

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']

    ssm = None
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe', 'clef'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(kv[f'{arch}.block_count']))
    elif arch == 'kimi-linear':
      ssm_layers = tuple(x == 0 for x in n_kv_heads)
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], n_heads, n_heads, n_heads*kv[f'{arch}.kda.head_dim'], kda=True)
      for i, is_ssm in enumerate(ssm_layers):
        if not is_ssm: continue
        state_dict[f"blk.{i}.attn_qkv.weight"] = state_dict.pop(f"blk.{i}.attn_q.weight").cat(
          state_dict.pop(f"blk.{i}.attn_k.weight"), state_dict.pop(f"blk.{i}.attn_v.weight"), dim=0).contiguous()
        state_dict[f"blk.{i}.ssm_conv1d.weight"] = state_dict.pop(f"blk.{i}.ssm_conv1d_q.weight").cat(
          state_dict.pop(f"blk.{i}.ssm_conv1d_k.weight"), state_dict.pop(f"blk.{i}.ssm_conv1d_v.weight"), dim=0).squeeze(1).contiguous()
        state_dict[f"blk.{i}.ssm_out.weight"] = state_dict.pop(f"blk.{i}.attn_output.weight")
    if arch in ('qwen35', 'qwen35moe', 'glm4moe', 'gpt-oss', 'clef'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}
    # MTP (nextn) blocks trail the main ones: blk.{main+k}.nextn.{enorm,hnorm,eh_proj,shared_head_norm} -> mtp_heads.{k}.*, the rest of
    # blk.{main+k} -> mtp_heads.{k}.block.*. its embed_tokens is the main token_embd
    num_mtp = kv.get(f'{arch}.nextn_predict_layers', 0)
    main_blocks, mtp_ssm_layer = kv[f'{arch}.block_count'] - num_mtp, False
    for i in range(num_mtp):
      pre = f'blk.{main_blocks+i}.'
      state_dict.pop(pre+'embed_tokens.weight', None)
      mtp_ssm_layer = pre+'attn_qkv.weight' in state_dict
      for name in [n for n in state_dict if n.startswith(pre)]:
        suffix = name[len(pre):]
        new = f'mtp_heads.{i}.{suffix[len("nextn."):]}' if suffix.startswith('nextn.') else f'mtp_heads.{i}.block.{suffix}'
        state_dict[new] = state_dict.pop(name)

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)
    yarn = YaRNConfig(factor=kv[f'{arch}.rope.scaling.factor'],
                      orig_ctx_len=kv.get(f'{arch}.rope.scaling.original_context_length', kv[f'{arch}.context_length']),
                      beta_fast=kv.get(f'{arch}.rope.scaling.yarn_beta_fast', 32.0),
                      beta_slow=kv.get(f'{arch}.rope.scaling.yarn_beta_slow', 1.0)) if kv.get(f'{arch}.rope.scaling.type') == 'yarn' else None

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if arch == 'kimi-linear': continue
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    config = TransformerConfig(
      num_blocks=main_blocks, dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'], rope_dim=rope_dim, yarn=yarn,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear')),
      expert_gating_func=ExpertGating(kv.get(f'{arch}.expert_gating_func',
        ExpertGating.SOFTMAX_WEIGHT if arch == 'gpt-oss' else ExpertGating.SOFTMAX)),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe', 'clef'), ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict,
      expert_proj_bias='blk.0.ffn_gate_exps.bias' in state_dict, attn_output_bias='blk.0.attn_output.bias' in state_dict,
      swiglu_alpha=1.702 if arch == 'gpt-oss' else 1.0, swiglu_clamp_exp=7.0 if arch == 'gpt-oss' else None,
      swiglu_up_bias=1.0 if arch == 'gpt-oss' else 0.0, attn_sinks='blk.0.attn_sinks.weight' in state_dict,
      sliding_window=kv.get(f'{arch}.attention.sliding_window', 0),
      sliding_window_pattern=kv.get(f'{arch}.attention.sliding_window_pattern', 2 if arch == 'gpt-oss' else 0),
      num_mtp_heads=num_mtp, mtp_ssm_layer=mtp_ssm_layer)
    model = Transformer(config)
    for p in (nn.state.get_parameters(model) if shard > 1 else []): p.to_(devices)
    if extras: model.output = Linear(1, 1, bias=False)  # matches the output.weight stub above
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    # last: get_state_dict (load_state_dict, linears) would take it for a weight
    if shard == 1 and embd_raw is not None and embd_raw.max_numel() % (4 * (vocab:=config.vocab_size)) == 0:
      model.embd_packed = (Tensor(embd_raw).bitcast(dtypes.uint32).reshape(vocab, -1).clone().realize(), embd_type) # one copy, as u32 words
    # constructing the model drew every weight from the rng (then replaced by the gguf ones): the counter is a long lazy add chain that
    # every later realize would walk (_apply_map_to_tensors). run it once
    Tensor.realize(*Tensor._device_rng_counters.values())
    model.extras = extras  # last: get_state_dict (load_state_dict, linears) would treat these as weights
    return model, kv

  def warmup(self):
    for _ in range(2): list(zip(range(2), self.generate([0])))
    if self.prefill_chunk > PREFILL_TAIL: # the full-chunk prefill jit too (a 1-token prompt only reaches the tail jit)
      for _ in range(2):
        self._cached_tokens = []
        list(zip(range(1), self.generate([0]*self.prefill_chunk)))
      self._cached_tokens = []

  def get_start_pos(self, tokens:list[int]) -> int:
    # recurrent state can't be partially reused after divergence: reuse it only when tokens extend the cached prefix
    if self.has_recurrent_block:
      return len(self._cached_tokens) if self._cached_tokens and len(self._cached_tokens) < len(tokens) \
        and tokens[:len(self._cached_tokens)] == self._cached_tokens else 0
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def generate(self, tokens:list[int], chunk_size:int|None=None, temperature:float=0.0):
    if chunk_size is None: chunk_size = self.prefill_chunk
    if self.has_recurrent_block and not gated_delta_kernel_supported(self.token_embd.weight.device): chunk_size = 1
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks, v_tail = UOp.variable("toks", 1, chunk_size), UOp.variable("tail_toks", 1, min(chunk_size, PREFILL_TAIL))
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      remaining = len(tokens) - start_pos
      tail = start_pos < prompt_len and remaining < chunk_size and chunk_size > PREFILL_TAIL
      n_toks = min(PREFILL_TAIL if tail else chunk_size, remaining)
      sp, nt = v_start_pos.bind(start_pos), (v_tail if tail else v_toks).bind(n_toks)
      if tail: out = self.prefill_tail_jit(t[:, sp:sp+nt].contiguous(), sp, temp).realize()
      else: out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
      start_pos += n_toks
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]

  def generate_mtp(self, tokens:list[int], K:int, chunk_size:int|None=None, temperature:float=0.0):
    """speculative decoding with the MTP head: draft K tokens in a chain, verify [last committed, drafts] in one forward, commit the
    longest accepted prefix plus the main model's next token. recurrent blocks verify on a copy of their state and commit_verify lands
    the accepted prefix (one jit per accept length); KV caches only ever get read up to the committed length"""
    assert K >= 1 and self.mtp_heads, "generate_mtp needs K >= 1 and a checkpoint with MTP (nextn) blocks"
    assert amd_custom_kernels_supported(self.token_embd.weight.device), "generate_mtp needs the AMD gated delta kernel"
    if chunk_size is None: chunk_size = self.prefill_chunk
    v_start_pos, v_toks = UOp.variable("start_pos", 0, self.max_context-1), UOp.variable("toks", 1, chunk_size)
    temp, prompt_len = Tensor([temperature]), len(tokens)
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # prefill all but the last prompt token: the verify window absorbs it as its position 0
    start_pos = self.get_start_pos(tokens)
    while start_pos < prompt_len - 1:
      n_toks = min(chunk_size, prompt_len - 1 - start_pos)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      self(t[:, sp:sp+nt], sp, temp).realize()
      start_pos += n_toks
    ssm_blocks = [b for b in self.blk if isinstance(b, GatedDeltaNetBlock)]
    if self._mtp_cache is None or self._mtp_cache[0] != K:
      # commit_verify needs accept as a python int (a static conv window slice): one jit per accept length. one draft jit per step: step
      # i reads step i-1's outputs, a jit never reads its own output buffer
      def commit_all(accept:int):
        if (stores:=[s for b in ssm_blocks for s in b.commit_verify(accept)]): Tensor.realize(*[Tensor(s) for s in stores])
      self._mtp_cache = (K, [TinyJit(functools.partial(commit_all, a)) for a in range(K+1)], [TinyJit(self._mtp_draft_step) for _ in range(K)],
                         TinyJit(self.forward_verify))
    _, commit_jits, draft_jits, verify_jit = self._mtp_cache
    # tokens and hidden states stay on the device: the drafts read the verify outputs through views at the bound accept position. the
    # first round reads seed buffers of the same shapes (the last prompt token, no hidden state yet: it only drafts worse)
    v_acc = UOp.variable("mtp_accept", 0, K)
    pred = Tensor([[tokens[-1]] * (K+1)], dtype=dtypes.int32).contiguous().realize()
    vhidden = Tensor.zeros(1, K+1, int(self.token_embd.weight.shape[1])).contiguous().realize()
    accept, accept_hist = 0, [0] * (K+1)
    try:
      while len(tokens) < self.max_context - K - 1:
        a = v_acc.bind(accept)
        last, tok, h = pred[:, a:a+1], pred[:, a:a+1], vhidden[:, a:a+1]
        drafts:list[Tensor] = []
        for i in range(K):
          tok, h = draft_jits[i](tok, h, v_start_pos.bind(start_pos+i), temp)
          drafts.append(tok)
        pred, vhidden = verify_jit(v_start_pos.bind(start_pos), temp, last, *drafts)
        preds, dvals = cast(list[int], pred.reshape(K+1).tolist()), [int(d.item()) for d in drafts]
        accept = 0
        while accept < K and preds[accept] == dvals[accept]: accept += 1
        committed = dvals[:accept] + [preds[accept]]
        accept_hist[accept] += 1
        if ssm_blocks: commit_jits[accept]()
        tokens.extend(committed)
        start_pos += accept + 1
        self._cached_tokens = tokens[:-1]
        yield from committed
    finally:
      if (n:=sum(accept_hist)) and DEBUG >= 1:
        print(f"mtp accept_hist={accept_hist} mean_accept={sum(i*c for i, c in enumerate(accept_hist))/n:.3f} rounds={n}", flush=True)
