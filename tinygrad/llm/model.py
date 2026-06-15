from __future__ import annotations
import functools, itertools, pathlib, re, time, collections
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function
from tinygrad.llm.gguf import gguf_load
from tinygrad.uop.ops import resolve

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

class ExpertWeights:
  """Like nn.Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)

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
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  full_attention_interval: int = 0
  attn_output_gate: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False
  num_mtp_heads: int = 0

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = nn.Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = nn.Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = nn.Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = nn.Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = nn.Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      if hasattr(self, 'exp_probs_b'):
        probs = logits.sigmoid()
        _, sel = pairwise_topk(probs + self.exp_probs_b["bias"], self.config.num_experts_per_tok)
        probs = probs.gather(-1, sel)
        if self.config.norm_topk_prob: probs = probs / probs.sum(axis=-1, keepdim=True)
      else:
        vals, sel = pairwise_topk(logits, self.config.num_experts_per_tok)
        probs = vals.softmax(-1) if self.config.norm_topk_prob else logits.softmax(-1).gather(-1, sel)
      probs = probs * self.config.routed_scaling_factor
      x_down = self.ffn_down_exps(sel, (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous())  # (B, T, k, D)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x).silu().contiguous() * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    if getenv("CUSTOM_MLP") and x.device == "AMD" and self.config.dim == 1024 and self.config.hidden_dim == 3584:
      from tinygrad.llm.amd_kernels import fused_gate_up
      return self.ffn_down(fused_gate_up(x, self.ffn_gate.weight, self.ffn_up.weight))
    # For T>1 (MTP verify) flatten before the FFN matmuls so the scheduler sees one batched
    # matmul instead of T separate per-position matmuls.
    B, T, D = x.shape
    if resolve(T != 1):
      xf = x.reshape(B*T, D)
      h = (self.ffn_gate(xf).silu().contiguous() * self.ffn_up(xf))
      return self.ffn_down(h).reshape(B, T, -1)
    # TODO: remove the need for this contiguous
    return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  # return writes that reset this block's state after a cache mismatch
  def _state_reset_ops(self) -> list[Tensor]: return []
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    if resolve(x.shape[1] != 1):
      # T>1 (MTP verify): hoist attn_norm/ffn_norm out of the @function precompile barrier so the
      # attn_q/attn_output/ffn_* matmuls compile without the per-row RMSnorm reduction pinning BEAM
      # to per-T-row dispatch. +15% real on the verify path (was Update 9's gated MTP_SPLIT_NORM).
      x_normed = self.attn_norm(x).contiguous()
      @function(precompile=True, allow_implicit=True)
      def _run_attn(x:Tensor, x_normed:Tensor, start_pos:int|UOp):
        return x + self._attention(x_normed, start_pos)
      h = _run_attn(x, x_normed, start_pos)
      h_normed = self.ffn_norm(h).contiguous()
      @function(precompile=True, allow_implicit=True)
      def _run_ffn(h:Tensor, h_normed:Tensor):
        return (h + self._feed_forward(h_normed)).contiguous()
      return _run_ffn(h, h_normed)
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp):
      h =     x + self._attention(self.attn_norm(x), start_pos)
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return _run(x, start_pos)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = nn.Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = nn.Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = nn.Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = nn.Linear(config.head_dim * config.n_heads, config.dim, bias=False)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    # For T>1 (MTP verify) flatten the batch+T dims before the projections so the scheduler sees
    # one (T, D) @ (D, D_out) matmul instead of T separate per-position matmuls. Safe to skip
    # for the T=1 hot path (rollout) where the shape signatures are already well-tuned.
    x_proj = x.reshape(B*T, -1) if resolve(T != 1) else x
    q, k, v = self.attn_q(x_proj), self.attn_k(x_proj), self.attn_v(x_proj)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

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
    assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
    k = assigned_kv[0, :, :, 0:start_pos+T, :]
    v = assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    out_in = attn if not self.config.attn_output_gate else (attn * gate.sigmoid())
    # Same T-batching trick on the way out so attn_output also runs as one matmul.
    if resolve(T != 1):
      return self.attn_output(out_in.reshape(B*T, -1)).reshape(B, T, -1)
    return self.attn_output(out_in)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # zeros (not empty): MTP head blocks are never prefilled, so they read at positions that were
      # never written; uninitialized memory there leaks NaN. Main blocks always overwrite before read.
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim, device=x.device).contiguous().realize()
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = nn.Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = nn.Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = nn.Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = nn.Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T]), dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = apply_rope(
      kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2),
      self.freqs_cis[start_pos:start_pos+T])

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
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv, self.attn_gate = nn.Linear(config.dim, self.conv_channels, bias=False), nn.Linear(config.dim, ssm.inner_size, bias=False)
    self.ssm_alpha, self.ssm_beta = nn.Linear(config.dim, self.num_v_heads, bias=False), nn.Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), nn.Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    if resolve(T != 1): return self._attention_tn(x)

    # input processing
    x = x.half()
    out_gate = self.attn_gate(x).reshape(B, 1, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, self.num_v_heads, 1, 1)
    alpha = ((self.ssm_alpha(x).float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).reshape(B, self.num_v_heads, 1, 1).exp()

    # qkv conv
    conv_window = self.conv_state.cat(self.attn_qkv(x), dim=1)
    conv_out = (conv_window * self.ssm_conv1d["weight"].T.unsqueeze(0)).sum(1).silu()
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q = q.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    k = k.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    v = v.reshape(B, self.num_v_heads, self.head_v_dim)
    q, k, v = q.mul(self.head_k_dim**-0.5).unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)

    # store the updated state
    if getenv("CUSTOM_GDN") and x.device == "AMD" and B == 1 and self.num_v_heads == 16 and self.head_v_dim == 128 and self.head_k_dim == 128:
      from tinygrad.llm.amd_kernels import gdn_recurrent_update_conv
      core_attn_in = gdn_recurrent_update_conv(self.recurrent_state, conv_out, alpha, beta)
      conv_state_store = self.conv_state.uop.after(core_attn_in.uop).store(conv_window[:, 1:, :].cast(self.conv_state.dtype).uop)
      core_attn_in = Tensor(core_attn_in.uop.after(conv_state_store))
    else:
      conv_state_store = self.conv_state.uop.store(conv_window[:, 1:, :].cast(self.conv_state.dtype).uop)
      # recurrent
      recurrent_state = self.recurrent_state * alpha
      recurrent_state = recurrent_state + ((v - recurrent_state@k) * beta)@k.transpose(-1, -2)
      recurrent_state_store = self.recurrent_state.uop.store(recurrent_state.cast(self.recurrent_state.dtype).uop)
      recurrent_state = Tensor(self.recurrent_state.uop.after(recurrent_state_store, conv_state_store))
      core_attn_in = (recurrent_state@q).squeeze(-1).reshape(B, 1, self.num_v_heads, self.head_v_dim)

    # output
    core_attn_out = self.ssm_norm(core_attn_in)
    return self.ssm_out((core_attn_out * out_gate.silu()).reshape(B, 1, -1).cast(x.dtype))

  def _attention_tn(self, x:Tensor) -> Tensor:
    """T>1 SSM forward used by MTP's K-wide verify pass.

    Projections, convolution, and per-position q/k/v are computed in one parallel matmul over
    all T inputs (bandwidth amortized). The DeltaNet recurrence itself
      `S_t = S_{t-1} * alpha_t + (v_t - S_{t-1} @ k_t) * beta_t @ k_t^T`
    is unrolled sequentially in-graph over T to keep every kernel in the same `(B,H,D,D)`
    shape signature as the baseline T=1 path. Per-position state is saved to `rs_stack[0..T-1]`
    / `cs_stack[0..T-1]` (stacked, shape (T, ...)). On the NEXT iter's verify, the starting
    state is read from `rs_stack[mtp_accept]` via a bound `UOp.variable` so partial-accept
    rollback is free — no explicit SSM restore between iters.

    NOTE: An earlier version used a Hillis-Steele parallel scan, but the resulting
    `(B,T,H,D,D)` 5D-tensor kernels compiled to suboptimal schedules. The sequential-in-graph
    form here is mathematically equivalent at small T (K=2–8) with negligible compute overhead
    and produces kernel shapes the JIT already schedules well.
    """
    B, T, _ = x.shape
    D = self.head_v_dim
    assert D == self.head_k_dim, "T>1 SSM path requires head_k_dim == head_v_dim"
    H = self.num_v_heads
    ratio = H // self.num_k_heads
    K = self.ssm_conv_kernel

    x = x.half()
    # EXPERIMENT: feed projections (T, D) instead of (B=1, T, D) so the scheduler sees a single
    # T-batched matmul rather than treating each T position as its own (1, D) row. This is meant
    # to attack the per-T kernel inflation seen in DEBUG=4 profiling (same r_* kernels firing 3x).
    x_flat = x.reshape(B*T, -1)                                                            # (T, D)
    out_gate = self.attn_gate(x_flat).reshape(B, T, H, D)                                 # (B,T,H,D)
    beta  = self.ssm_beta(x_flat).sigmoid().reshape(B, T, H).float()                     # (B,T,H)
    alpha = ((self.ssm_alpha(x_flat).float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).reshape(B, T, H).exp()  # (B,T,H)

    # --- starting state: read from rs_stack[mtp_accept] (Variable-indexed) when MTP is active.
    # First verify of the run seeds rs_stack[T-1] from recurrent_state (see Transformer.generate_mtp)
    # and binds mtp_accept = T-1, so this path also covers the cold case.
    mtp_accept = getattr(self, '_mtp_accept_uop', None)
    if mtp_accept is not None and self.rs_stack is not None:
      conv_start = self.cs_stack[mtp_accept:mtp_accept+1].reshape(*self.conv_state.shape)
      S0 = self.rs_stack[mtp_accept:mtp_accept+1].reshape(*self.recurrent_state.shape).float()
    else:
      conv_start = self.conv_state
      S0 = self.recurrent_state.float()                                                    # (B,H,D,D)

    # --- depthwise conv over the (K-1 prior + T new) qkv window, producing T outputs ---
    conv_window = conv_start.cat(self.attn_qkv(x_flat).reshape(B, T, -1), dim=1)         # (B, K-1+T, C)
    weight = self.ssm_conv1d["weight"].T.reshape(1, 1, K, -1)                             # (1,1,K,C)
    # Build T sliding windows of length K via Python-level stack (T is a concrete int at trace time).
    windows = Tensor.stack(*[conv_window[:, t:t+K, :] for t in range(T)], dim=1)          # (B,T,K,C)
    conv_out = (windows * weight).sum(axis=2).silu()                                       # (B,T,C)

    q_part, k_part, v_part = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q = q_part.reshape(B, T, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, 1, ratio, 1)
    k = k_part.reshape(B, T, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, 1, ratio, 1)
    v = v_part.reshape(B, T, H, self.head_v_dim)
    q = q.mul(self.head_k_dim ** -0.5).unsqueeze(-1).float()                              # (B,T,H,D,1)
    k = k.unsqueeze(-1).float()
    v = v.unsqueeze(-1).float()

    # --- recurrent SSM update: sequential-in-graph over T (see class docstring NOTE) ---
    S = S0
    S_list:list[Tensor] = []
    core_list:list[Tensor] = []
    for t in range(T):
      alpha_t = alpha[:, t].reshape(B, H, 1, 1)
      beta_t  = beta[:, t].reshape(B, H, 1, 1)
      q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]                                           # each (B,H,D,1)
      S = S * alpha_t
      S = S + ((v_t - S @ k_t) * beta_t) @ k_t.transpose(-1, -2)
      S_list.append(S)
      core_list.append((S @ q_t).squeeze(-1))                                             # (B,H,D)
    # Persist final state (used by non-MTP T=1 paths) AND ALL T per-position states into rs_stack /
    # cs_stack (used by next iter's verify as the variable-indexed starting state).
    final_S = S_list[-1]                                                                  # (B,H,D,D)
    final_conv = conv_window[:, T:, :]                                                    # (B,K-1,C)
    recurrent_state_store = self.recurrent_state.uop.store(final_S.cast(self.recurrent_state.dtype).uop)
    conv_state_store     = self.conv_state.uop.store(final_conv.cast(self.conv_state.dtype).uop)
    stores = [recurrent_state_store, conv_state_store]
    if self.rs_stack is not None and self._mtp_accept_uop is not None:
      # Verify-mode only (mtp_accept bound). During chunked PREFILL this path also runs at T>1 but
      # with _mtp_accept_uop=None: we then read/write only the live conv_state/recurrent_state above,
      # never the verify stack (whose T=K+1 sizing wouldn't match a prefill chunk's T anyway).
      # Store all T positions; accept = j picks rs_stack[j] as next iter's starting state, j ∈ [0, T-1].
      for t in range(T):
        stores.append(self.rs_stack[t].uop.store(S_list[t].cast(self.rs_stack.dtype).uop))
        slice_t = conv_window[:, t+1:t+self.ssm_conv_kernel, :]
        stores.append(self.cs_stack[t].uop.store(slice_t.cast(self.cs_stack.dtype).uop))
    # Read final_S back via .after(stores) and use it for the last position's output, threading
    # the store side effects through the returned core_attn_in.
    post_state = Tensor(self.recurrent_state.uop.after(*stores)).float()                  # (B,H,D,D)
    core_list[-1] = (post_state @ q[:, -1]).squeeze(-1)                                   # (B,H,D)
    # .contiguous() after stack forces the scheduler to materialize core_attn_in as a single
    # (B,T,H,D) tensor instead of T disjoint sub-graphs. Without it, downstream ops (ssm_norm,
    # ssm_out, the residual add into FFN) inherit T separate sub-graphs and fire per-T kernels
    # instead of one T-batched kernel.
    core_attn_in = Tensor.stack(*core_list, dim=1).contiguous()                           # (B,T,H,D)

    core_attn_out = self.ssm_norm(core_attn_in)
    return self.ssm_out((core_attn_out * out_gate.silu()).reshape(B, T, -1).cast(x.dtype)).contiguous()

  # recurrent state can't be partially reused after divergence, force a full rebuild
  def _state_reset_ops(self):
    return [self.conv_state.assign(self.conv_state.const_like(0)),
            self.recurrent_state.assign(self.recurrent_state.const_like(0))] if hasattr(self, "conv_state") else []
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return 0 if prefix_len != cached_len else prefix_len

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_v_dim, device=x.device).clone()
      # Per-position state history (rs_stack[t], cs_stack[t] for t in 0..T-1, T = verify_T = K+1) used
      # by MTP speculative decoding. Index `mtp_accept` via a bound UOp.variable so partial-accept
      # rollback is free — the next iter's verify reads its starting state from rs_stack[accept]
      # directly, no explicit restore copy between iters.
      self.rs_stack:Tensor|None = None
      self.cs_stack:Tensor|None = None
      self._mtp_accept_uop:UOp|None = None

  def _alloc_history(self, T:int):
    # T = verify_T = K+1. rs_stack[0..T-1] / cs_stack[0..T-1] cover all accept ∈ [0, T-1].
    if self.rs_stack is None or self.rs_stack.shape[0] != T:
      self.rs_stack = Tensor.zeros(T, *self.recurrent_state.shape, dtype='float32', device=self.recurrent_state.device).contiguous().realize()
      self.cs_stack = Tensor.zeros(T, *self.conv_state.shape, dtype=self.conv_state.dtype, device=self.conv_state.device).contiguous().realize()

  def _seed_history_from_state(self, j:int) -> list[Tensor]:
    """Copy current recurrent_state / conv_state into rs_stack[j] / cs_stack[j] — used to seed the
    first verify of a run (no prior verify to populate the stack)."""
    return [
      self.rs_stack[j].assign(self.recurrent_state.cast(self.rs_stack.dtype)),
      self.cs_stack[j].assign(self.conv_state.cast(self.cs_stack.dtype)),
    ]

class MTPHead:
  """Qwen/DeepSeek-style Multi-Token Prediction head.

  Takes the previous-step hidden state `h_prev` and the embedding of the just-sampled
  next token `tok_embed` (both (B,1,D)), fuses them via eh_proj([hnorm(h_prev), enorm(tok_embed)]),
  then runs a single transformer block. Returns the new hidden state (B,1,D).

  The shared output_norm + lm_head from the main Transformer are reused for sampling
  rather than duplicated; any dedicated `shared_head.*` weights in the GGUF are ignored.
  """
  def __init__(self, config:TransformerConfig):
    self.config = config
    self.enorm = nn.RMSNorm(config.dim, config.norm_eps)
    self.hnorm = nn.RMSNorm(config.dim, config.norm_eps)
    self.eh_proj = nn.Linear(2 * config.dim, config.dim, bias=False)
    self.block = TransformerBlock(config)
    # MTP-specific final norm before the (shared) lm_head. NOT tied to main output_norm.
    self.shared_head_norm = nn.RMSNorm(config.dim, config.norm_eps)

  def __call__(self, h_prev:Tensor, tok_embed:Tensor, start_pos:int|UOp) -> Tensor:
    # Concat order: [enorm(embed); hnorm(h_prev)] per Qwen3 MTP convention (opposite of DeepSeek-V3).
    fused = self.eh_proj(self.enorm(tok_embed).cat(self.hnorm(h_prev), dim=-1))
    return self.block(fused, start_pos)

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(config, config.ssm) if config.ssm and (i+1) % config.full_attention_interval != 0 else
                               block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # MTP heads (dense Qwen/DeepSeek-style next-token-prediction heads, run after the main stack)
    dense_mtp_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0,
                               hidden_dim=config.dense_hidden_dim or config.hidden_dim, ssm=None)
    self.mtp_heads:list[MTPHead] = [MTPHead(dense_mtp_config) for _ in range(config.num_mtp_heads)]
    # side-effect buffer holding the latest MTP-chain hidden state; allocated lazily on first MTP call
    # so it lives on the same device as the input. Float32 to match _body's output dtype.
    self._mtp_h_buf:Tensor|None = None
    # per-position hidden state from verify pass (shape (1, K+1, D)); used to refresh _mtp_h_buf
    # at any accepted verify position without an extra main forward.
    self._verify_h_buf:Tensor|None = None
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)
    # extra JITs for MTP path
    self.mtp_main_jit = TinyJit(self._forward_with_hidden)        # main step, hidden written to self._mtp_h_buf
    self.mtp_draft_jits = [TinyJit(self._make_mtp_step(i)) for i in range(len(self.mtp_heads))]
    # first draft of each iter reads h_prev from self._verify_h_buf[seed_accept] (a bound UOp.variable)
    # instead of self._mtp_h_buf, so the per-iter seed copy+realize (and its per-accept static-slice
    # kernels) is eliminated — the seed read fuses into the draft forward. See generate_mtp.
    self.mtp_first_draft_jit = TinyJit(self._make_mtp_first_step(0)) if self.mtp_heads else None
    self.mtp_verify_jit = TinyJit(self._verify_forward)           # verify step: T=k, returns per-pos samples

  def _body(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens).float()                   # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    return x  # pre-output-norm hidden state

  def _gumbel_argmax(self, logits:Tensor, temperature:Tensor) -> Tensor:
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) ~ sample from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    x = self.output_norm(self._body(tokens, start_pos))
    if getenv("CUSTOM_VOCAB_ARGMAX") and x.device == "AMD" and x.shape[0] == 1 and x.shape[-1] == 1024 and self.output.weight.shape[0] == 248320:
      from tinygrad.llm.amd_kernels import q8_lmhead_gumbel_argmax
      return q8_lmhead_gumbel_argmax(x[:, -1, :], self.output.weight, temperature)
    return self._gumbel_argmax(self.output(x)[:, -1, :], temperature)

  def _forward_with_hidden(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    """Main-model step that writes the last-position pre-norm hidden into self._mtp_h_buf
    (single owned buffer) and returns the sampled token. The .assign() side-effect mirrors
    nn.Optimizer's pattern so the tensor binding doesn't leak across JIT invocations."""
    h = self._body(tokens, start_pos)                                       # (B, T, D)
    last_h = h[:, -1:, :].contiguous()                                      # (B, 1, D)
    self._mtp_h_buf.assign(last_h.cast(self._mtp_h_buf.dtype))
    return self._gumbel_argmax(self.output(self.output_norm(h))[:, -1, :], temperature)

  def _verify_forward(self, tokens:Tensor, start_pos:int|UOp, mtp_accept:UOp, temperature:Tensor) -> Tensor:
    """Run main model over T tokens, returning per-position samples (B, T, 1). Also writes the
    per-position pre-norm hidden state to self._verify_h_buf so generate_mtp can seed _mtp_h_buf
    at whichever verify position the accept rolls back to. `mtp_accept` is a bound UOp.variable
    that SSM blocks consume (stashed below) to pick which rs_stack slot to read as the starting
    recurrent state — eliminating explicit per-iter restore copies."""
    # Plumb the BOUND mtp_accept to SSM blocks for this trace. Using the arg (not the closure-stored
    # unbound variable) ensures the binding flows through the JIT's arg-time var_vals collection.
    for blk in self.blk:
      if isinstance(blk, GatedDeltaNetBlock): blk._mtp_accept_uop = mtp_accept
    h = self._body(tokens, start_pos)                                                     # (B, T, D)
    h_store = self._verify_h_buf.uop.store(h.cast(self._verify_h_buf.dtype).uop)
    # Read post-store hidden and use it for the lm_head — guarantees the store side-effect fires.
    h_post = Tensor(self._verify_h_buf.uop.after(h_store)).cast(h.dtype)
    return self._gumbel_argmax(self.output(self.output_norm(h_post)), temperature)

  def _make_mtp_step(self, head_idx:int):
    """Returns a jit-friendly closure that runs one step of MTP head `head_idx`.
    Writes the new hidden into self._mtp_h_buf (single owned buffer) as a side effect,
    and returns only the sampled token. This avoids leaking per-call variable bindings
    through a tuple-returned hidden tensor between JIT invocations."""
    def _step(head_h_prev:Tensor, tok:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
      tok_embed = self.token_embd(tok).float()
      h = self.mtp_heads[head_idx](head_h_prev, tok_embed, start_pos)       # (1, 1, D)
      self._mtp_h_buf.assign(h.cast(self._mtp_h_buf.dtype))
      # MTP-specific final norm (not main output_norm); shares lm_head with main.
      return self._gumbel_argmax(self.output(self.mtp_heads[head_idx].shared_head_norm(h))[:, -1, :], temperature)
    return _step

  def _make_mtp_first_step(self, head_idx:int):
    """Like _make_mtp_step but reads h_prev from self._verify_h_buf[seed_accept] (bound UOp.variable)
    rather than self._mtp_h_buf. The previous iter's verify wrote per-position hiddens into
    _verify_h_buf; seed_accept selects the accepted position. This fuses the _mtp_h_buf seed into the
    first draft's forward — no per-iter assign+realize, and one symbolic-indexed kernel instead of a
    distinct static-slice kernel per accept value. Still writes _mtp_h_buf so later drafts chain off it."""
    def _step(verify_h_buf:Tensor, seed_accept:UOp, tok:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
      head_h_prev = verify_h_buf[:, seed_accept:seed_accept+1, :]           # (1, 1, D), symbolic slot
      tok_embed = self.token_embd(tok).float()
      h = self.mtp_heads[head_idx](head_h_prev, tok_embed, start_pos)       # (1, 1, D)
      self._mtp_h_buf.assign(h.cast(self._mtp_h_buf.dtype))
      return self._gumbel_argmax(self.output(self.mtp_heads[head_idx].shared_head_norm(h))[:, -1, :], temperature)
    return _step

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, state_dict = gguf_load(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Remap MTP (Multi-Token Prediction / nextn) block tensors. The trailing `nextn` blocks carry
    # MTP-specific tensors under a `nextn.` infix alongside a standard transformer block.
    # Observed layout (Qwen3.6 MTP): blk.{i}.nextn.{enorm,hnorm,eh_proj,shared_head_norm}.weight + blk.{i}.{attn_*, ffn_*, *_norm}.
    nextn = kv.get(f'{arch}.nextn_predict_layers', 0)
    if nextn:
      main_blocks = kv[f'{arch}.block_count'] - nextn
      for name in list(state_dict.keys()):
        m = re.match(r'blk\.(\d+)\.(.*)', name)
        if not m: continue
        bi = int(m.group(1))
        if bi < main_blocks: continue
        head_idx, rest = bi - main_blocks, m.group(2)
        if rest in ('nextn.enorm.weight', 'nextn.hnorm.weight', 'nextn.eh_proj.weight'):
          state_dict[f'mtp_heads.{head_idx}.{rest.split(".",1)[1]}'] = state_dict.pop(name)
        elif rest in ('nextn.shared_head_norm.weight', 'nextn.shared_head.norm.weight', 'shared_head.norm.weight'):
          # NOT tied to main output_norm — distinct weights. Load into per-head shared_head_norm.
          state_dict[f'mtp_heads.{head_idx}.shared_head_norm.weight'] = state_dict.pop(name)
        elif rest in ('nextn.shared_head.head.weight', 'nextn.embed_tokens.weight',
                      'embed_tokens.weight', 'shared_head.head.weight'):
          # tied to main token_embd / lm_head — drop the dedicated copy
          del state_dict[name]
        else:
          # standard transformer-block weights live under .block.*
          state_dict[f'mtp_heads.{head_idx}.block.{rest}'] = state_dict.pop(name)

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
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
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe')),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      full_attention_interval=kv.get(f'{arch}.full_attention_interval', 0),
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict,
      num_mtp_heads=kv.get(f'{arch}.nextn_predict_layers', 0))
    model = Transformer(config)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def get_start_pos(self, tokens:list[int]) -> int:
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def _prefill_chunked(self, t:Tensor, start_pos:int, prompt_len:int, v_start_pos:UOp, chunk:int,
                       temp:Tensor, seed_mtp:bool) -> tuple[Tensor, int]:
    """Chunked prefill for hybrid SSM/attention models.

    Each forward processes a CONCRETE-T slice `t[:, start_pos:start_pos+nt]` (symbolic start_pos,
    Python-int length) so GatedDeltaNetBlock._attention_tn — which Python-iterates its conv window
    and recurrence over T — gets a concrete T at trace time, while full-attention blocks T-batch.
    A fixed `chunk` is used while a full chunk fits; the remainder runs as a T=1 tail (reusing the
    rollout T=1 SSM path). The forward that consumes the final prompt position uses mtp_main_jit
    when `seed_mtp` so it seeds self._mtp_h_buf. Returns (final-forward output, new start_pos)."""
    out_main = None
    while start_pos < prompt_len:
      nt = chunk if prompt_len - start_pos >= chunk else 1
      sp = v_start_pos.bind(start_pos)
      if seed_mtp and start_pos + nt == prompt_len:
        out_main = self.mtp_main_jit(t[:, sp:sp+nt].contiguous(), sp, temp).realize()
      else:
        out_main = self(t[:, sp:sp+nt], sp, temp).realize()
      start_pos += nt
    return out_main, start_pos

  def generate_mtp(self, tokens:list[int], k:int=2, chunk_size:int=32, temperature:float=0.0):
    """Speculative decoding using the MTP heads to draft k tokens per main-model step.

    Single-batch only. On accept-length j (0 <= j <= k), this iteration commits j+1 tokens via
    one verify forward (T=K+1) plus k MTP-draft forwards. SSM state is rolled back in O(1)
    via per-position history saved during the verify pass. KV cache rollback is implicit —
    next iteration's writes overwrite stale slots.

    NOTE: Correctness has been verified, but current performance is worse than baseline non-MTP
    decoding on hybrid SSM/attention models. The verify forward (T>1) generates new kernel
    shapes that tinygrad's JIT scheduler hasn't tuned as well as the well-trodden T=1 path.
    Closing the gap is a kernel-layer task, not a Python-level one.
    """
    assert self.mtp_heads, "model has no MTP heads loaded"
    assert k >= 1
    # GatedDeltaNet._attention_tn Python-iterates conv windows, so its T must be a concrete
    # int at JIT-trace time. Verify is fixed at T=k+1. Prefill CAN run CONCRETE-T chunks
    # (PREFILL_CHUNK>1) through the same _attention_tn path + a T=1 tail, but this is OFF by default
    # (PREFILL_CHUNK=1 → token-at-a-time): on the 27B model a T=N forward does NOT amortize the
    # weight read across N (per-T-row dispatch), so chunking gave 0 speedup AND drifts the prefill
    # argmax vs the serial path. Kept opt-in for when the verify-kernel T-tiling work also covers
    # prefill shapes; see the doc's "de-fuse the attn_q epilogue" next step.
    prefill_chunk = getenv("PREFILL_CHUNK", 1) if self.has_recurrent_block else chunk_size
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_mtp_sp = UOp.variable("mtp_sp", 0, self.max_context-1)
    temp = Tensor(temperature).contiguous()
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)

    # allocate (or reallocate if device or k changed) the MTP buffers
    dim = self.blk[0].config.dim
    verify_T = k + 1  # verify processes [last_committed, drafts[0..k-1]] in one batched forward
    if self._mtp_h_buf is None or self._mtp_h_buf.device != t.device:
      self._mtp_h_buf = Tensor.zeros(1, 1, dim, device=t.device, dtype='float32').contiguous().realize()
    if self._verify_h_buf is None or self._verify_h_buf.device != t.device or self._verify_h_buf.shape[1] != verify_T:
      self._verify_h_buf = Tensor.zeros(1, verify_T, dim, device=t.device, dtype='float32').contiguous().realize()

    # ----- prefill (chunked) using the standard path; ensures main KV is populated -----
    start_pos = self.get_start_pos(tokens)
    if start_pos < len(self._cached_tokens) and (resets := [r for b in self.blk for r in b._state_reset_ops()]): Tensor.realize(*resets)
    # Clear any verify-mode flag left on SSM blocks by a prior generate_mtp call so prefill chunks
    # read live conv_state/recurrent_state (not the rs_stack) — see _attention_tn's accept guard.
    for blk in self.blk:
      if isinstance(blk, GatedDeltaNetBlock): blk._mtp_accept_uop = None
    prompt_len = len(tokens)
    out_main, start_pos = self._prefill_chunked(t, start_pos, prompt_len, v_start_pos, prefill_chunk, temp, seed_mtp=True)

    # commit the first sampled token (from the last prefill chunk)
    tokens.append(int(out_main.item()))
    self._cached_tokens = tokens[:-1]
    yield tokens[-1]
    # main KV reflects positions [0, prompt_len). start_pos == prompt_len.
    # self._mtp_h_buf holds the hidden at position prompt_len-1 used to predict tokens[-1].

    # Now that all SSM blocks have called _init_state during prefill, allocate the verify-position
    # state stack (size verify_T so accept ∈ [0, verify_T-1] = [0, K] all index valid slots) and
    # seed rs_stack[K] from the post-prefill recurrent_state. The mtp_accept UOp is bound at each
    # verify call site and stashed on blocks inside _verify_forward (see the body there).
    v_mtp_accept = UOp.variable("mtp_accept", 0, k)
    ssm_blocks:list[GatedDeltaNetBlock] = [blk for blk in self.blk if isinstance(blk, GatedDeltaNetBlock)]
    # Pre-stash the unbound variable so it's referenced in the JIT trace (the BOUND version is also
    # stashed inside _verify_forward; both reference the same Variable so trace bookkeeping aligns).
    seed_ops:list[Tensor] = []
    for blk in ssm_blocks:
      blk._alloc_history(verify_T)
      blk._mtp_accept_uop = v_mtp_accept
      seed_ops.extend(blk._seed_history_from_state(k))   # rs_stack[K] = current recurrent_state
    if seed_ops: Tensor.realize(*seed_ops)

    # MTP heads share absolute RoPE position with the main sequence — initialize to the post-prefill position
    # and advance by the number of committed tokens each iter (accept drafts + 1 verify-bonus).
    mtp_start_pos = start_pos
    last_committed = tokens[-1]
    # head_h is a stable view of the side-effect buffer; the JIT writes into it each step.
    head_h_view = self._mtp_h_buf
    # First draft reads h_prev from _verify_h_buf[seed_accept] (bound below). Seed slot K from the
    # prefill hidden and start seed_accept=K so iter-1's draft 0 reads the prefill seed; thereafter
    # each verify overwrites _verify_h_buf[0..K] and seed_accept tracks the accepted position.
    v_seed_accept = UOp.variable("seed_accept", 0, k)
    Tensor.realize(self._verify_h_buf[:, k:k+1, :].assign(self._mtp_h_buf))
    seed_accept = k
    # buffer tensors are allocated once on first use; verify input width is fixed at K+1.
    # Realize up-front so their UOps are bare BUFFER (no pending assigns) — lets the JIT trace
    # capture the underlying buffer once and lets us mutate contents via raw copyin per iter
    # (skipping the ~30 ms/iter cost of constructing `Tensor([[cur_tok]], dtype=...)`).
    verify_buf = Tensor.zeros(1, verify_T, dtype="int32").contiguous().realize()
    tok_buf = Tensor.zeros(1, 1, dtype="int32").contiguous().realize()
    import array
    tok_stage = array.array('i', [0])
    verify_stage = array.array('i', [0] * verify_T)

    prof = getenv("MTP_PROF", 0)
    dbg_n = getenv("MTP_DEBUG", 0)
    dbg_i = 0
    prof_accept_hist:collections.Counter = collections.Counter()
    prof_sums = {"draft": 0.0, "verify": 0.0, "rollback_seed": 0.0, "iters": 0, "committed": 0}
    prof_rs_by_accept:dict[int, list[float]] = {a: [] for a in range(k+1)}
    def _sync():
      if prof: Tensor([0], device=t.device).realize()

    # accept value to use for the NEXT verify call's mtp_accept binding. Seed = K so the first iter
    # reads its starting SSM state from rs_stack[K] (which we just seeded from recurrent_state above).
    prev_accept = k
    while len(tokens) < self.max_context:
      # ----- draft k tokens with the MTP heads -----
      _sync()
      t_draft = time.perf_counter() if prof else 0.0
      drafts:list[int] = []
      cur_tok = last_committed
      for i in range(k):
        head = i % len(self.mtp_heads)
        # Raw copyin into tok_buf's device buffer — skips Tensor([[cur_tok]]) construction (~30 ms).
        # tok_buf's UOp is a bare BUFFER (no pending assigns), so the JIT trace captures the slot once
        # and replay reads the current bytes; no UOp invalidation needed.
        tok_stage[0] = cur_tok
        tok_buf.uop.buffer.copyin(memoryview(tok_stage))
        sp_m = v_mtp_sp.bind(mtp_start_pos + i)
        if i == 0:
          # draft 0 reads h_prev from _verify_h_buf[seed_accept] directly (fuses the seed in, no realize)
          sample = self.mtp_first_draft_jit(self._verify_h_buf, v_seed_accept.bind(seed_accept), tok_buf, sp_m, temp)
        else:
          # head_h_view aliases self._mtp_h_buf; each call reads its prior value (draft 0's hidden) and overwrites it.
          sample = self.mtp_draft_jits[head](head_h_view, tok_buf, sp_m, temp)
        cur_tok = int(sample.item())
        drafts.append(cur_tok)
      _sync()
      t_verify = time.perf_counter() if prof else 0.0

      # ----- verify: T=K+1 forward over [last_committed, drafts[0..K-1]] in one main pass -----
      # SSM blocks save per-position state to rs_stack/cs_stack during this forward AND read their
      # starting state from rs_stack[mtp_accept] — so no explicit restore copy is needed between iters
      # on partial accept; rebinding mtp_accept on the NEXT call selects the right slot in-place.
      verify_stage[0] = last_committed
      for i, d in enumerate(drafts): verify_stage[i+1] = d
      verify_buf.uop.buffer.copyin(memoryview(verify_stage))
      sp = v_start_pos.bind(start_pos)
      acc = v_mtp_accept.bind(prev_accept)
      samples = self.mtp_verify_jit(verify_buf, sp, acc, temp).realize()  # (1, K+1, 1)
      # One host sync via .tolist() instead of verify_T separate .item() calls — each .item() on a
      # sliced view forces a fresh host-device sync (~44 ms on AMD).
      pred = samples.reshape(verify_T).tolist()
      _sync()
      t_rollback = time.perf_counter() if prof else 0.0
      # pred[i] is main's prediction at verify position i = token that should follow verify_input[i].
      # Compare pred[0..K-1] against drafts[0..K-1]. pred[K] is the bonus if all accepted.
      # pred[0] is main's "what comes after last_committed" -> compare with drafts[0]
      # pred[i] is main's "what comes after drafts[i-1]"     -> compare with drafts[i]   (for i>=1)
      # ...except pred[k-1] which has no draft to compare to: it's a free bonus token if all earlier accepted.

      # find longest accepted prefix among drafts[0..K-1]
      accept = 0
      while accept < k and pred[accept] == drafts[accept]:
        accept += 1

      if dbg_i < dbg_n:
        dbg_i += 1
        print(f"[mtp-dbg iter={dbg_i}] last_committed={last_committed} drafts={drafts} pred={pred} "
              f"accept={accept} mtp_start_pos={mtp_start_pos} start_pos={start_pos}", flush=True)
      # Committed sequence: drafts[0..accept-1] + pred[accept]. j+1 tokens. (When accept==K, pred[K] is bonus.)
      new_tokens = drafts[:accept] + [pred[accept]]

      # SSM rollback is free: next iter binds mtp_accept=accept and the verify body reads
      # rs_stack[accept] / cs_stack[accept] directly as its starting state. The _mtp_h_buf seed is
      # also free now: next iter's first draft reads _verify_h_buf[seed_accept] directly (no copy).
      prev_accept = accept
      seed_accept = accept

      start_pos += accept + 1
      mtp_start_pos += accept + 1
      last_committed = new_tokens[-1]

      if prof:
        _sync()
        t_end = time.perf_counter()
        prof_sums["draft"] += (t_verify - t_draft) * 1000
        prof_sums["verify"] += (t_rollback - t_verify) * 1000
        rs_ms = (t_end - t_rollback) * 1000
        prof_sums["rollback_seed"] += rs_ms
        prof_sums["iters"] += 1
        prof_sums["committed"] += len(new_tokens)
        prof_accept_hist[accept] += 1
        prof_rs_by_accept[accept].append(rs_ms)
        if prof_sums["iters"] % prof == 0:
          n = prof_sums["iters"]
          hist = " ".join(f"{a}:{prof_accept_hist[a]}" for a in range(k+1))
          mean_accept = sum(a*c for a,c in prof_accept_hist.items()) / n
          rs_by = " ".join(f"{a}:{(sum(v)/len(v)) if v else 0:.0f}ms(n={len(v)})" for a, v in prof_rs_by_accept.items())
          print(f"[mtp-prof] iters={n} K={k} draft={prof_sums['draft']/n:.1f}ms verify={prof_sums['verify']/n:.1f}ms "
                f"rollback+seed={prof_sums['rollback_seed']/n:.1f}ms accept_mean={mean_accept:.2f} "
                f"tok/iter={prof_sums['committed']/n:.2f} hist[{hist}] rs_by_accept[{rs_by}]", flush=True)

      for nt in new_tokens:
        tokens.append(nt)
        self._cached_tokens = tokens[:-1]
        yield nt
        if len(tokens) >= self.max_context: return

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    if start_pos < len(self._cached_tokens) and (resets := [r for b in self.blk for r in b._state_reset_ops()]): Tensor.realize(*resets)
    prompt_len = len(tokens)
    if self.has_recurrent_block:
      # SSM blocks need a concrete T (their conv/recurrence is Python-unrolled). Prefill runs
      # CONCRETE-T chunks (PREFILL_CHUNK) + a T=1 tail, then generates at T=1. Default PREFILL_CHUNK=1
      # = token-at-a-time (byte-identical serial prefill); >1 is opt-in and currently gives no
      # speedup on the 27B model (the T>1 forward doesn't amortize the weight read — see generate_mtp).
      out, start_pos = self._prefill_chunked(t, start_pos, prompt_len, v_start_pos, getenv("PREFILL_CHUNK", 1),
                                             temp, seed_mtp=False)
      while True:
        tokens.append(int(out.item()))
        self._cached_tokens = tokens[:-1]
        yield tokens[-1]
        if len(tokens) >= self.max_context: return
        out = self(out, v_start_pos.bind(start_pos), temp).realize()
        start_pos += 1
    v_toks = UOp.variable("toks", 1, chunk_size)
    out = None
    while len(tokens) < self.max_context:
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(min(chunk_size, len(tokens) - start_pos))
      out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
      start_pos += nt.val
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
