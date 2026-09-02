from __future__ import annotations
import array, enum, functools, itertools, pathlib
from typing import cast
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes
from tinygrad.device import Buffer
from tinygrad.llm.kernels.amd import Linear, gated_delta_prefill, flash_attention, amd_custom_kernels_supported
from tinygrad.llm.gguf import gguf_load
from tinygrad.uop.ops import resolve

class ExpertGating(enum.IntEnum):
  SOFTMAX = 1
  SIGMOID = 2
  SOFTMAX_WEIGHT = 3  # softmax over the top-k selected logits
  SQRT_SOFTPLUS = 4

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
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
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False
  num_mtp_heads: int = 0
  mtp_ssm_layer: bool = False  # is the trailing MTP/nextn block itself a GatedDeltaNetBlock (vs a regular attention block)

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
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
      x_down = self.ffn_down_exps(sel, (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous())  # (B, T, k, D)
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
  # MTP speculative-decode verify path: default is just _attention (KV-cache attention blocks are already T>1-capable
  # and self-heal on the next round since future reads never look past the newly-committed prefix length)
  def _attention_verify(self, x:Tensor, start_pos:int|UOp) -> Tensor: return self._attention(x, start_pos)
  # lazily allocate any verify-only state buffers; must run eagerly, before the @function-traced region (like _init_state)
  def _init_verify_state(self, x:Tensor) -> None: pass
  # after MTP verify+accept, roll any block-local speculative state back to the committed prefix. no-op by default
  def commit_verify(self, accept:int|UOp) -> list[UOp]: return []

  def __call__(self, x: Tensor, start_pos: int|UOp, verify:bool=False):
    self._init_state(x)
    if verify: self._init_verify_state(x)
    attn_fn = self._attention_verify if verify else self._attention
    # we pass in the weights implicitly so we unpack the GGUF on the fly
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
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=False)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)

  def _attention(self, x:Tensor, start_pos:int|UOp, _fast:bool=True) -> Tensor:
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
    store = self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).cast(dtypes.half).uop)
    assigned_kv = Tensor(self.cache_kv.uop.after(store))
    # on RDNA3, hybrid models use custom flash attention kernels on the KV cache. the fast kernel needs the query tile
    # 32-aligned (BLOCK_M); the MTP verify window (T=K+1) is padded up to 32 in flash_attention() and its unaligned
    # start_pos is passed as an explicit q_start, so verify can use the fast path too. _fast stays as an escape hatch.
    if _fast and amd_custom_kernels_supported(x.device) and self.config.ssm is not None:
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
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _attention_verify(self, x:Tensor, start_pos:int|UOp) -> Tensor: return self._attention(x, start_pos, _fast=True)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # zeroed so the flash kernels can safely read whole tiles past the valid region (masked lanes multiply by 0)
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim,
                                   dtype=dtypes.half, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

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
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

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

  # verify-window side info (see _init_verify_state); declared here so mypy can track the type across methods
  _verify_N: int|UOp
  _verify_q: Tensor
  _verify_K: Tensor
  _verify_v: Tensor
  _verify_beta: Tensor
  _verify_alpha: Tensor
  _verify_conv_window: Tensor

  def _project(self, x:Tensor, start_pos:int|UOp):
    """conv1d + q/k/v/beta/alpha/out_gate projection shared by the T=1/prefill recurrence (_attention) and the
    TreeWY closed-form verify-window solve (_attention_verify). Returns operands laid out (B, H, T_pad, *) float32,
    plus the raw conv_window (needed by _attention_verify to reconstruct conv_state at an arbitrary accept position)
    and the buffered conv_state store (must be threaded into the recurrent_state read so the conv write always lands
    before the state is consumed, exactly like the original code's `state.uop.after(conv_state_store)` splice)."""
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
    win = Tensor.zeros(B, self.ssm_conv_kernel-1 + T_pad, self.conv_channels).uop
    win = win.after(win[:, :self.ssm_conv_kernel-1].store(conv_state.cast(win.dtype).uop))
    win = win.after(win[:, self.ssm_conv_kernel-1:self.ssm_conv_kernel-1+T].store(self.attn_qkv(x).cast(win.dtype).uop))
    conv_window = Tensor(win)
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
    alpha = log_alpha.transpose(1, 2).exp()  # per-channel decay for kda, per-head otherwise (B, H, T, V|1)
    return q, k, v, beta, alpha, out_gate, conv_state_store, conv_window, initial, is_kda, symbolic, T_pad, start_pos

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q, k, v, beta, alpha, out_gate, conv_state_store, _, initial, is_kda, symbolic, T_pad, start_pos = self._project(x, start_pos)

    # recurrent: scan over the (padded) tokens, updating the recurrent state. collect the per-step outputs
    state = Tensor(self.recurrent_state.uop.after(conv_state_store))  # carry the conv write into this graph
    if self.head_k_dim % 32 == 0 and self.head_v_dim % 4 == 0 and amd_custom_kernels_supported(x.device):
      # one fused kernel for the whole scan; it resets and updates the recurrent state in place (RDNA3)
      core = gated_delta_prefill(q, k, v, beta, alpha, state, Tensor(start_pos)).transpose(1, 2)
    else:
      q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
      alpha = alpha.unsqueeze(-1)
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

  def _attention_verify(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    """Verify-window forward for MTP speculative decoding. Runs the SAME fused gated_delta_prefill kernel the baseline
    scan uses (one custom kernel for the whole N=K+1 window, instead of ~50 generic matmuls from the earlier TreeWY
    closed-form), but on a COPY of the recurrent state so the committed state is untouched until commit_verify picks
    an accept length. The projected per-step operands (q,k,v,beta,alpha) and conv_window are stashed so commit_verify
    can re-run the fused kernel on the real state with the rejected tail masked to no-ops."""
    B, N, _ = x.shape
    assert not isinstance(N, UOp), "verify window length must be a static int (K is fixed for a generate_mtp() run)"
    q, k, v, beta, alpha, out_gate, conv_state_store, conv_window, initial, is_kda, symbolic, T_pad, start_pos = self._project(x, start_pos)
    assert not symbolic and T_pad == N, "verify never uses symbolic/padded chunking"
    assert not is_kda, "verify implements the scalar-alpha (non-KDA) gated delta rule, not KDA's per-channel decay"
    assert self.head_k_dim % 32 == 0 and self.head_v_dim % 4 == 0 and amd_custom_kernels_supported(x.device), \
      "verify fast path requires the RDNA3 gated_delta_prefill custom kernel"

    # forward on a COPY of the committed state: the fused kernel writes the final state in place, and we must not
    # advance the real state before the accept length is known. initial.where(0, ...) applies the position-0 reset.
    # CRITICAL: do NOT thread conv_state_store here -- firing it would mutate the real conv_state during verify (a
    # causality violation: a later verify call at the same position would then read a corrupted conv_state, making
    # pred[0] depend on the draft token). verify must be side-effect-free; only commit_verify writes real state.
    # conv_window is already built from the pre-write conv_state, so verify has everything it needs.
    state_copy = initial.where(0, self.recurrent_state.float()).contiguous()
    core = gated_delta_prefill(q, k, v, beta, alpha, state_copy).transpose(1, 2)  # (B, N, H, Dv)

    # stash the projected operands + conv_window so commit_verify(a) can re-run the fused kernel on the real state.
    # thread stores through .after() (never reassign self._verify_* -- they stay bare buffer refs across calls, like
    # self.recurrent_state) and fire them all with the block output's one bundled realize.
    q_store = self._verify_q.uop.store(q.contiguous().uop)
    k_store = self._verify_K.uop.after(q_store).store(k.contiguous().uop)
    v_store = self._verify_v.uop.after(k_store).store(v.contiguous().uop)
    beta_store = self._verify_beta.uop.after(v_store).store(beta.contiguous().uop)
    alpha_store = self._verify_alpha.uop.after(beta_store).store(alpha.contiguous().uop)
    win_store = self._verify_conv_window.uop.after(alpha_store).store(conv_window.contiguous().uop)

    z = (self.ssm_norm(core) * (out_gate.sigmoid() if is_kda else out_gate.silu())).cast(dtypes.half).contiguous()
    out = self.ssm_out(z.reshape(B, N, -1)).contiguous()
    # splice the verify-state writes into the block output so realizing the caller's forward also fires these stores
    return Tensor(out.uop.after(win_store))

  def commit_verify(self, accept:int|UOp) -> list[UOp]:
    """Commit the recurrent_state/conv_state to accept position `a` (0-indexed within the just-verified N-token window,
    i.e. "accept a+1 tokens") by re-running the fused gated_delta_prefill kernel on the REAL state with the rejected
    tail (i>a) masked to no-ops: beta=0 disables the delta update and alpha=1 disables decay, so a full fixed-N scan
    lands exactly on the state after the accepted prefix (validated numerically). Fixed shape -> JIT-safe; the mask is
    parameterized by the bound `a` variable. Returns the state-write uops so generate_mtp bundles all 48 SSM blocks'
    writes into ONE realize/host sync (a per-block realize is a full USB4 round-trip each -- ~20ms x 48 = ~1s/iter)."""
    if not hasattr(self, "_verify_q"): return []  # this block's verify path was never exercised this run
    N = self._verify_N
    assert isinstance(N, int), "verify window length must be a static int"
    assert isinstance(accept, int), "commit_verify accept must be a plain int (it drives a static conv-window slice)"
    keep = (Tensor.arange(N) <= accept).float()             # (N,) 1 for i<=accept, 0 for the rejected tail
    beta_m = self._verify_beta * keep.reshape(1, 1, N)      # rejected steps: beta=0 -> delta rule off
    alpha_m = self._verify_alpha * keep.reshape(1, 1, N, 1) + (1 - keep).reshape(1, 1, N, 1)  # rejected: alpha=1 (no decay)
    # run the fused kernel on the real recurrent_state (holds S_0; verify ran on a copy so it wasn't advanced). the
    # None branch mutates the state buffer in place and its math is exact for full accept (unit-tested 0.00000). the
    # masking makes rejected steps no-ops so a fixed-N scan lands on S_a.
    state = Tensor(self.recurrent_state.uop)
    core = gated_delta_prefill(self._verify_q, self._verify_K, self._verify_v, beta_m, alpha_m, state)
    # conv_state for the committed prefix is the trailing (kernel-1) columns of the window ending at a+1. use a PLAIN
    # INT slice (a symbolic bound-Variable offset silently reads the wrong columns) and a PLAIN store (matching the
    # working buffer.uop.store(value) pattern in _project -- putting .after() on the store TARGET does not land).
    # return BOTH writes: core.uop fires the in-place SSM-state mutation, conv_store fires the conv write; they touch
    # independent buffers so generate_mtp bundles them (and all other blocks') into one realize.
    new_conv_state = self._verify_conv_window[:, accept+1:accept+1+self.ssm_conv_kernel-1]
    conv_store = self.conv_state.uop.store(new_conv_state.cast(self.conv_state.dtype).contiguous().uop)
    # return READ-AFTER-STORE uops, not bare STOREs: Tensor(bare_store).realize() does NOT execute the store, but
    # realizing a buffer read chained .after(store) does fire it (this is why core.uop -- a read of the kernel's
    # in-place mutation -- lands but a bare conv_store did not).
    return [core.uop, self.conv_state.uop.after(conv_store)]

  def _init_verify_state(self, x:Tensor) -> None:
    B, N, device = x.shape[0], x.shape[1], x.device
    assert isinstance(N, int), "verify window length must be a static int (K is fixed for a generate_mtp() run)"
    if hasattr(self, "_verify_q") and self._verify_N == N and self._verify_q.shape[0] == B: return
    self._verify_N = N
    Dv, Dk, H = self.head_v_dim, self.head_k_dim, self.num_v_heads
    # CRITICAL: zero-init, never Tensor.empty -- commit_verify reads these and a stray NaN would silently poison every
    # subsequent restored state (see tmp/qwen_mtp_ssm-more_than_you_wanted_to_know.md)
    self._verify_q = Tensor.zeros(B, H, N, Dk, device=device).contiguous().realize()
    self._verify_K = Tensor.zeros(B, H, N, Dk, device=device).contiguous().realize()
    self._verify_v = Tensor.zeros(B, H, N, Dv, device=device).contiguous().realize()
    self._verify_beta = Tensor.zeros(B, H, N, device=device).contiguous().realize()
    self._verify_alpha = Tensor.zeros(B, H, N, 1, device=device).contiguous().realize()
    self._verify_conv_window = Tensor.zeros(B, self.ssm_conv_kernel-1+N, self.conv_channels, device=device).contiguous().realize()

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, device=x.device).clone()

class MTPHead:
  """DeepSeek-V3-style multi-token-prediction head (GGUF's "nextn" block): combines the main model's pre-lm_head
  hidden state at position t with the embedding of the (drafted) token at t+1, feeds that through one transformer
  block, and predicts the token at t+2. Reuses the main model's token_embd/output/output_norm; the head brings its
  own enorm/hnorm/eh_proj plus a dedicated final-norm (shared_head_norm is NOT tied to output_norm -- verified
  numerically distinct, mean abs diff ~0.3, on the qwen3.8:27b-uncensored checkpoint this was built against)."""
  def __init__(self, config:TransformerConfig):
    self.enorm = nn.RMSNorm(config.dim, config.norm_eps)
    self.hnorm = nn.RMSNorm(config.dim, config.norm_eps)
    self.eh_proj = Linear(2*config.dim, config.dim, bias=False)
    self.shared_head_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.block: FFNBlock = GatedDeltaNetBlock(config, config.ssm) if config.mtp_ssm_layer and config.ssm else \
      (MLATransformerBlock(config) if config.kv_lora_rank > 0 else TransformerBlock(config))

  def __call__(self, h_prev:Tensor, tok_embed:Tensor, start_pos:int|UOp) -> Tensor:
    # concat order [enorm(embed); hnorm(h_prev)] -- Qwen3.5/3.6-family MTP uses the opposite order from DeepSeek-V3
    # (verified empirically against real generation for this model family; see tmp/qwen_mtp_ssm-more_than_you_wanted_to_know.md)
    x = self.eh_proj(self.enorm(tok_embed).cat(self.hnorm(h_prev), dim=-1))
    return self.block(x, start_pos)

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(dense_config if i < config.leading_dense_blocks else config, config.ssm)
                               if config.ssm and config.ssm_layers[i] else
                               block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.mtp_heads: list[MTPHead] = [MTPHead(config) for _ in range(config.num_mtp_heads)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)
    # MTP speculative-decode hot path: each K-loop draft step and the verify forward get their own TinyJit, mirroring
    # rollout_jit above -- start_pos is bound to a UOp Variable at the call site (generate_mtp), never passed as a
    # fresh Python int, so one capture serves every position (exactly like rollout_jit already does for start_pos).
    # NOTE: this is a *per-step* JIT (one call = one draft token), not a single JIT wrapping the whole K-token loop --
    # wrapping the whole loop in one capture hit a "bind mismatch" error from rebinding the same Variable expr to
    # different concrete values within a single trace; per-step JIT (replayed K times from a plain Python loop, like
    # rollout_jit is replayed once per generated token) sidesteps that entirely.
    self.mtp_draft_jit = TinyJit(self._mtp_draft_step)
    self.verify_jit = TinyJit(self.forward_verify)

  def _run_blocks(self, tokens:Tensor, start_pos:int|UOp, verify:bool=False) -> Tensor:
    x = self.token_embd(tokens).float()                   # (B, T, D)
    for block in self.blk: x = block(x, start_pos, verify=verify)
    return x

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    x = self._run_blocks(tokens, start_pos)
    # only run the output projection on the last token
    logits = self.output(self.output_norm(x[:, -1:]))[:, -1, :]
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  def forward_verify(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> tuple[Tensor, Tensor]:
    """MTP verify forward: T=N=K+1 tokens [last_committed, draft_0, ..., draft_{K-1}], all through the TreeWY closed
    form on GatedDeltaNetBlocks (see GatedDeltaNetBlock._attention_verify). Returns (pred, hidden) for every position:
    pred are the greedy/sampled tokens the *main* model would produce at each position, hidden is the pre-output_norm
    hidden state (needed to seed the next round's MTP draft chain from the accepted position)."""
    x = self._run_blocks(tokens, start_pos, verify=True)  # (B, N, D)
    logits = self.output(self.output_norm(x))             # (B, N, vocab)
    pred = (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1)
    return pred, x

  def _mtp_draft_step(self, tok:Tensor, h_prev:Tensor, start_pos:int|UOp, temperature:Tensor) -> tuple[Tensor, Tensor]:
    """one MTP head step: embed `tok`, combine with `h_prev`, run the head block, sample the next draft token."""
    tok_embed = self.token_embd(tok).float()
    h = self.mtp_heads[0](h_prev, tok_embed, start_pos)
    logits = self.output(self.mtp_heads[0].shared_head_norm(h))[:, -1, :]
    samp = (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)
    return samp, h

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
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe'):
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
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)
    num_mtp_heads = kv.get(f'{arch}.nextn_predict_layers', 0)
    main_num_blocks = kv[f'{arch}.block_count'] - num_mtp_heads
    mtp_ssm_layer = False

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

    # MTP ("nextn") head remap: blk.{main_num_blocks+k}.{nextn.enorm,nextn.hnorm,nextn.eh_proj,nextn.shared_head_norm,
    # attn_*, ffn_*, *_norm} -> mtp_heads.{k}.{enorm,hnorm,eh_proj,shared_head_norm,block.*}. shared_head_norm is kept
    # (NOT tied to output_norm -- verified numerically distinct on qwen3.8:27b-uncensored, mean abs diff ~0.31).
    # embed_tokens.weight (if present) is dropped: tied to the main token_embd.
    for k in range(num_mtp_heads):
      blk_idx = main_num_blocks + k
      state_dict.pop(f'blk.{blk_idx}.embed_tokens.weight', None)
      mtp_ssm_layer = f'blk.{blk_idx}.attn_qkv.weight' in state_dict
      for name in [n for n in state_dict if n.startswith(f'blk.{blk_idx}.')]:
        v = state_dict.pop(name)
        suffix = name[len(f'blk.{blk_idx}.'):]
        if suffix.startswith('nextn.'): state_dict[f'mtp_heads.{k}.{suffix[len("nextn."):]}'] = v
        else: state_dict[f'mtp_heads.{k}.block.{suffix}'] = v

    config = TransformerConfig(
      num_blocks=main_num_blocks, dim=kv[f'{arch}.embedding_length'],
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
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear')),
      expert_gating_func=ExpertGating(kv.get(f'{arch}.expert_gating_func', ExpertGating.SOFTMAX)),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict,
      num_mtp_heads=num_mtp_heads, mtp_ssm_layer=mtp_ssm_layer)
    model = Transformer(config)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def warmup(self):
    for _ in range(2): list(zip(range(2), self.generate([0])))

  def get_start_pos(self, tokens:list[int]) -> int:
    # recurrent state can't be partially reused after divergence: reuse it only when tokens extend the cached prefix
    if self.has_recurrent_block:
      return len(self._cached_tokens) if self._cached_tokens and len(self._cached_tokens) < len(tokens) \
        and tokens[:len(self._cached_tokens)] == self._cached_tokens else 0
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
    if self.has_recurrent_block and not amd_custom_kernels_supported(self.token_embd.weight.device): chunk_size = 1
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      n_toks = min(chunk_size, len(tokens) - start_pos)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
      start_pos += n_toks
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]

  def generate_mtp(self, tokens:list[int], K:int, chunk_size:int=32, temperature:float=0.0):
    """Speculative decoding via the MTP head(s): chain-draft K tokens, verify all K+1 (last_committed + K drafts) in
    one batched TreeWY forward, commit the longest accepted prefix (+1 bonus token), and reconstruct the SSM state at
    the accept position in O(1) via GatedDeltaNetBlock.commit_verify -- no per-position state snapshots."""
    assert K >= 1 and self.mtp_heads, "generate_mtp requires --mtp K>=1 and a checkpoint with MTP heads"
    if self.has_recurrent_block and not amd_custom_kernels_supported(self.token_embd.weight.device): chunk_size = 1
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    temp = Tensor([temperature])
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    start_pos = self.get_start_pos(tokens)
    prompt_len = len(tokens)
    ssm_blocks = [b for b in self.blk if isinstance(b, GatedDeltaNetBlock)]
    accept_hist = [0] * (K + 1)

    # prefill exactly like generate(), but keep the pre-output_norm hidden state of the last processed position --
    # that seeds the first MTP draft head call
    # prefill all but the last prompt token: verify window position 0 IS last_committed and re-absorbs it, so the
    # committed state entering verify must exclude it (else it's double-counted in the SSM recurrence AND written to
    # the KV cache one position too late). the last prompt token is carried as last_committed. hidden ends up as the
    # trunk state at position prompt_len-2 -- exactly the h_prev seed the first draft head call needs.
    hidden = None
    while start_pos < prompt_len - 1:
      n_toks = min(chunk_size, prompt_len - 1 - start_pos)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      hidden = self._run_blocks(t[:, sp:sp+nt], sp).realize()
      start_pos += n_toks
    # .contiguous().realize() normalizes the view to a bare buffer -- h_prev otherwise carries a slice ShapeTracker
    # whose Python-int offset would make mtp_draft_jit's captured input signature mismatch across iterations.
    # when there's nothing to prefill (1-token prompt, or a continuation where state is already cached), there's no
    # valid h_prev position -- seed with zeros. this only degrades iteration-1 draft quality (-> more rejects that
    # round), never correctness: verify always recomputes the true hidden and reseeds h_prev from vhidden after.
    h_prev = hidden[:, -1:].contiguous().realize() if hidden is not None else \
      Tensor.zeros(1, 1, self.token_embd.weight.shape[1], device=self.token_embd.weight.device).contiguous().realize()
    last_committed = tokens[-1]

    # pre-allocated device buffers for the per-iteration token inputs -- copyin() writes raw bytes directly into an
    # already-realized buffer, skipping Tensor(python_list) construction (~30ms/call on AMD, see
    # tmp/qwen_mtp_ssm-more_than_you_wanted_to_know.md Update 5) entirely. dtype/shape stay fixed for the whole run
    # (K is static), so these buffers -- and the JIT captures that read them -- are reused across every iteration.
    device = self.token_embd.weight.device
    tok_buf = Tensor.zeros(1, 1, dtype="int32", device=device).contiguous().realize()
    verify_buf = Tensor.zeros(1, K + 1, dtype="int32", device=device).contiguous().realize()
    tok_stage = array.array('i', [0])
    verify_stage = array.array('i', [0] * (K + 1))

    def _copyin(buf:Tensor, stage:array.array) -> None:
      b = buf.uop.buffer
      assert isinstance(b, Buffer), "mtp draft/verify token buffers are never multi-device"
      b.allocator._copyin(b._buf, memoryview(stage))

    while len(tokens) < self.max_context - K - 1:
      drafts: list[int] = []
      cur_tok, cur_h = last_committed, h_prev
      for i in range(K):
        tok_stage[0] = cur_tok
        _copyin(tok_buf, tok_stage)
        samp, cur_h = self.mtp_draft_jit(tok_buf, cur_h, v_start_pos.bind(start_pos + i), temp)
        # cur_h comes back as a GETTUPLE(CALL(...)) result (the @function-decorated block's return convention) --
        # it still carries the just-used start_pos bind symbolically. Feeding it straight back as next iteration's
        # h_prev would pull that stale bind into the next call's schedule alongside the new one and raise "bind
        # mismatch" in create_linear_with_vars. .contiguous().realize() collapses it to a bare buffer first.
        cur_h = cur_h.contiguous().realize()
        cur_tok = int(samp.item())
        drafts.append(cur_tok)

      verify_stage[0] = last_committed
      for i, d in enumerate(drafts): verify_stage[i + 1] = d
      _copyin(verify_buf, verify_stage)
      pred, vhidden = self.verify_jit(verify_buf, v_start_pos.bind(start_pos), temp)
      pred_list = cast(list[int], pred.reshape(K + 1).tolist())
      accept = 0
      while accept < K and pred_list[accept] == drafts[accept]: accept += 1
      committed = drafts[:accept] + [pred_list[accept]]
      accept_hist[accept] += 1

      # bundle every SSM block's state-write into ONE realize -- a per-block realize is a full USB4 host round-trip
      commit_stores = [s for b in ssm_blocks for s in b.commit_verify(accept)]
      if commit_stores: Tensor.realize(*[Tensor(s) for s in commit_stores])

      tokens.extend(committed)
      start_pos += accept + 1
      # see the .contiguous().realize() note above: normalize the accept-offset slice before it becomes next
      # iteration's mtp_draft_jit h_prev input
      h_prev, last_committed = vhidden[:, accept:accept+1].contiguous().realize(), committed[-1]
      self._cached_tokens = tokens[:-1]
      for tk in committed: yield tk

    total = sum(accept_hist)
    if total: print(f"[mtp] accept_hist={accept_hist} mean_accept={sum(i*c for i,c in enumerate(accept_hist))/total:.3f} iters={total}")
