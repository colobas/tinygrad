from __future__ import annotations
import functools
from typing import cast
from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

# ******** Q1_0 (1-bit Bonsai) decode on Apple GPUs ********
# The weights are re-packed block-major: word (b, o, i) holds the 32 sign bits of weights b*128+i*32..+31 of output row o, so the 16
# bytes a thread loads per block are contiguous and a simdgroup's loads are fully coalesced. The f16 scales follow as (b, o) halves.
# The activations are quantized to int8 per 128-block and split into 8 bit-planes (word p of a 32-group has bit j = bit p of q_j),
# so sum_j bit_j*q_j = sum_p 2^p * popcount(word & plane_p) (plane 7 weighted -128). The weights are 2*bit-1, so the kernel uses
# 2*dot - sum_j q_j. Apple GPUs have no dp4a: this is ~1 op per weight vs ~5 for the generic fused dequant.

Q1_BLOCK = 128

def _popc(v:UOp) -> UOp: return UOp(Ops.CUSTOMI, src=(v,), arg=("popcount({})", dtypes.uint32))

def q1_repack(block_bytes:Tensor, out_features:int, in_features:int) -> Tensor:
  # (out*in/128, 18) ggml blocks -> one uint32 buffer: [nb, out, 4] bit words, then the [nb, out] f16 scales two per word
  nb = in_features // Q1_BLOCK
  blocks = block_bytes.reshape(out_features, nb, 18)
  words = blocks[:, :, 2:].contiguous().bitcast(dtypes.uint32).permute(1, 0, 2)   # (nb, out, 4)
  scales = blocks[:, :, :2].contiguous().bitcast(dtypes.half).reshape(out_features, nb).permute(1, 0).contiguous()
  return words.flatten().cat(scales.flatten().bitcast(dtypes.uint32)).contiguous()

def q1_dequant(packed:Tensor, out_features:int, in_features:int) -> Tensor:
  # (out, in) f16 view of the re-packed weight, for prefill matmuls (fused into the matmul by the scheduler)
  nb = in_features // Q1_BLOCK
  words = packed[:nb*out_features*4].reshape(nb, out_features, 4).permute(1, 0, 2)          # (out, nb, 4)
  scales = packed[nb*out_features*4:].bitcast(dtypes.half).reshape(nb, out_features).permute(1, 0)
  bits = (words.unsqueeze(-1) >> Tensor.arange(32, dtype=dtypes.uint32)) & 1                   # (out, nb, 4, 32)
  return ((bits.cast(dtypes.half)*2 - 1).reshape(out_features, nb, Q1_BLOCK) * scales.unsqueeze(-1)).reshape(out_features, in_features)

@functools.cache
def _q1_decode_kernel(out:UOp, w:UOp, planes:UOp, xsc:UOp, qsum:UOp, out_features:int, in_features:int, lanes:int, ksplit:int) -> UOp:
  tokens, nb = cast(int, out.shape[0]), in_features // Q1_BLOCK
  words, scales = w.flatten()[:nb*out_features*4].reshape(nb, out_features, 4), w.flatten()[nb*out_features*4:]
  ob, ks, lane = UOp.range(out_features//lanes, 0, AxisType.GLOBAL), UOp.range(ksplit, 1, AxisType.GLOBAL), UOp.range(lanes, 2, AxisType.LOCAL)
  o = ob*lanes + lane
  acc = UOp.placeholder((tokens,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0)))
  br = UOp.range(nb//ksplit, 3, AxisType.REDUCE)
  b = ks*(nb//ksplit) + br
  bw = tuple(words[b, o, i].load() for i in range(4))
  sidx = b*out_features + o
  d = ((scales[sidx//2].load() >> ((sidx % 2)*16).cast(dtypes.uint32)) & 0xffff).cast(dtypes.uint16).bitcast(dtypes.half).float()
  upd = []
  for t in range(tokens):
    dot = UOp.const(0, dtypes.int32)
    for i in range(4):
      for p in range(8):
        c = _popc(bw[i] & planes[t, b*4 + i, p].load()).cast(dtypes.int32)
        dot = dot + (c * -128 if p == 7 else c << p)
    upd.append(acc[t].store(acc.after(br)[t].load() + d * xsc[t, b].load() * (dot*2 - qsum[t, b].load()).cast(dtypes.float32)))
  loop = UOp.group(*upd).end(br)
  st = [out[t, o, ks].store(acc.after(loop)[t].load()) for t in range(tokens)]
  return UOp.group(*st).end(ob, ks, lane).sink(arg=KernelInfo(name="linear_q1_0_metal", opts_to_apply=()))

# ******** Q1_0 prefill: port of mlx's qmm_t (mlx/backend/metal/kernels/quantized.h + steel/gemm/mma.h) ********
# 32x32x32 tiles, 4 simdgroups in a 2x2 grid, each owning 16x16 of the output as 2x2 8x8 simdgroup_matrix accumulators. every K step
# the workgroup stages the activation tile and the dequantized weight tile in threadgroup memory. with the block-major re-pack, a
# 32-wide K tile of an output row is exactly one bit word.

Q1_BM = Q1_BN = Q1_BK = 32
Q1_PAD = 8  # halves of padding per threadgroup row (mlx: BK_padded = BK + 16 / sizeof(T))

def _frag_coord(lane:UOp) -> tuple[UOp, UOp]:
  # mlx steel BaseMMAFrag::get_coord: lane holds (row fm, cols fn and fn+1) of every 8x8 fragment
  qid = lane // 4
  return (qid & 4) + (lane // 2) % 4, (qid & 2) * 2 + (lane % 2) * 2

@functools.cache
def _q1_prefill_kernel(out:UOp, w:UOp, x:UOp, out_features:int, in_features:int) -> UOp:
  tokens, nb = cast(int, out.shape[0]), in_features // Q1_BLOCK
  words, scales = w.flatten()[:nb*out_features*4].reshape(nb, out_features, 4), w.flatten()[nb*out_features*4:]
  tile_n, tile_m = UOp.range(out_features//Q1_BN, 0, AxisType.GLOBAL), UOp.range(tokens//Q1_BM, 1, AxisType.GLOBAL)
  sg, lane = UOp.range(4, 2, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  tid, fm, fn = sg*32 + lane, *_frag_coord(lane)
  tm, tn = (sg // 2) * 8, (sg % 2) * 8
  accs = [UOp.placeholder((2,), dtypes.float32, slot=i, addrspace=AddrSpace.REG) for i in range(4)]
  accs = [a.after(a.store(a.const_like(0))) for a in accs]
  xs_lds = UOp.placeholder((Q1_BM, Q1_BK + Q1_PAD), dtypes.half, slot=10, addrspace=AddrSpace.LOCAL)
  ws_lds = UOp.placeholder((Q1_BN, Q1_BK + Q1_PAD), dtypes.half, slot=11, addrspace=AddrSpace.LOCAL)
  kt = UOp.range(in_features//Q1_BK, 3, AxisType.REDUCE)
  # stage: thread t copies 8 activations of row t//4 and dequantizes 8 weights (bits 8*(t%4)..+7 of one word) of output row t//4
  row, part = tid // 4, tid % 4
  stores = [xs_lds.after(kt)[row, part*8 + j].store(x[tile_m*Q1_BM + row, kt*Q1_BK + part*8 + j].load()) for j in range(8)]
  n = tile_n*Q1_BN + row
  word = words[kt // 4, n, kt % 4].load()
  sidx = (kt // 4)*out_features + n
  d = ((scales[sidx//2].load() >> ((sidx % 2)*16).cast(dtypes.uint32)) & 0xffff).cast(dtypes.uint16).bitcast(dtypes.half)
  stores += [ws_lds.after(kt)[row, part*8 + j].store(((word >> (part*8 + j)) & 1).ne(0).where(d, -d)) for j in range(8)]
  xs_t, ws_t = xs_lds.after(UOp.barrier(UOp.group(*stores))), ws_lds.after(UOp.barrier(UOp.group(*stores)))
  vals = [accs[i*2+j].after(kt) for i in range(2) for j in range(2)]
  cur = [UOp.stack(v[0].load(), v[1].load()) for v in vals]
  for kk in range(Q1_BK // 8):
    afr = [UOp.stack(*(xs_t[tm + 16*i + fm, kk*8 + fn + e].load() for e in range(2))) for i in range(2)]
    # B[k][n] = W[n][k]: the lane's B elements are (k = kk*8+fm, n = fn and fn+1)
    bfr = [UOp.stack(*(ws_t[tn + 16*j + fn + e, kk*8 + fm].load() for e in range(2))) for j in range(2)]
    cur = [UOp.wmma(afr[i], bfr[j], cur[i*2+j], (8, 8, 8), "METAL", 32) for i in range(2) for j in range(2)]
  upd = UOp.group(*(accs[i].store(cur[i]) for i in range(4))).barrier().end(kt)
  st = [out[tile_m*Q1_BM + tm + 16*i + fm, tile_n*Q1_BN + tn + 16*j + fn + e].store(accs[i*2+j].after(upd)[e].load())
        for i in range(2) for j in range(2) for e in range(2)]
  return UOp.group(*st).end(tile_n, tile_m, sg, lane).sink(arg=KernelInfo(name="linear_q1_0_qmm", opts_to_apply=()))

def q1_prefill(packed:Tensor, x:Tensor, out_features:int, in_features:int) -> Tensor:
  tokens = cast(int, x.numel()) // in_features
  assert tokens % Q1_BM == 0 and out_features % Q1_BN == 0 and in_features % Q1_BLOCK == 0
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_q1_prefill_kernel, out_features=out_features, in_features=in_features)
  return Tensor.custom_kernel(out, packed, x.reshape(tokens, in_features).cast(dtypes.half).contiguous(), fxn=fxn)[0] \
    .reshape(*x.shape[:-1], out_features)

def q1_activation_planes(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor, Tensor]:
  if (memo := _planes_memo.get(x.uop)) is not None: return memo
  nb = in_features // Q1_BLOCK
  xr = x.reshape(tokens, nb, Q1_BLOCK).float()
  xsc = (xr.abs().max(-1) / 127).maximum(1e-8)
  q = (xr / xsc.unsqueeze(-1)).round().clip(-127, 127).cast(dtypes.int32)
  qu = q.bitcast(dtypes.uint32).reshape(tokens, nb*4, 32, 1)
  planes = (((qu >> Tensor.arange(8, dtype=dtypes.uint32)) & 1) << Tensor.arange(32, dtype=dtypes.uint32).reshape(32, 1)).sum(2, dtype=dtypes.uint32)
  _planes_memo[x.uop] = ret = (planes.contiguous(), xsc.contiguous(), q.sum(-1).contiguous())
  return ret
_planes_memo:dict[UOp, tuple[Tensor, Tensor, Tensor]] = {}

def _tiling(out_features:int, in_features:int) -> tuple[int, int]:
  # 64 threads per group; split K until there are enough groups to fill the GPU (measured on an M2 Pro: 272 groups x 2, 80 x 8)
  lanes = next(l for l in (64, 32, 16, 8, 4, 2, 1) if out_features % l == 0)
  nb, ksplit = in_features // Q1_BLOCK, 1
  while (out_features//lanes)*ksplit < 512 and nb % (ksplit*2) == 0: ksplit *= 2
  return lanes, ksplit

def q1_linear(packed:Tensor, x:Tensor, out_features:int, in_features:int) -> Tensor:
  tokens = cast(int, x.numel()) // in_features
  planes, xsc, qsum = q1_activation_planes(x, tokens, in_features)
  lanes, ksplit = _tiling(out_features, in_features)
  out = Tensor.empty(tokens, out_features, ksplit, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_q1_decode_kernel, out_features=out_features, in_features=in_features, lanes=lanes, ksplit=ksplit)
  return Tensor.custom_kernel(out, packed, planes, xsc, qsum, fxn=fxn)[0].sum(-1).reshape(*x.shape[:-1], out_features)
