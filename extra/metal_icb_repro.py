# Metal ICB miscompute repro (no tinygrad): a register-heavy kernel gives different results executed from an indirect command buffer than
# dispatched directly. Seen on an M2 Pro (Apple8), macOS 27.0.1. run: uv run --with pyobjc-framework-Metal python extra/metal_icb_repro.py
import struct, random
import Metal

dev = Metal.MTLCreateSystemDefaultDevice()
N_IN = 65536

def make_src(nacc:int, loop:int, terms:int) -> str:
  fields = "device float* out; " + " ".join(f"device float* in{i};" for i in range(nacc))
  s = ["#include <metal_stdlib>", "using namespace metal;", f"struct args_t {{ {fields} }};",
       "kernel void k(constant args_t& args [[buffer(0)]], uint3 gid [[threadgroup_position_in_grid]]) {"]
  s += [f"  float acc{i}[1];" for i in range(nacc)] + ["  int x = gid.x;"]
  for i in range(nacc):
    s += [f"  acc{i}[0] = 0.0f;", f"  for (int l = 0; l < {loop}; l++) {{", f"    float v = args.in{i}[(x*{loop} + l) % {N_IN}];"]
    for u in range(terms):
      s += [f"    float w{u} = args.in{i}[(x*{loop} + l + {u*37}) % {N_IN}]*{1.0+u*0.01}f + {u*0.001}f;",
            f"    v = v + w{u}*(1.0f/((1.0f+exp2(w{u}*-1.4426950216293335f))*0.7f));"]
    s += [f"    acc{i}[0] = acc{i}[0] + v;", "  }"]
  s += ["  int z = gid.z;", "  float r = acc0[0];"] + [f"  r = (z == {i}) ? acc{i}[0] : r;" for i in range(1, nacc)]
  s += ["  args.out[x + z*4096] = r;", "}"]
  return "\n".join(s)

def run(src:str, nacc:int, icb:bool, gx:int=4096) -> tuple[float, ...]:
  lib, err = dev.newLibraryWithSource_options_error_(src, None, None)
  assert lib is not None, err
  desc = Metal.MTLComputePipelineDescriptor.new()
  desc.setComputeFunction_(lib.newFunctionWithName_("k"))
  desc.setSupportIndirectCommandBuffers_(icb)
  pso, err = dev.newComputePipelineStateWithDescriptor_options_reflection_error_(desc, 0, None, None)
  random.seed(3)
  ins = [dev.newBufferWithBytes_length_options_(struct.pack(f'{N_IN}f', *[random.uniform(-1, 1) for _ in range(N_IN)]), N_IN*4, 0)
         for _ in range(nacc)]
  out = dev.newBufferWithLength_options_(gx*nacc*4, 0)
  argb = dev.newBufferWithBytes_length_options_(struct.pack(f'{nacc+1}Q', out.gpuAddress(), *[b.gpuAddress() for b in ins]), 8*(nacc+1), 0)
  cb = dev.newCommandQueue().commandBuffer()
  enc = cb.computeCommandEncoder()
  for b in ins + [out, argb]: enc.useResource_usage_(b, 3)
  grid, tg = Metal.MTLSizeMake(gx, 1, nacc), Metal.MTLSizeMake(1, 1, 1)
  if icb:
    idesc = Metal.MTLIndirectCommandBufferDescriptor.new()
    idesc.setCommandTypes_(Metal.MTLIndirectCommandTypeConcurrentDispatch)
    idesc.setMaxKernelBufferBindCount_(1)
    ib = dev.newIndirectCommandBufferWithDescriptor_maxCommandCount_options_(idesc, 1, 0)
    cmd = ib.indirectComputeCommandAtIndex_(0)
    cmd.setComputePipelineState_(pso)
    cmd.setKernelBuffer_offset_atIndex_(argb, 0, 0)
    cmd.concurrentDispatchThreadgroups_threadsPerThreadgroup_(grid, tg)
    enc.useResource_usage_(ib, 1)
    enc.executeCommandsInBuffer_withRange_(ib, Metal.NSMakeRange(0, 1))
  else:
    enc.setComputePipelineState_(pso)
    enc.setBuffer_offset_atIndex_(argb, 0, 0)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(grid, tg)
  enc.endEncoding()
  cb.commit()
  cb.waitUntilCompleted()
  return struct.unpack(f'{gx*nacc}f', bytes(out.contents().as_buffer(gx*nacc*4)))

if __name__ == "__main__":
  print(dev.name())
  for terms in (4, 16, 32):
    src = make_src(20, 128, terms)
    direct, icb = run(src, 20, False), run(src, 20, True)
    print(f"20 accumulators, {terms} terms per step: direct vs icb maxdiff={max(abs(a-b) for a, b in zip(direct, icb)):.3g}")
