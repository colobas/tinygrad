# Metal ICB miscompute repro (no tinygrad): a register-heavy kernel gives different results executed from an indirect command buffer than
# dispatched directly. Seen on an M2 Pro (Apple8), macOS 27.0.1. run: uv run --with pyobjc-framework-Metal python extra/metal_icb_repro.py
import struct, Metal, random
dev = Metal.MTLCreateSystemDefaultDevice()
def make_src(nacc, L, body_heavy):
  fields = "device float* out; " + " ".join(f"device float* in{i};" for i in range(nacc))
  s = ["#include <metal_stdlib>", "using namespace metal;", f"struct args_t {{ {fields} }};",
       "kernel void k(constant args_t& args [[buffer(0)]], uint3 gid [[threadgroup_position_in_grid]]) {"]
  s += [f"  float acc{i}[1];" for i in range(nacc)]
  s += ["  int x = gid.x;"]
  for i in range(nacc):
    s += [f"  acc{i}[0] = 0.0f;", f"  for (int l = 0; l < {L}; l++) {{", f"    float v = args.in{i}[(x*{L} + l) % 65536];"]
    for u in range(body_heavy):
      s += [f"    float w{u} = args.in{i}[(x*{L} + l + {u*37}) % 65536]*{1.0+u*0.01}f + {u*0.001}f;",
            f"    v = v + w{u}*(1.0f/((1.0f+exp2(w{u}*-1.4426950216293335f))*0.7f));"]
    s += [f"    acc{i}[0] = acc{i}[0] + v;", "  }"]
  s += ["  int z = gid.z;", "  float r = acc0[0];"] + [f"  r = (z == {i}) ? acc{i}[0] : r;" for i in range(1, nacc)]
  s += ["  args.out[x + z*gid.x*0 + z*4096] = r;", "}"]
  return "\n".join(s)
def run(src, nacc, icb, gx=4096):
  lib, err = dev.newLibraryWithSource_options_error_(src, None, None); assert lib, err
  d = Metal.MTLComputePipelineDescriptor.new(); d.setComputeFunction_(lib.newFunctionWithName_("k")); d.setSupportIndirectCommandBuffers_(icb)
  pso, err = dev.newComputePipelineStateWithDescriptor_options_reflection_error_(d, 0, None, None)
  random.seed(3)
  ins = [dev.newBufferWithBytes_length_options_(struct.pack('65536f', *[random.uniform(-1, 1) for _ in range(65536)]), 65536*4, 0) for _ in range(nacc)]
  out = dev.newBufferWithLength_options_(gx*nacc*4, 0)
  argb = dev.newBufferWithBytes_length_options_(struct.pack(f'{nacc+1}Q', out.gpuAddress(), *[b.gpuAddress() for b in ins]), 8*(nacc+1), 0)
  q = dev.newCommandQueue(); cb = q.commandBuffer(); enc = cb.computeCommandEncoder()
  for b in ins + [out, argb]: enc.useResource_usage_(b, 3)
  if icb:
    desc = Metal.MTLIndirectCommandBufferDescriptor.new(); desc.setCommandTypes_(Metal.MTLIndirectCommandTypeConcurrentDispatch)
    desc.setMaxKernelBufferBindCount_(1); desc.setInheritBuffers_(False); desc.setInheritPipelineState_(False)
    ib = dev.newIndirectCommandBufferWithDescriptor_maxCommandCount_options_(desc, 1, 0)
    c = ib.indirectComputeCommandAtIndex_(0); c.setComputePipelineState_(pso); c.setKernelBuffer_offset_atIndex_(argb, 0, 0)
    c.concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSizeMake(gx, 1, nacc), Metal.MTLSizeMake(1, 1, 1))
    enc.useResource_usage_(ib, 1); enc.executeCommandsInBuffer_withRange_(ib, Metal.NSMakeRange(0, 1))
  else:
    enc.setComputePipelineState_(pso); enc.setBuffer_offset_atIndex_(argb, 0, 0)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSizeMake(gx, 1, nacc), Metal.MTLSizeMake(1, 1, 1))
  enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  n = gx*nacc; return struct.unpack(f'{n}f', bytes(out.contents().as_buffer(n*4)))
for nacc in (20,):
  for heavy in (4, 16, 32):
    src = make_src(nacc, 128, heavy)
    a, b = run(src, nacc, False), run(src, nacc, True)
    print(f"nacc={nacc} heavy={heavy} direct-vs-icb maxdiff={max(abs(x-y) for x,y in zip(a,b)):.3g}", flush=True)
