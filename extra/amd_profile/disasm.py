# Compile a kernel from a DEBUG=4 log and print its resource usage and instruction mix (needs llvm-objdump/llvm-readelf with AMDGPU).
#   DEBUG=4 DEV=AMD python bench.py > log.txt
#   DEV=AMD python extra/amd_profile/disasm.py log.txt <kernel name> [--asm out.s] [--order]
# --order prints the sequence of WMMAs (W), 16-byte LDS loads (R), barriers (B), global loads (G), LDS stores (S) and waits of the loop body
import argparse, collections, re, shutil, subprocess, tempfile
from tinygrad import Device

def llvm_tool(name:str) -> str:
  for cand in (name, f"/opt/homebrew/opt/llvm/bin/{name}", f"/opt/homebrew/opt/llvm@22/bin/{name}", f"/opt/homebrew/opt/llvm@21/bin/{name}"):
    if shutil.which(cand): return cand
  raise RuntimeError(f"{name} not found (brew install llvm)")

def kernel_source(log:str, name:str) -> str:
  i = log.find(name + "(")
  if i < 0: raise RuntimeError(f"kernel {name} not in the log")
  start = log.rfind("typedef long unsigned int size_t;", 0, i)  # the HIP renderer's preamble starts every program
  # the program ends at its first line starting with "}": a progress bar ("compiling ...") can be printed right after the brace
  return log[start:log.find("\n}", i)] + "\n}\n"

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("log")
  ap.add_argument("kernel")
  ap.add_argument("--asm")
  ap.add_argument("--order", action="store_true")
  args = ap.parse_args()
  dev = Device["AMD"]
  lib = dev.compiler.compile(kernel_source(open(args.log).read(), args.kernel))
  with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as f: f.write(lib)
  notes = subprocess.run([llvm_tool("llvm-readelf"), "--notes", f.name], capture_output=True, text=True).stdout
  def get(k:str) -> str: return m.group(1) if (m:=re.search(rf"\.{k}:\s+(\d+)", notes)) else "?"
  print(f"vgpr {get('vgpr_count')}  sgpr {get('sgpr_count')}  vgpr spill {get('vgpr_spill_count')}  lds {get('group_segment_fixed_size')}B  "
        f"scratch {get('private_segment_fixed_size')}B  max workgroup {get('max_flat_workgroup_size')}")
  arch = f"gfx{dev.target[0]}{dev.target[1]:x}{dev.target[2]:x}"
  asm = subprocess.run([llvm_tool("llvm-objdump"), "-d", f"--mcpu={arch}", f.name], capture_output=True, text=True).stdout
  if args.asm: open(args.asm, "w").write(asm)
  ins = [l.split("//")[0].strip() for l in asm.split("\n") if l.startswith("\t")]
  ops = [x.split()[0] for x in ins if x]
  print(len(ops), "instructions")
  memops = ("v_wmma", "ds_", "global_", "buffer_", "scratch_", "s_waitcnt", "s_barrier")
  for op, n in collections.Counter(o for o in ops if o.startswith(memops)).most_common(): print(f"  {op:28s} {n}")
  if args.order and (w:=[n for n, o in enumerate(ops) if o.startswith("v_wmma")]):
    tag = {"v_wmma": "W", "ds_load_b128": "R", "s_barrier": "B", "global_load": "G", "ds_store": "S"}
    def short(n:int, o:str) -> str|None:
      if (t:=next((t for p, t in tag.items() if o.startswith(p)), None)) is not None: return t
      return f"w({ins[n].split(None, 1)[1].replace(' ', '')})" if o == "s_waitcnt" else None
    seq = [short(n, o) for n, o in enumerate(ops[max(0, w[0]-80):w[-1]+5], max(0, w[0]-80))]
    print(" ".join(s for s in seq if s))
