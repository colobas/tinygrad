# Run a script with AMD performance counters (PMC) and print one kernel's counter totals.
#   python extra/amd_profile/pmc.py <kernel name substring> <COUNTER,COUNTER,...> <script.py> [ENV=VAL ...]
# example:
#   python extra/amd_profile/pmc.py linear_iq4_xs TA_TA_BUSY,GRBM_GUI_ACTIVE,SQ_WAVE_CYCLES,SQ_WAIT_ANY bench.py IQ4_LDS=0
# an unknown counter name makes the runtime print the list of available counters.
import os, re, subprocess, sys

# settings that work through TinyGPU (macOS): the default SQTT buffer (256MB per shader engine, allocated even with SQTT=0) does not map,
# and a full one breaks the readback. the last PROF_SLOTS kernels are kept: run the kernel of interest last.
PMC_ENV = {"DEV": "AMD", "PMC": "1", "SQTT": "0", "SQTT_BUFFER_SIZE": "64", "PROF_SLOTS": "8", "VIZ": "-1"}

def strip(s:str) -> str: return re.sub(r"\x1b\[[0-9;]*m", "", s)

def counters(kernel:str, names:str, script:str, extra_env:dict[str, str]) -> dict[str, int]:
  env = {**os.environ, **PMC_ENV, "PMC_COUNTERS": names, **extra_env}
  run = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True)
  if run.returncode != 0: raise RuntimeError(f"{script} failed:\n{run.stderr[-2000:]}")
  listing = subprocess.run([sys.executable, "-m", "tinygrad.viz.cli"], capture_output=True, text=True).stdout.split("\n")
  # the listing keeps the kernel name colors, the lookup takes the plain name
  srcs = [strip(l)[len("--- SOURCE "):] for l in listing if l.startswith("--- SOURCE ") and l.endswith(" PMC") and kernel in strip(l)]
  if not srcs: raise RuntimeError(f"no PMC sample of a kernel matching {kernel!r}")
  out = subprocess.run([sys.executable, "-m", "tinygrad.viz.cli", "-s", srcs[-1]], capture_output=True, text=True).stdout
  # the first line of every counter is the total over all its instances, the next lines are per shader engine / array / WGP or CU
  return {m.group(1): int(m.group(2)) for l in out.split("\n") if (m:=re.match(r"^([A-Z0-9_]+) (-?[0-9]+)$", strip(l)))} | {"_kernel": srcs[-1]}

if __name__ == "__main__":
  if len(sys.argv) < 4: sys.exit("usage: pmc.py <kernel name substring> <COUNTER,...> <script.py> [ENV=VAL ...]")
  res = counters(sys.argv[1], sys.argv[2], sys.argv[3], dict(a.split("=", 1) for a in sys.argv[4:]))
  print("#", res.pop("_kernel"))
  for k, v in res.items(): print(f"{k:32s} {v}")
