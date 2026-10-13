# AMD kernel profiling

Use these tools to find where a hand-written AMD kernel spends its time. They were built while tuning the IQ4_XS prefill kernels in
`tinygrad/llm/kernels/amd.py` on a 7900 XTX (gfx1100) through TinyGPU.

## Tools

`pmc.py` runs a script with performance counters and prints the totals of one kernel:

```bash
python extra/amd_profile/pmc.py linear_iq4_xs TA_TA_BUSY,GRBM_GUI_ACTIVE,SQ_WAVE_CYCLES,SQ_WAIT_ANY bench.py SOME_ENV=1
```

`disasm.py` compiles a kernel from a `DEBUG=4` log. It prints the register and LDS usage, the instruction mix and, with `--order`,
the instruction order of the main loop:

```bash
DEBUG=4 DEV=AMD python bench.py > log.txt
DEV=AMD python extra/amd_profile/disasm.py log.txt linear_iq4_xs_lds_wmma --order
```

`--order` writes W for a WMMA, R for a 16-byte LDS load, G for a global load, S for an LDS store and B for a barrier.
`w(...)` is an `s_waitcnt`, the point where the wave stops for its loads.
`disasm.py` needs `llvm-objdump` and `llvm-readelf` with AMDGPU support (`brew install llvm`).

The settings that work through TinyGPU are in `tinygrad/viz/README.md`.

## Useful counters (gfx11)

| counter | meaning |
|---|---|
| `GRBM_GUI_ACTIVE` | GPU busy cycles: the time base |
| `SQ_WAVE_CYCLES`, `SQ_BUSY_CYCLES` | wave-cycles, SQ busy cycles: their ratio is the average number of waves resident |
| `SQ_WAIT_ANY` | wave-cycles spent waiting (memory counters, barriers): compare with `SQ_WAVE_CYCLES` |
| `SQ_WAIT_INST_ANY`, `SQ_WAIT_INST_LDS` | waits on instruction dependencies, on LDS |
| `SQ_INSTS_VALU`, `SQ_INSTS_LDS`, `SQ_INST_CYCLES_VMEM` | instruction counts (VALU includes the WMMAs) |
| `SQC_LDS_BANK_CONFLICT` | LDS bank conflict cycles |
| `TA_TA_BUSY` | busy cycles of the vector memory address unit, summed over the 96 CUs: divide by the CU count and by `GRBM_GUI_ACTIVE` |
| `GL2C_HIT`, `GL2C_MISS` | L2 hits and misses (128B lines): misses x 128B is the memory traffic |

## Method

1. Measure the kernel time alone with a synthetic input of the real shape.
2. Make variants that each remove one part: replace the activation or weight fragments by constants, skip the LUT, skip the
   dequant math, skip the global loads, skip the WMMAs. The time of each variant shows the cost of the removed part.
3. If a variant is clearly faster, read the counters of the full kernel and of the variant to find the cause.
4. Read the instruction order of the loop with `disasm.py --order` before changing the code structure. The compiler often
   already does the obvious scheduling.

Removing a part can change register use and occupancy. Check with `disasm.py` that a variant did not change them.

## Example: IQ4_XS prefill (Clef, 5120->17408, 256 tokens)

| kernel | time | finding |
|---|---|---|
| per-wave f16 WMMA, 128-token tiles | 1015us | the tile choice: 64 tokens x 2 output tiles is 28% faster |
| per-wave, 64x2 tiles | 734us | TA 82% busy, waves wait 83%. The weights come from 16 scattered rows, in both half-waves, in every wave |
| LDS tiles of activations and weights, register prefetch | 632us | LDS-bandwidth bound: 48 x 512B fragment loads per step per wave. Without them it runs at the pure WMMA time (375us) |
| weight tile in LDS, activations from global memory | 598us | waves wait on the activation loads (see below) |

Tried and rejected:
- An int8 WMMA kernel: slower, and int8 activations are 35x less accurate. The scale of each 32-weight subblock has to be
  applied to the int32 WMMA result in VALU.
- A software prefetch of the weight words in the per-wave kernel.
- An XOR-swizzled LDS layout: the 16-byte loads were split into scalar loads.
- Double-buffered LDS weight tiles.
- Higher occupancy in the per-wave kernel: the L2 misses tripled.

Things to know about this hardware:
- On RDNA3 the WMMAs run on the VALU. Dequantization instructions take issue slots from the matrix math.
- Both half-waves of a wave32 WMMA hold the same A and B fragments, so all loads and dequantization of fragments are done twice.
- `vmcnt` counts the global loads in issue order. A wave that waits for one load also waits for all the loads issued before it.

## Leads

- The activation loads of the LDS kernel are issued in pairs just before their WMMA, each pair followed by `s_waitcnt vmcnt(0)`
  (`disasm.py --order`). The tinygrad source already has all 64 activation loads at the top of the loop body: LLVM's AMDGPU
  scheduler moves each one down to its use. These did not change that schedule:
  - an order dependency of the weight prefetch on the activation loads,
  - `amdgpu_waves_per_eu(1, N)` with N from 1 to 4,
  - `__builtin_amdgcn_sched_group_barrier` groups (all VMEM reads then the WMMAs, or interleaved) in the loop body.

  Compiler options do change it (`AMD_LLVM_OPTS` adds codegen options, `AMD_WAVES_PER_EU=min,max` adds the kernel attribute):

  | settings | VGPRs | loads issued ahead | ffn_up 256 tokens |
  |---|---|---|---|
  | default | 119 | 0 (each pair waited for at its WMMA) | 577us |
  | `-mllvm -amdgpu-sched-strategy=max-ilp` or `-amdgpu-schedule-relaxed-occupancy=1` | 252 | all | 601us |
  | `-amdgpu-schedule-relaxed-occupancy=1` + `AMD_WAVES_PER_EU=8,16` | 189 | 18 | 559us |

  The last row is 1-10% faster on every Clef IQ4_XS shape and makes a Clef request 3% faster. Both settings apply to every
  kernel, and they cost Bonsai PQ2_0 10% of its prefill: they stay opt-in until they can be set per kernel.
- Prefetch the activation fragments of the next K step in registers: tried and no gain. All 16 fragments (128 VGPRs) spilled.
  One or two K slices fit (the kernel uses 119 VGPRs) but ran 2-5% slower.
