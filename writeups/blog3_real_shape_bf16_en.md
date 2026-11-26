# blog3 · Real shape and bf16 — which conclusions survive on real weights, and the two bills paid separately

> English version of `blog3_real_shape_bf16.md`. It is a **translation**, not a second text: every number, every
> `md5`-style evidence pointer, every archive filename and every table cell comes from the Chinese original verbatim;
> where the two disagree, **the Chinese text wins**. Term map for what is kept in Chinese: archive series labels
> `甲/乙/丙/丁` = i–iv (literal ids inside the archive filenames) · printed column/marker names `配对可用`
> (pairing-usable), `KL比(other/主)` (KL ratio other/main) and the archived row name `①异质` (row ① heterogeneous).

> **Writing red lines (shared by the four write-ups, pinned in the header):**
> ① **First-order and second-order quantities in two columns, always together**: "error <1%" is not allowed to stand as a
> sentence of its own (where L2 differs by 11.6x, KL differs by 303x; in the gc=0→1 cell the two columns point in
> opposite directions).
> ② **Quality is reported as a relative quantity only, and always next to the reference row**: swapping the real layer-8
> MLP for an "fp32 dense 8-of-2 MoE" is itself worth KL 1.897e-01 (bs=32) / 8.854e-02 (bs=128) ⇒ **668x / 871x** more
> expensive than the harshest engine cell.
> ③ **Memory must say whether the number is peak MB or steady-state MB**: at gc=5 both variants peak at 186.1MB; it is
> the steady state that separates 169.3 (fp32) vs 109.8 (bf16).
> ④ **The three-way exchange uses only the two P0-④a archives**; every number points back to a verdict table run by hand
> by the user.
> ⑤ **Inside a paired section only the rows whose `配对可用` marker is ✓ may be read as Δ**; the Δ of a ✗ row is the noise floor.
> ⑥ **Cross-process bimodal cells are quoted as an interval, never as a point** (A bs=128 `gc=3 bf16` = 1.35 / 1.74ms).

## 0. In one sentence

Every one of the first 14 cuts was measured on a **toy shape** (D=1024 / N=2048, `randn` weights with std=1, and the A/C
cells manufactured by a seed). P0-① moved them onto **SmolLM-135M's real weights + real activations** (D=576 / N=1536,
routing = embed, x = the input of the layer-8 MLP) and re-ran; the conclusions split into three kinds: **the ones that
survived**, **the ones whose price tag changed**, and **the bugs the toy shape had hidden**. This piece covers the three
kinds, then the two new accounts that fell out on the real shape: **the price of dropping precision** (−16% → −53%,
rising with gc) and **the two bills of quantisation** (int8 is **274~364x** dearer than bf16).

| Conclusion on the toy | On real weights | Evidence (all run by hand by the user) |
|---|---|---|
| the gains of the first 14 cuts | ✅ **not one cut was overturned**: the ⓪T4 baseline trails by **+110%~+648%** | P0-① four paths |
| CPU byte wall 42GB/s | ✅ same formula, only the bytes per expert changed: **2.53MB ⇒ 63us/expert** | P0-① |
| whether the working set crowds L3 | ✅ it actually got **roomier**: 6.00→**2.53MB** per expert (0.42x), 3 CPU-resident = **7.6MB < L3 12MB** (the toy could only fit 1) | same |
| hit skew | 🆕 **not seeded, measured**: `[42.3, 13.2, 17.7, 6.8, 1.4, 12.6, 6.0, 0.1]`, hottest/coldest **433x** | same |
| the four precision accounts | 🚨 **dearer, and dearer in a new place**: re-quantising h **+1.94pp** (toy +0.44pp) | same |
| fp16 overflows | 🚨 **toy-only**: on real weights max\|h\|=**13.6** does not overflow (toy: **223 inf/nan**) | same |
| bit-for-bit diffs all green | 🚨 **the toy shape hid a mine**: D=1024 divides ⇒ no tail block ⇒ an out-of-bounds read is invisible | red line H |
| "heterogeneous always wins" | 🚨 **at bs=1 both configurations are beaten by pure CPU** (−20.8% / −13.9%) — **the scope has been narrowed**: this holds only under a **dense GPU leg**; once the GPU leg reads weights by real traffic, bs=1 is instead **−34.1%** faster than pure CPU (乙-2d) | P0-① four paths + 乙-2d |

---

## 1. What the real distribution changed: routing, hits, and that h account

**Changing the activation = changing the object under test** (P0-① bs=1 C, the same cell with three activations
swapped in, and the routing swaps along with them):

| Activation swapped in | whole-line relRMSE | CPU-leg slots |
|---|---|---|
| real activation (input of the layer-8 MLP) | **7.2267%** | 2 |
| same-scale gaussian | 4.9813% | 1 |
| std=1 toy | 0.0001% | **0** |

The std=1 toy beat the routing into "everything goes into the GPU", **0 slots** left on the CPU leg ⇒ the "quantisation
error" simply goes to zero. **A toy is not a simplification, it deletes the object under test.** That is the entire
provenance of the `①异质` row's relRMSE=0.0001% under toy activation.

**The four precision accounts** (measuring only those 2 slots on the CPU leg; the denominator is the fp32 ground truth of
that instance's CPU leg, rms=0.796):

| account | real weights (SmolLM-135M) | toy comparison column |
|---|---|---|
| ① quantise W only | relRMSE **4.4110%** | 2.03% |
| ② plus quantising x | 5.2886% (**+0.88pp**) | +0.29pp |
| ③ plus re-quantising h | 7.2267% (**+1.94pp**) | +0.44pp |
| ④ kernel as measured | 7.2267% (④vs③ absolute **5.16e-06**, relative 4.6e-07 ⇒ ≤1 LSB level) | — |

⇒ Not one of these accounts can be saved by "dropping W's precision further": **the real distribution amplifies h
(the re-quantised swiglu intermediate) by roughly 3.6x**, and it is now the **single big head** of the CPU-leg error. If
you want to save more bytes, first give h a finer grid (one scale per 128) — keep your hands off W.
While we are at it, the fp16 question gets closed here: on real weights max\|g\|=5.92 / max\|u\|=6.2 /
max\|h\|=**13.6**, a long way below 65504 ⇒ **bf16 remains the only candidate because engineering does not want to manage
a second overflow boundary, not because fp16 would blow up.**

---

## 2. The price of dropping precision: rising to −53% with gc, but **never 2x**

The sentence "bytes ÷2 ⇒ time ÷2" is killed first on the **toy shape** (`bench_dtype` is exactly D=1024/N=2048): 乙-2c,
bs=1, gc=5, bit-comparable under the same seed, paired **5/5 rounds faster**.

| cell (bare end-to-end) | GPU bytes/frame | GPU half | effective bandwidth | end-to-end |
|---|---|---|---|---|
| gemm fp32 → bf16 | 40 → **20MB** | 0.179 → 0.144 (**−19.6%**) | 235 → **146GB/s** | **−15.4%** |
| swiglu fp32 → bf16 | 120 → **60MB** | 0.473 → 0.290 (**−38.8%**) | 266 → 217GB/s | **−33.9%** |

Bytes exactly halved, time cut by only a third to a fifth ⇒ **the half-precision kernel does not get full bandwidth on a
skinny GEMV**.
(The mechanistic explanation rests on an attribution probe: the big `gu` cell −49% with full cash-in, the `down` cell
starting at only 4MB gets no bandwidth at all, and the **≈0.05ms** of pre+D2H inside the graph does not move with dtype —
that probe was run by the model itself, **not part of the record**; the prices in the text are all in the next table.)

**The real price of dropping precision lives in P0-④a's paired section** (real shape, real activation, both variants
resident at the same gc plus odd/even order swapping; only rows whose `配对可用` marker is ✓ are listed):

| bs | config | gc | p50 fp32 → bf16 | paired Δ | faster rounds | steady-state MB saved |
|---|---|---|---|---|---|---|
| 1 | A | 3 / 5 / 7 / 8 | 0.257→0.215 / 0.321→0.262 / 0.395→0.300 / 0.431→0.312 | −16.3 / −18.3 / −24.1 / **−27.5%** | 7/7 ×4 | 35.4 / 59.6 / 84.6 / 95.5 |
| 32 | A | 5 / 7 / 8 | 0.511→0.396 / 0.693→0.351 / 0.743→0.354 | −22.5 / −49.3 / **−52.4%** | 7/7 ×3 | 59.6 / 84.6 / 95.5 |
| 128 | A (main) | 5 / 7 / 8 | 0.761→0.700 / 1.021→0.482 / 1.097→0.567 | −8.1 / **−52.8** / **−48.3%** | 7/7 ×3 | 59.6 / 84.6 / 95.5 |
| 128 | A (appendix) | 7 / 8 | 1.036→0.491 / 1.101→0.571 | **−52.6 / −48.2%** | 7/7 | 84.6 / 95.5 |
| 128 | C (main/appendix) | 8 | 1.098→0.571 / 1.154→0.594 | **−48.0 / −48.5%** | 7/7 | 95.5 |

One monotone line, read plainly: **the cash-in rate of the precision cut = whether the GPU half became the critical
path.**

- bs=1 (the CPU leg and the fixed serial cost are still on the critical path): −16%~−27%;
- bs≥32 with gc≥7 (the CPU leg is almost emptied out, the GPU half *is* the critical path): **−48%~−53%**, the two
  rounds' credentials differing by ≤0.5pp;
- the large-bs cells with gc≤5 (the CPU leg still has 22~234 slots) **do not even amount to an effect size** — of the
  30 paired rows only 15 are ✓, and the most expensive misreading among the remaining ✗ rows is A bs=128 `gc=3` at
  **+29.9%** (per-round median −1.2%, faster rounds 5/7), which turned into −19.3% in the appendix round:
  **two rounds giving Δ of opposite sign for the same cell is the direct demonstration of the noise floor.**

⇒ The sentence you are allowed to write: "on this machine, when the placement pushes the GPU half into being the
critical path, bf16 buys **−48%~−53%** of frame time and **85~96MB** of steady-state memory; at bs=1 it is only worth
−16%~−27%."
🚨 Forbidden: "dropping precision = 2x / bf16 is certainly twice as fast" — the bytes really do halve, the time never
does.

---

## 3. The two bills: int8 on the CPU leg and bf16 on the GPU leg, **each owning a whole row**

The split was not fitted; it was obtained through **zero traffic having no numerical entry point**: at `gc=0` the GPU leg
has **0 slots** (gate ⑤: changing dtype leaves a bitwise 0), and with `gc≥3` placed correctly the CPU leg has **0 slots**.
So one and the same table contains two rows that are "the pure int8 cost" and "the pure bf16 cost":

| bs | CPU-leg int8 (gc=0) | GPU-leg bf16 (0 slots on the CPU leg) | L2 ratio | **KL ratio** |
|---|---|---|---|---|
| 1 | L2 7.2267% / KL **4.928e-03** / TV 3.912e-02 | 0.7862% / **1.353e-05** / 2.060e-03 | 9.2x | **364x** |
| 32 | 5.9174% / **2.841e-04** / 5.967e-03 | 0.5121% / **9.376e-07** / 3.914e-04 | 11.6x | **303x** |
| 128 | 5.5114% / **1.016e-04** / 3.384e-03 | 0.4929% / **3.705e-07** / 2.250e-04 | 11.2x | **274x** |

(the bf16 column takes the cells with 0 CPU-leg slots: for bs=1 the four cells `gc=3/5/7/8` have **bit-identical** KL
1.353e-05; for bs=32/128 take `gc=8`.)

Only when three things hold at once do you have the actual conclusion of this section:

1. **the int8 bill is 274~364x dearer than the bf16 one** (a second-order quantity), while the first-order quantity
   differs by only **9~12x** ⇒ the first-order quantity cannot see the magnitude of this money at all (blog1 §3).
2. **bf16 is almost free at token level**: of the 12 cells of the whole bs=128 table, switching to bf16 moves only the
   `gc=5` cell's top-1 from 100.00 → 99.22 — **1 of 128 tokens flipped**, and that cell's resolution was 0.78pp anyway.
3. **ΔNLL / ppl is noise across the entire table**: every engine cell has |ΔNLL| ≤ **2.743e-03**, the ppl ratio stays
   inside **[0.9973, 1.0023]** with both signs present; while the reference row (the MoE swap itself) is
   ΔNLL **+1.378e-01** (bs=32) / **+5.290e-02** (bs=128), i.e. **50x** larger than the harshest engine cell.
   ⇒ **P0-③ verdict ⑥**: ppl may not appear in the quality section, and not one sentence of the "int8 made ppl better"
   kind may be written.

Two table-reading disciplines: ① when the paired section prints a `KL比(other/主)` of something like **2.2e+05**, those
are cells where the fp32 row was already ≈1e-11 — **a ratio whose denominator tends to zero is not read**; read the
absolute KL instead (bs=128 bf16 = 3.705e-07). ② Every KL here is a relative quantity — the reference row is
**668x (bs=32) / 871x (bs=128)** dearer than the harshest engine cell, so do not use this table's KL as if it were ppl
(red line ②).

---

## 4. The two memory columns and the Pareto front: bf16 is the front's main shareholder

The two column bases may not be mixed (red line ③): **peak MB** = window high-water minus the idle baseline (it contains
the **fp32 staging** of graph-build and warm-up, which is why dropping precision **does not touch it**: bs=1 `gc=5`
peaks at the same 186.1MB for both variants); **steady-state MB** = what is still resident after warm-up (the precision
cut is visible in this column only: 169.3 → **109.8**). The weight formula gives fp32 **10.1** / bf16 **5.1** / int8
**2.53** MB per expert; both columns are larger than the formula = `wcat` + dead bmm copies + graph static buffers, not a
bug. That row of gate ⑤, `峰值列之后额外高水位 +92~94MB`, is **the fp32 referee's own** CPU-leg weight copy
(`(E−gc)×10.1MB`), **not engine cost**.

The Pareto front (latency ↓ / steady-state memory ↓ / KL ↓, dominated in none of the three; tie band 2%):

| cell | front points | dominated (money paid for nothing) | reading |
|---|---|---|---|
| bs=1 A | 4 (`0fp32 1bf16 3fp32 3bf16`) | 8 points | the front retreats to `gc≤3`: **at bs=1 the GPU half is not worth money** (blog2 §4) |
| bs=32 A | 7 | `0fp32 1bf16 3fp32 5fp32 7fp32` | **5 of the 7 front points are bf16**; **4 of the 5 dominated points are fp32** |
| bs=128 A | 10 | **only `5fp32` `7fp32`** (two rounds word-for-word identical) | fp32 occupies more memory at equal KL ⇒ dominated |
| bs=128 C | main 8 / appendix 7 | main `1fp32 3fp32 5fp32 7fp32`; the appendix has one extra **`5bf16`** | **the only cross-process unstable front cell = `gc=5 bf16`**: bimodality 2.257 / 3.282 pushes it straight into the dominated set |

⇒ the claim "dropping precision is an optional optimisation" does not hold here: **fp32's two cells fall out of the front
outright at large bs when placed correctly.**
🚨 When quoting memory you must name which column, and the front set itself is, on bimodal cells, a **process-level**
property rather than a configuration property.

---

## 5. The three-cell claim of this project (the three green lines of fig2; quote them in this form)

The baseline is **`gc=0` at the same dtype** (the pure-CPU lower bound), and memory is reported **steady-state**:

| cell (placed correctly, bf16) | p50 | steady-state MB | KL(P‖Q) | in one sentence |
|---|---|---|---|---|
| bs=1 `gc=0 → 3` | 0.273 → **0.215** | 0.1 → **65.9** | 4.928e-03 → **1.353e-05** | pay 66MB ⇒ **21.2%** faster and **364x** better in quality |
| bs=32 `gc=0 → 5` | 1.378 → **0.390** | 0.3 → **110.0** | 2.841e-04 → **1.180e-05** | pay 110MB ⇒ **71.7%** faster and **24x** better |
| bs=128 `gc=0 → 7` | 3.536→**0.510** / 3.427→**0.493** | 1.0 → **155.3** | 1.016e-04 → **1.532e-06** | pay 155MB ⇒ **−85.6% faster (the same value in both rounds)** and **66x** better |

The three green lines share **one and the same mechanism**: `gc↑` moves slots from the int8 leg onto the bf16 leg and KL
drops by two orders of magnitude; the time is a by-product (blog2's two walls decide when it is a real saving).
Quoting discipline: ① the bs=128 endpoints must be handed over with **both credentials together** (main table
3.536→0.510, appendix round 3.427→0.493); ② do not quote the absolute p50 of A bs=128 `gc=3 bf16` (bimodal
**1.35 / 1.74ms**); ③ the misplaced cell's debt is reported only on the **fp32 side +101~106%** — the two single-point
figures 297%/472% on the bf16 side may not enter the text.

---

## 6. The third bill priced along the way: a zero-traffic expert still pays every frame (full table in blog2 §6)

At bs=1 placed correctly, the `gc=3 → gc=8` cells carry **exactly the same traffic** (GPU leg 2 slots in both, CPU leg 0
slots in both), p50 fp32 **0.257 → 0.434**, bf16 **0.215 → 0.317** ⇒ **every extra resident zero-traffic expert costs
35us (fp32) / 20us (bf16)**, and five extra residents cost **69% / 47% of the frame time in exchange for 0 hits**. In the
same cell the precision Δ is **−16.1% (7/7 rounds, values untouched bit-for-bit)** ⇒ the **reclaimable-magnitude probe**
for this money is already in hand (乙-2d target ÷2~÷2.5), and it is also the root cause of **the rule's books and the
measurement disagreeing**.

---

## 7. What is left over (in the paper this is exactly these rows, no gloss)

| What is missing | Current state | Which claim it touches |
|---|---|---|
| **held-out ppl** | the corpus is a patchwork of this repository's readme (**out of domain**, 7244 tokens) | absolute ppl is meaningless ⇒ the whole quality section reports relative quantities only (red line ②); it gets swapped the moment `CORPUS=/path/to/wt2.txt` arrives |
| **sample resolution** | the top-1 denominator = that cell's bs: T=128 gives only **0.78pp**, T=1 is a single-token coin toss | the "bf16 is free" line in §3 only reaches a 1-token granularity; the `BS=512` follow-up run was not done |
| **cross-process credentials** | only bs=128 has a second-process round; **3 bimodal cells** (A`gc3`+28.6% / C`gc5`+45.4% / C`gc7`+8.4%, all bf16) | the bs=1/32 endpoints of §5 are still single-process; methodology 26 |
| **the GPU half at bs=1** | beaten by pure CPU even when placed correctly (−20.8% / −13.9%) — **dense code path only**; once weights are read by traffic the best bs=1 is **0.184ms**, **34.1%** faster than the **0.279ms** median of the 8 cells of the same pure-CPU code | the sentence "heterogeneous gains start only from **bs=4**" **has been rewritten by 乙-2d**: the decode-regime gain comes from **pricing the GPU half by traffic**, not from raising bs |
| **the cost of misplacing** | the C cell at bs≥32 is overtaken by pure GPU by **−30.2% / −62.9%** | a wrong placement rule makes the whole optimisation worthless ⇒ P1 "dynamic hot-expert placement" |
| **fp16** | real weights do not overflow (max\|h\|=13.6), the toy scale blows up with 223 inf/nan | only bf16 remains a candidate; fp16 stays a control and never enters the ledger |
| **delivered ≠ future work** [correction of this row, 2026-11-24] | **delivered, no longer future work**: 乙-2d traffic-sparse batching, placement rule v2 (already the default cell), `n_threads` adaptivity **ran its verdict round and landed on ⚪ do-not-write-a-rule** [7 of the 8 real-shape cells show a tax of ≤0.2% for the fixed `nt=6`, `5088fec0`/`556d5e9b`]; **still frozen**: T15 / int4 / VNNI / merging the `down` graph / large-bs tuning | only the frozen items are "future work"; this-box constants (42GB/s, 250GB/s, 12MB L3) do not extrapolate |

> Data sources (all run by hand by the user, in `results/`): `2026-11-21_P0-①_real_swiglu_bs1-128.txt`,
> `2026-11-21_乙-2c_dtype_gc5-bs1.txt`, `2026-11-21_P0-③_metrics_top1-kl.txt`,
> `2026-11-21_P0-④a_swap_3way.txt`, `2026-11-21_P0-④a附_bs128_crossproc.txt`, `2026-11-21_red_after_P0-①.txt`;
> the toy comparison column and the four-path verdict table are in the P0-① entry of OPTLOG §2 (verdict tables of the
> same archive).
