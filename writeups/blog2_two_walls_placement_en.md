# blog2 · Two walls and the placement rule — the most valuable output of this system, refuted by measurement four times

> English version of `blog2_two_walls_placement.md`. It is a **translation**, not a second text: every number, every
> `md5`-style evidence pointer, every archive filename and every table cell comes from the Chinese original verbatim;
> where the two disagree, **the Chinese text wins**. Term map for what is kept in Chinese: archive series labels
> `甲/乙/丙/丁` = i–iv (they are literal ids inside the archive filenames) · printed column/marker names `配对可用`
> (pairing-usable) and `规则选gc` (rule-picks-gc) are column headers of the archived tables.

> **Writing red lines (shared by the four write-ups, pinned in the header):**
> ① **First-order and second-order quantities in two columns, always together**: "error <1%" is not allowed to stand as a
> sentence of its own (methodology 22: where L2 differs by 11.6x, KL differs by 303x; in the gc=0→1 cell the two columns
> point in opposite directions).
> ② **Quality is reported as a relative quantity only, and always next to the reference row**: swapping the real layer-8
> MLP for an "fp32 dense 8-of-2 MoE" is itself worth KL 1.897e-01 (bs=32) / 8.854e-02 (bs=128) ⇒ **668x / 871x** more
> expensive than the harshest engine cell.
> ③ **Memory must say whether the number is peak MB or steady-state MB**: at gc=5 both variants peak at 186.1MB; it is the
> steady state that separates 169.3 (fp32) vs 109.8 (bf16).
> ④ **The three-way exchange uses only the two P0-④a archives**; every number points back to a verdict table run by hand
> by the user.
> ⑤ **Inside a paired section only the rows whose `配对可用` marker is ✓ may be read as Δ**; the Δ of a ✗ row is the noise floor.
> ⑥ **Cross-process bimodal cells are quoted as an interval, never as a point** (A bs=128 `gc=3 bf16` = 1.35 / 1.74ms).

## 0. In one sentence

**The placement rule (which experts go into the GPU) is the real output of this whole thing; the kernels and the graphs are
only the price paid for letting the rule speak.**
And its history is exactly four refutations by measurement: the reason is the same every time — **the rule was missing a
leg**. This piece gives each leg its own price list: two walls on the CPU side, one wall on the GPU side, a new leg on the
quality side, and in §8 the fourth leg that drives the first three into a corner — **unit price**.

The model is a single line (every table pays by it):

```
per-frame latency ≈ max(CPU leg, GPU leg) + fixed serial host overhead
```

**Overlap itself has a price, and the price is positive**: P0-② measured it with `overlap=False` (flipping one boolean on
the same model object, outputs bit-identical, mutual difference 0.000000000) and got **positive in 16/16 cells,
+8.5%~+67.6%**, rising monotonically with bs when A is placed correctly (gemm +8.5→+39.5, swiglu +11.6→+67.6, at
bs=1/4/32/128 in that order). This is the **only legitimate source** of that "X times speedup".

---

## 1. Two walls on the CPU side, and one crossing point

| Wall | Its price | Who stands on this wall | How it was proven |
|---|---|---|---|
| **byte wall** | every expert **with traffic** reads its whole pass 2MB ÷ 42GB/s = **49us/expert** (real-shape swiglu 2.53MB ⇒ **63us**) | config A (placed correctly, about 1/5 of a slot per expert) | pure-CPU calliper measured the 45GB/s band; changing the arithmetic does nothing |
| **compute wall** | 2M int8 MACs per slot ÷ **288 GMAC/s** = **7us/slot** (**independent of** how many times the weights are re-scanned) | config C (misplaced, 29/26/4 slots per expert) | single core, **1 pass 55.4 vs 8 passes 54.2 GMAC/s** — re-scanned weights are free in L2/L3 |
| **crossing point** | about **5 slots** per expert | | the two formulas are equal |

Two conclusions that have to be remembered:

1. **"the CPU kernel is bandwidth-bound" is half a sentence** (T8 was half-corrected by T14): it only holds for
   configurations with **few slots per expert**. This box has no AVX-VNNI (`vpdpbusd`=0); the inner instruction count of
   AVX2 `maddubs` sets the compute wall, and **reordering the loops (fewer re-scans) measurably does nothing**.
2. **the true cost of the CPU leg is `max(byte wall, compute wall)`, while the rule only booked the byte wall**
   ⇒ config C is systematically underestimated ⇒ this to-do (T15) **has been frozen into P2** (a this-box-constant kind of
   job: written into "Limitations", not acted on).

**Straggler tax (a third wall on the same machine, the human wall)**: inside a frame the main thread has to do graph
replay / D2H enqueue, and it lands **on the same core** as OMP thread 0 (pinned to core0) ⇒ thread 0 runs at half speed,
while `static,32` **splits all 2048 rows up front, before the run even starts** ⇒ the whole parallel region waits on that
one worker.

| Schedule | exclusive min/p50 | with 1 core-stealer attached | with 2 attached |
|---|---|---|---|
| `static,32` (historical default) | 478 / 559 us | **839 / 905** (← the 811us measured inside a frame is this) | 862 / 865 |
| `guided` (new default, bit-for-bit equivalent) | 453 / 505 | **569 / 590 (−32%)** | 652 / 706 (−24%) |

End-to-end (11 paired rounds, L2 p50): **A −6.8%, C −12.4%**; with 1 core-stealer attached C drops **−17.5%
(0/11 rounds slower)**.
🚨 **The key reading**: "no core-stealing" end-to-end ≠ "no core-stealing" on the calliper — the in-frame main thread *is*
that stealer, so this cut holds without any attached process at all; the attached processes only amplify it for viewing.
**Transferable conclusion: on a machine with contention, "split it up front" gets its length decided by the straggler.**

---

## 2. The wall on the GPU side: you pay by weight bytes, and the calibration value must be measured

| Calibration item | What goes in | The price of putting in the wrong thing |
|---|---|---|
| GPU skinny GEMM effective bandwidth | **250GB/s** (measured) | 🚨 put in the peak 360 ⇒ GPU cost overestimated ⇒ **the rule dares not raise gc ⇒ 12% left on the table** |
| CPU-side effective bandwidth | **42GB/s** (DDR4, measured) | same as above, opposite direction |
| L3 budget | 0.75×12MB ⇒ **at most 3 residents** (real shape int8 2.53MB/expert) | all 8 experts resident = 20.2MB ⇒ every frame pays 5×63us more |

The most expensive appearance of this wall was the unsolved case in 乙-2: swiglu carries a cost of **+0.22~0.30ms that has
nothing to do with the number of slots on the CPU leg**, and A bs=1 (which reads not a single byte on the CPU leg) pays it
anyway. 乙-2b split it open (independent user replication, header loadavg **0.16** = the cleanest run in the ledger):

| Config | Shape | GPU half | CPU half | max(half, half) | End-to-end |
|---|---|---|---|---|---|
| A (**0 slots** on the CPU leg) | gemm | 0.177 | 0.001 | 0.177 | — |
| A | swiglu | 0.471 | 0.001 | 0.471 | ΔGPU **+0.294** / ΔCPU +0.162 (hidden by the overlap) / Δend-to-end **+0.298** |

Root cause: **the hot experts' weights are stored as fp32**, so every frame reads 125.8MB ⇒
**415us / 126MB = 303GB/s, already this card's read ceiling**.
⇒ anything byte-bound can only be fixed by reading fewer bytes — that is blog3's precision cut, and blog2's fourth leg.

---

## 3. Placement rule v1 → v2: what was missing was never the coefficient, it was the other half of the cost

| Version | The rule | The question it answers | What it cannot answer |
|---|---|---|---|
| **v1** | "stop once the weights touched on the CPU side are ≤ **0.5×L3**" | is the CPU side still roomy | no bs term, no price set on the GPU half ⇒ always picks gc=5 |
| **v2** | `gc = argmin_gc max(CPU half, GPU half)` + **a tie-break rule** | is buying one more GPU expert worth it | no price set on dtype, traffic/quality (⇒ P1) |

Two design decisions deserve their own paragraph:

- **Taking the max, not the sum**: the two legs already overlap (that is the model in §0). Adding them ⇒ double charging
  ⇒ the rule becomes systematically conservative.
- **The tie-break rule is the other half, not a detail**: at bs=16/A the model puts gc=2 and gc=3 **1us (6%)** apart
  while measurement puts them **12.8%** apart ⇒ at that scale the model **has no resolving power**. So inside an 8% band
  take first the smaller CPU-byte count, then the smaller gc — **only then do all eight cells land on the measured optimum
  or inside the noise band**.

**Why "retuning the coefficient" cannot save v1** (T13, measured): to make v1 pick gc=3 for A you would have to squeeze
the budget below 4MB, and that pushes C to gc=5 (measured 2.6% slower than its optimum). **No single coefficient can fix
both configurations at once — what is missing is the other half of the cost.**

### 3.1 The rule vs the measured optimum (this is the rule's entire bill)

**P0-② gc sweep (toy shape, 8 cells, exclusive-process basis)**: the rule **always gc=5**, while the measured optimum
moves **between {0,1,2,3,7}**; **8/8 cells are overpriced by +16.4%~+95.0%** (the worst: swiglu bs=1 A, optimum gc=1 =
**0.279ms** vs the rule's gc=5 = **0.545ms**, i.e. 95.0% dearer).
**The direction is wrong too**: with C misplaced and bs≥32 the measured optimum is **gc=7** (more aggressive than the
rule), at bs=1 it is **gc=0/1** (more conservative than the rule).

**P0-④a three-way exchange (real weights, placed correctly; "dear" = the rule's p50 relative to the measured optimum)**:

| bs | fp32: rule / measured optimum | bf16: rule (rule's true value) / measured optimum | what the misplaced cell owes |
|---|---|---|---|
| 1 | 0 / 0 (🚨the rule's input degenerates: bs=1 has only **2 slots**) | 1 (true value 2) / 3 | — |
| 32 | **5 / 5 = dead centre** | 5 (true value 6) / 7 ⇒ dearer by **+11.4%** | C cell fp32 **+10.7%** (the bf16 side is unstable across processes, not quoted as a point) |
| 128 | **5 / 5 = dead centre** | 5 (true value 6) / 7 ⇒ dearer by **+27.0%** | C cell fp32 **+101.5% / +106.5%** (two processes) |

Reading: v2's two legs are **already dead centre** on fp32 at bs≥32; bf16 discounts the GPU half and the rule **does dare
to give a larger gc** (true value 5→6), but all three times it is still one notch short — **that is the half-price v2 is
missing (blog3's precision cut)**; and a misplaced cell owing +101~106% shows that **the rule does not look at the hit
distribution at all** (the next leg).

---

## 4. The first missing leg: dtype pricing (measurement already in hand, the rule is one notch short)

④a really wired `gpu_weight_bytes` to dtype (gate: the `规则选gc` column moves 5→6 from fp32→bf16 while all idx
differences stay 0 ⇒ the precision cut did not leak into the routing). **The rule dares to give a larger gc but not the
measured optimum**: because the rule books the GPU half at the **byte price**, while in measurement the GPU half only
cashes in when it is on the **critical path** (see blog3's −8.1% vs −52.8% contrast).
**This leg is already on the P1 list and the data backing is sufficient; what is missing is the second calibration inside
the rule.**

---

## 5. The second missing leg: quality and traffic (a new conclusion that fell out of P0-③)

**Misplacing is not only slower, it also costs quality** (C/A KL ratio at the same bs and the same gc):

| bs | gc=3 | gc=5 | gc=7 | does top-1 really fall |
|---|---|---|---|---|
| 32 | **11.6x** | **20.3x** | 5.7x | gc=5: **100.00 → 96.88** |
| 128 | **10.3x** | **21.8x** | **13.8x** | gc=3: 100.00 → 98.44 |

The root cause is mechanical: misplacing drops more slots back onto the int8 leg (gc=5 bs=128: CPU leg
**22 → 177 slots**). **Quality falls monotonically with gc, but sub-linearly in the slot count of the CPU leg**: at
bs=128, going from 256 slots → 1 slot multiplies the slots by **256x** while KL only rises **81x**
⇒ **quality is dominated by whether the one or two hottest slots made it into the GPU**, and it flattens quickly after.

⇒ **Placement rule v2 therefore acquires a second leg: it used to price latency only, now quality is priced by CPU-leg
slot count as well.**
But **the rule cannot eat KL itself** (§blog1 red line ①: in the gc=0→1 cell L2 and KL point in opposite directions) —
**what it can eat is the hit distribution (slot count × hit%)**, and KL serves only as acceptance evidence.
> 🚨**Retroactive correction (P1-②c, measured; see §8)**: this sentence is【right in direction, wrong in dimension】—
> the hit distribution is only a **proxy**, not the ledger. In the artificial-drift segment the within-segment Spearman
> between "covered traffic %" and measured KL collapses to **+0.60**, while "the sum of the CPU-leg experts' unit prices"
> stays at **+1.00**; the member set chosen by maximum coverage is **1.60x dearer** in KL than the one chosen by unit
> price in the drift segment. The real ledger is "who makes int8 pay more KL"; hits are only the shadow it leaves on this
> machine when nothing drifts. The old text is kept exactly as it was, with no silent rewrite (methodology: a mistake must
> leave a trace).
And the hit distribution on this machine's real source is a **measured, natural skew**: hits
`[42.3, 13.2, 17.7, 6.8, 1.4, 12.6, 6.0, 0.1]`, **hottest/coldest = 433x**, coldest expert **0.1%** — that is the most
direct motivation for "dynamic hot-expert placement" (P1 first priority).

---

## 6. The third missing leg: zero traffic still costs money (the dense tax)

P0-④a put a first price on this money: at bs=1 placed correctly, the `gc=3 → gc=8` cells carry **exactly the same
traffic** (GPU leg 2 slots in both, CPU leg 0 slots in both), p50 fp32 **0.257 → 0.434ms**, bf16
**0.215 → 0.317ms** ⇒ **every extra resident expert with zero traffic costs 35us (fp32) / 20us (bf16)**, and five extra
residents cost **69% / 47% more frame time in exchange for 0 hits**.
⇒ booking the rule by "hit% × weight bytes" is correct, but **the implementation must actually charge by traffic**;
today's dense code path makes the rule's books and the measurement disagree. (The precision Δ in the same cell is
**−16.1% (7/7 rounds)** — values untouched bit-for-bit, time still paid; that is exactly a **reclaimable upper-bound
probe**, see methodology 25.)

---

## 7. What a change to the rule has to look like to count as correct (the acceptance test is written here, not just the idea)

| Missing leg | Data backing (already in hand) | The action that is missing | Acceptance test |
|---|---|---|---|
| dtype pricing | ④a: rule's true value 5→6; the measured optimum reaches 7; pairing-usable rows −48~−53% | GPU-half cost = bytes ÷ **nominal bandwidth × cash-in rate (critical path)** | at the three bs the rule's cell = the measured optimum or lies inside the noise band |
| ~~traffic/quality pricing~~ | ④a/③: C/A KL **5.7~21.8x**; hit skew **433x** | the objective eats **slot count × hit%**, not KL | 🚨superseded by the unit-price row below: hits are only a proxy (drift-segment Spearman only +0.60) |
| **unit-price pricing (§8, already in hand)** | ②c: `Σu_e` with **0 free parameters**, within-segment Spearman **+1.00** in all three segments (covered traffic only +0.60 in drift); endpoints all-CPU/Σu = 0.902/1.070/0.920 | the objective eats **the sum of the unit prices of the experts on the CPU leg**; the price = one offline **8-float table** | 3/3 segments (including the artificial drift) land on the gc=3 member set with the **smallest measured KL** in that segment |
| zero traffic not charged (乙-2d) | ④a: 35/20us per expert; probe expectation **÷2~÷2.5** | pre-record a few graphs for this frame's hot subset, zero host synchronisation | bs=1 A cell `gc=8` p50 ≈ the `gc=3` cell |
| CPU leg takes `max(two walls)` | T14: compute wall 7us/slot; crossing ≈5 slots | **T15 already frozen into P2** (this-box constant) | does not enter P1, written into "Limitations" |

**One last piece of discipline** (methodology 14, bought with two self-inflicted wounds): **the measured half of a
split-account table (a diagnostic) may not be fed straight into the rule as input.** An isolated tight loop measures the
warm-cache steady state and, at the cache boundary, systematically underestimates true in-frame cost; what the rule eats
is **calibrated bandwidth + the tie-break rule**. When a diagnostic table prints "the model is not enough", do two things
first: re-run with the pre-change backup (rule out a regression) and go and check what the rule **actually** eats (rule
out a confused basis).

---

## 8. The fourth missing leg: unit price (P1-②c/②d, measurement already in hand)

§5's leg said "quality is priced by **CPU-leg slot count × hit%**". P1-②c held that down and measured it again, and the
conclusion is: **hit% is only a proxy; the real ledger is each expert's own int8 error unit price.**

### 8.1 How the price is measured: one 8-float table, 0 free parameters

`u_e` = the KL measured when expert e alone is put on the CPU leg (`gc=E−1`, the last slot pinned, everything else on the
GPU); `kappa_e = u_e / 命中_e` = the expert's intrinsic error. Excerpt from the readme segment (`2e7f1cb2`: 22 engines /
66 cells / 196s / 27 archive anchors):

| eid | hit% | `u_e` = KL(only it on the CPU leg) | `kappa_e` | relative to the cheapest |
|---|---|---|---|---|
| 0 | 42.3 | 1.8054e-05 | 4.269e-05 | 227x |
| 2 | 17.7 | 2.7506e-05 | 1.556e-04 | 346x |
| 3 | 6.8 | 3.9594e-06 | 5.792e-05 | 50x |
| 4 | 1.4 | 7.9542e-08 | 5.818e-06 | 1x (the cheapest) |
| 7 | 0.1 | 3.4673e-07 | **3.550e-04 = the most expensive intrinsic error in the table** | 4.4x |

**The two hottest ones (0, 2) really are expensive, but "expensive" and "hot" are not the same ordering**: eid7, with
only **0.1%** of the traffic, has the most expensive intrinsic error in the whole table; the `kappa` spectrum spans
**61x** (3.550e-04 / 5.818e-06), whereas §5's rod eats the **hit spectrum** (433x).

Three rods with **0 free parameters** fight each other; what is compared is **predicted KL ÷ measured KL**
(1.0 = on target):

| Rod (how it predicts a member set's KL) | Within-segment Spearman: readme / input.txt / drift | Ratio over the 23 cross-segment cells |
|---|---|---|
| M0 covered traffic% (= §5's leg) | +0.98 / +0.98 / 🚨**+0.60** | geometric mean 1.03, range **0.15~1.86** (22/23 inside [0.5,2]) |
| **M1 unit price `Σu_e`** | **+1.00 / +1.00 / +1.00** (n=9/10/6) | geometric mean **1.01**, range **0.92~1.11**, **23/23** inside [0.5,2] |
| M2 the ledger's old L2rel% basis | +0.98 / +1.00 / +0.71 | — |

The two endpoints line up too (harder than a correlation coefficient): **all-CPU measured KL ÷ `Σu_e` =
0.902 / 1.070 / 0.920** — int8 error is essentially additive in the KL domain, so the "price" is not fitted, it is
**measured addition**. The harshest cell: only `{7}` left on the CPU leg (0.1% of the traffic), and M0 gives only
**0.15x** of the measured value (**underestimating by 6.7x**) — **the error of a counting-style mechanism quantity drifts
systematically with "who is left on the CPU leg", and no amount of coefficient tuning brings it back.**
That is a missing leg, not a missing calibration.

### 8.2 After the criterion switched from【hot】to【unit price】: 3/3 segments land on the member set with the smallest measured KL

P1-②d changed only the quantity the criterion compares, from hit share to `u_e`; the skeleton (capacity / cooldown /
one-in-one-out / migration mechanism) untouched by a single character:
**11 strategies × 3 stages = 33 rows in the same cell**, gate ② (the engine after migration vs rebuilt from the runtime
`slot2eid` = bit-identical) passed everywhere, gate ③ (the CUDA Graph does not re-record) 33/33.

| Segment | Unit-price-optimal member set (`price_*` final state, measured KL) | Coverage-optimal member set (the old "floor" cell) | Which one is cheaper |
|---|---|---|---|
| readme | `{0,2,3}` = 9.370e-06 | `{0,1,2}` = 9.711e-06 | the coverage-optimal set is **3.6% dearer**, and its covered traffic 73.1% > 66.8% |
| input.txt | `{0,2,5}` = 6.145e-06 | the same set | tie |
| drift | `{0,2,3}` = 1.301e-05 | `{0,5,6}` = 2.079e-05 | 🚨the coverage-optimal is **1.60x dearer** |

⇒ In all three segments the unit-price optimum is exactly the member set with the smallest KL among the gc=3 member sets
actually measured in that segment (3/3, including the artificial-drift segment).
⇒ **The cell called "the floor" in the old documents must be renamed【coverage-optimal】**: its members were chosen by the
hit histogram, it is **not a KL minimiser** — `KL/floor < 1` is a **legitimate reading** (§7 and reading hint ⑧ already
reserved that sentence for it; now it has a mechanistic explanation).
⇒ The starting lineup / god's-eye view `{0,2,1}` is **not the unit-price optimum in any of the three segments**: price
gap = **8.6% / 44.1% / 21.7%** — however well the static placement is done, there is still money left on the table.

**The price tag and its cashing-in** (this is the real self-verification of that cut; replication round `27f8c744`):
for the `{0,2,5} → {0,2,3}` migration the criterion printed a predicted **8.0% drop** before it fired, and the measurement
gave `1.414e-05 → 1.301e-05` = **a 7.99% drop**; the price-gap column printed by `price_cal@input.txt` says **30.3%**, and
the r2 archive's measured values say **30.43%** was collectable (8.833e-06 → 6.145e-06);
for the unit-price-optimal cells `M1 ÷ measured` = **0.92 / 1.01 / 1.01**.
⇒ `thr_rel = 5%` ("if the predicted drop is below the price of one migration, you should not act"; one migration on this
box measured 13.8~22.8ms = **2.6~4.4 frames**) turns from a slogan into **a numerical criterion checkable cell by cell**.

### 8.3 What about cross-domain: one refutation, plus a much cheaper substitute

- **Refutation**: the cross-segment Spearman mean of `kappa_e` is only **+0.16** (−0.38 / +0.76 / +0.10) ⇒ **when you
  change corpus you may not look up the same kappa table**; the `u_e` ranking in the same cell is far more stable
  (+0.62 / +0.88 / +0.81) ⇒ **what should be looked up is `u_e`, not bare kappa**.
- **A cut saved along the way**: `price_cal` (look up the `u_e` table from the calibration segment) and `price_hit`
  (re-price by `kappa_e ×` the share inside the online window) give **bit-identical final member sets and KL** ⇒ online
  re-pricing **does not change decisions** ⇒ the planned "shadow probe" (P1-②e) **will not be opened**.
- What remains is only the change-corpus cell: `price_cal@input.txt` gives KL/A = 0.85, while an oracle using that
  segment's true prices gives 0.59 (still 30.3% away). The substitute is much cheaper: **keep one 8-float price table per
  domain, offline**, and write no online probe.

> **In one sentence**: "who is hotter" is a proxy; "who makes int8 pay more KL" is the ledger — and this ledger holds
> **only 8 floats** on this machine.

---

## 9. The fifth missing leg: the price tag of migration — once it is measured, the fourth leg's gain zeroes out into "one-off placement at start-up"

§8 replaced the criterion from "who is hotter" with "who makes int8 pay more KL", and 3/3 segments landed on the member
set with the smallest measured KL (that cut of `27f8c744` also left the reconciliation of a predicted **8.0%** against a
measured **7.99%**). This section is the cost side: **how much one switch costs**, and a not-very-flattering ending —
**under the metering-round basis, not one of those payments was ever actually made.**

### 9.1 Three cuts at the price tag: every one overturns the basis of the previous

| Cut | Archive | How it was measured | Conclusion | Landing |
|---|---|---|---|---|
| i | `6eabc1de` | flip-window wall-clock difference ÷ the frame-length ruler | sync price tag **19.30 frames/switch** × frame length **3.911ms** = **75.5ms/switch**; while flip's own p50 is only **15855µs** (≈4.1 frames) ⇒ **only 1/5 of it** | booked (the bet was that G1 gets refuted): the money is not in the flip, but what the other 4/5 is stayed unanswered |
| ii | `0e305a89` | the window cut into C (negative control) / H (recovery window) / **M (steady state)** / P (residual) | sync `C 0.30 / H 21.76 / **M 64.80** / P −6.92 / T 79.94` ⇒ **M/T = 81%** | **I2** ⇒ 🚨**the wall-clock price tag of table 3 is void as a basis**: it booked the entire steady-state placement penalty of the flipped state as switching cost; `thr_rel` may not be recalibrated against it |
| iii | `22ff0967` | each round emits two extra segments【entirely inside the flipped state】basef/swapf ⇒ `once=(tag+tagf)/2` cancels the steady-state term **by design** | sync **once_main = 4.89 frames/switch = 19.9ms**; async_staged once = 1.09 frames ⇒ **ratio_once = 0.22**; the steady-state penalty listed separately `sync 4.645 / async_direct 5.230 / async_staged 4.436 ms/frame` | **J2** (the bet on J2 cashed in) ⇒ no old table is touched, but from now on "one-off" and "steady state" must be written separately |

The self-consistency check of the third cut is the crux: the cancellation only holds if the per-frame penalty is a
constant, so both rulers are normalised to **ms/frame** before being compared — self-consistency difference sync **9%** /
async_direct 6% / async_staged 7% (<50%) ⇒ the cancellation holds, and only then is `19.9ms` entitled to be used.

**This leg is now inside the rule**: the switch-on/off criterion is `thr_rel = 5%`, and what it needs is "how many frames
does one switch cost" = the **one-off price tag**; the steady-state term has **the opposite sign** in a real decision
(switching usually makes things better), so it cannot serve as the threshold.

### 9.2 And then: the gain side reads as "one-off placement at start-up"

`P1-②附2附` (`d9d54f5b`, n=25) runs a **decision counter** rather than a latency column, so it is not afraid of
bimodality, and its conclusion is the hardest one here:

> **0 decisions per arm in the metering rounds** (A1/A2/A3/C1/H1/H2 all of them 0 times/0ms); across the whole run the
> only physical migrations were H1 2 times/26ms and H2 1 time/13ms.

⇒ that sentence in the earlier dyn tables, "dyn saves 20%~27%", happened **outside the metering rounds** (in the placement
done at start-up), and under the metering basis it can only be read as **【one-off placement at start-up】**: pick the
resident set by §8's unit-price criterion, **then move nothing for the rest of the frames**.
This is not evidence that "dynamic placement is useless" (see below) — it is evidence that **it was never assessed**.

### 9.3 An upper-bound anchor only says how much is collectable, not that you can collect it

The `P1-②附6` series (drifttrack) deliberately builds a corpus whose membership really drifts, and installs a god's-eye
anchor O1 (the probe knows each block's answer and has zero reaction latency, but pays the same flip):

| Quantity | Value | Source | Reading |
|---|---|---|---|
| O1 GAIN | **0.6866** ⇒ placement-gain ceiling **31.3%** | `48499173` | 🚨**corroboration only**: it comes from a ruler that failed its pre-check |
| per-frame collectable ceiling | blk1 **3.168** / blk2 **3.022 ms/frame** | same, table 5 | comparable only because it is in the **same units** as the r3 price tag |
| payback frames | sync **6.3 / 6.6 frames**, async_staged **1.4 frames**, block length **24 frames** | same + `22ff0967` | ⇒ **mathematically reachable** (one migration pays back inside 1/4 of the block length) |
| H1 tracking rate T | **0.64** (across rounds 0.86 → 0.84 → 0.64 → **0.578**, the last cell being the **cross-process** median from `1f69a7cb`) | same, table 3 + r5 | the corroboration column of the same ruler, not part of the claim; the four cells 0.86/0.84/0.64/0.578 fall monotonically, and r5 is the first to drop below `T_LO=0.60` — 🚨but that round's landing is booked ⇒ **this position may not be read as a conclusion** |
| O1 GAIN (r5 cross-process median) | **0.7285** ⇒ ceiling **27.2%** | `1f69a7cb` | the same corroboration reproduced at the cross-process level (31.3% → 27.2%, same order of magnitude); 🚨still **corroboration only** |

"mathematically reachable" is the strongest sentence this section can safely say aloud: **price tag 19.9ms (≈4.9 frames) <
block length 24 frames × collectable ceiling 3.17ms/frame ⇒ the excuse "too expensive to bother" does not exist.** As for
whether the dynamic mechanism can actually collect that gain, this box's ruler **is not entitled to answer** (four
bookings; and that ≈8% offset was re-ruled in r5 as **per-instance one-off**, `σ_inst≈1.6%`, worst per-process cell range
`4.73~7.42%` — all details in blog1 §9 and README §6.3.)

> **The fifth leg's bill**: price tag = one-off **19.9ms** (sync) / ≈4.4ms (async_staged, 🚨needs the user's nod before it
> becomes the default), with the steady-state term **4.645ms/frame** booked separately; the switch threshold `thr_rel=5%`
> eats only the one-off payment.
> On the gain side the only citable thing today is **placing the resident set correctly by `u_e` at start-up**; the word
> "adaptive" **has never once passed a valid assessment in this repository.**

---

> Data sources: `results/2026-11-19_T14_ab.txt`、`…乙-2_swiglu_realshape.txt`、`…乙-2b_halves_bs1.txt`、
> `…P0-②_base_4paths_user7r.txt`、`…P0-③_metrics_top1-kl.txt`、`…P0-④a_swap_3way.txt`、`…P0-④a附_bs128_crossproc.txt`、
> `…P1-②d附_dyn_gc3_bs512_r3_coolfix.txt` (`27f8c744`), `…P1-②_dyn_gc3_bs512.txt` (`bd8eae3b`),
> `…P1-②b_dyn_gc3_bs512_startshuffle.txt` (`8f33ba36`), `…P1-②附2附_dyn发数标度_dynpair_r2.txt` (`d9d54f5b`),
> `…P1-②附3_迁移价签重定_swapcost_r1.txt` (`6eabc1de`), `…P1-②附4_价签事后归因_swapcost_r2.txt` (`0e305a89`),
> `…P1-②附5_一次性价签_swapcost_r3.txt` (`22ff0967`), `…P1-②附6_真漂移追踪_drifttrack_r1.txt` (`c299b2b9`),
> `…P1-②附6附_磨尺子_drifttrack_r3.txt` (`96ecadf1`), `…P1-②附6附2_换口径_drifttrack_r4.txt` (`48499173`),
> `…P1-②附6附3_B1换装置_drifttrack_r5.txt` (`1f69a7cb`) (all of them run by hand by the user).
