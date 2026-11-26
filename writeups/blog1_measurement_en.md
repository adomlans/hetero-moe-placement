# blog1 · Metrology — what "telling the truth" costs on this machine

> English version of `blog1_measurement.md`. It is a **translation**, not a second text: every number, every `md5`-style
> evidence pointer and every table cell comes from the Chinese original verbatim; where the two disagree,
> **the Chinese text wins**. Term map for the field names kept in Chinese: `底_pair` = pair floor · `底_序` = order floor.

> **Writing red lines (shared by the four write-ups, pinned in the header):**
> ① **First-order and second-order quantities in two columns, always together**: relRMSE/L2rel and top-1 agreement / KL / TV
> must appear together; "error <1%" is not allowed to stand as a sentence of its own (methodology 22).
> ② **Quality is reported as a relative quantity only, and always next to the reference row**: swapping the real layer-8 MLP
> for an "fp32 dense 8-of-2 MoE" is itself worth KL 1.897e-01 (bs=32) / 8.854e-02 (bs=128) ⇒ **668x / 871x** more expensive
> than the harshest engine cell.
> ③ **Memory must say whether it is [peak MB] or [steady-state MB]**: at gc=5 both dtypes peak at 186.1MB; only the steady
> state separates into 169.3 (fp32) vs 109.8 (bf16) (methodology 24).
> ④ **The three-way exchange uses only the two P0-④a archives**; every number points back to a verdict table run by hand by
> the author (`results/`, with md5 / date / header loadavg).
> ⑤ **Δ in the paired section is read only from rows marked [pairing usable] ✓** (both legs' L3-stepping within ±10% and the
> faster round 7/7 or 0/7); Δ on a ✗ row is the noise floor, not an effect size.
> ⑥ **Cross-process bimodal cells are quoted as an interval, never as a point**: A bs=128 `gc=3 bf16` =
> **1.35 / 1.74ms bimodal** (within-round σ=1%).

## 0. What this piece answers

OPTLOG §4 has 26 methodology entries. They are not style preferences — **each one is a refund slip for a round that was burned.**
This piece sorts them into four things: **three error axes, two kinds of memory, two kinds of scales, ten gates**, each with one
measured number plus a pointer back to the archived evidence.
One-sentence conclusion: **on this i5-10400 + RTX3060, the cost of "saying one sentence truly" is about a third of the cost of
"building the system".**
(§9 is appendix-like: it takes the four unsettled bookings as its measurement object and shows how a **pre-registered decision
tree** separates "no effect was measured" from "this machine cannot measure the effect".)

---

## 1. Three error axes: first say whose noise it is

| Axis | Where it comes from | Measured size | The only countermeasure | Source |
|---|---|---|---|---|
| ① **within-round sampling noise** (σ% per round) | the desktop steals cores, turbo, core-pinning order | the same binary measured at **0.288 / 0.231**; σ% per round 0~23% inside the ④a table | `ROUNDS≥7` + same-process rotation + per-round pairing + p50; **min-of-N forbidden** | §4-1/2 |
| ② **same-process co-residency** (L3-stepping) | several models resident at once stepping on each other's cache | bs=1 alone **0.250 → three models resident 0.274 (+9%)**; the first version of P0-② built all 9 models up front (72MB total on the CPU side) ⇒ absolute values inflated overall by **17~37%**; the largest L3-stepping in the ④a paired section is **+44%** | latency from the rotation section, memory from the exclusive section — **the two columns do not share a measurement section**; the co-residency section prints its own L3-stepping column | §4-16 |
| ③ **cross-process** | page / physical address layout, huge pages, L3 residency luck, CUDA Graph pool allocation order — all of it fixed at the instant the process starts | A bs=128 `gc=3 bf16`: main table **1.349ms (σ% per round = 1%, all 7 rounds agree)** vs an exclusive-process replication round **1.735ms (σ=12%) ⇒ a cross-process gap of 28.6%** | any absolute value that enters the text needs a **second-process replication round**; cross-process gaps >8% are flagged bimodal and quoted as intervals | ④a appendix / §4-26 |

**These three axes do not cover each other** — that is the most important sentence in this piece:
`ROUNDS + σ% per round` measures only ①, and same-process rotation pairing also measures only ① (the same piece of luck
repeated 7 times); the co-residency penalty is ②;
**③ can only be measured by opening another process.** So the timing discipline of this project is three sentences, not one:
**≥7 rounds paired at p50 within a round / co-residency inside one process must print its own cost / any absolute value in the text needs second-process evidence.**

Bimodality candidates are not found by blind scanning — there are morphological features: **CPU residency ≈ L3 capacity**
(12MB on this box: A gc=3 has 12.7MB CPU-resident) or **a very large number of slots on the CPU leg**
(C gc=5/7: 177/102 slots). All three bimodal cells uncovered by the ④a appendix are **in bf16, where the CPU leg is heavy**.

**Why min-of-N is forbidden**: it rewards variants with high variance. T8's "−16%" was manufactured exactly that way;
the P0-② smoke run used `ROUNDS=2 ITERS=3` and manufactured a fake "② is 27% faster than ①", which disappeared once
`ROUNDS=7` plus odd/even order swapping were used. **Noise floor**: A misplaced 2~3%, C misplaced 8~18%
(C's absolute values move with loadavg, so only relative values inside one run may be read).

---

## 2. Two kinds of memory: high-water ≠ steady state (one accident, two columns bought)

| Column | Definition | Does lowering precision show up in it | Measured (gc=5) |
|---|---|---|---|
| **peak MB** | window high-water after `reset_peak` − idle baseline (the P0-② definition) | ❌ invisible | fp32 **186.1** vs bf16 **186.1** (bit-identical values) |
| **steady-state MB** | what is still resident after warm-up (the P0-③ `CUDA-resident MB` definition) | ✅ only here | fp32 **169.3** vs bf16 **109.8** |

The cause is plain: **the graph-build / warm-up instant is fp32 staging** (`real_src` emits a `.float()` copy first, then casts).
⇒ reporting only the high-water mark erases the bf16 memory gain to 0; reporting only the steady state loses the fact that
**graph building also needs 186MB momentarily** ⇒ **report both columns**, and say which one is being quoted (methodology 24).
From ④a on the two columns carry the same names on the same measurement basis, which incidentally settled the old account
between P0-②'s peak column and P0-③'s weight-residency column that "did not add up".

**This column was polluted four times, and each time it left one rule behind:**

| Accident | Symptom | Fix |
|---|---|---|
| high-water not reset | four baselines printed **the same 405.5MB** (it was measuring the high-water mark of every live tensor at that moment) | `reset_peak_memory_stats()` before the window |
| idle baseline not checked | the "memory saved" **column shifted by one row** (① saves −1 / ③ saves 40 / ④ saves 65) | inside `win()`, idle >2MB **alarms on the spot** |
| the autograd graph holding weights (a real bug) | while the output tensor is alive, **gc×8MB** is never returned: 41.0MB still there after `del m`, only `del out` drops it back to 0.1MB | `requires_grad_(False)` at the end of the ctor: after the fix **129.3→89.3 / 201.6→137.6**, output **bit-identical** (methodology 18) |
| dead weights I built myself | only the `bmm` variant ever reads `_wstack/_wstack_gu`; under the cat variant it squats on **gc×8MB** for nothing | do not build it: pairwise bitwise difference 0.000000000 |

> "I manufactured a 200MB memory phantom for myself" was only caught by **comparing against someone else** (P0-②) —
> comparing against yourself never exposes it.
> A project whose selling point is "saving memory" has to get the denominator right first.

---

## 3. Two kinds of scales: deterministic vs timed; first-order vs second-order

**① Timing discipline does not apply to deterministic tables** (methodology 21). In P0-③ the header loadavg of the last cell
climbed to **1.47**, yet gate ④ cross-round KL came out with a **bitwise difference = 0.000e+00** — with the same input it simply
has no notion of "noise". The converse also holds: **do not add rotation pairing to a deterministic table**, it only makes you
believe you are doing statistics. `ROUNDS` inside the `metrics` table exists only to serve gate ④.

**② First-order and second-order quantities are not interchangeable** (methodology 22). P0-③ bs=32: L2rel
`5.9174% → 0.5121%` (a factor of **11.6x**), while KL of the same two cells goes `2.841e-04 → 9.376e-07` (a factor of **303x**).
Worse, **they can even point in opposite directions** (P0-④a, correctly placed, gc=0→1):

| bs | L2rel | KL(P‖Q) | How to read it |
|---|---|---|---|
| 32 | 5.9174% → 4.8139% (**−19%**) | 2.841e-04 → 3.498e-04 (**+23%**) | first order down, second order up |
| 128 | 5.5114% → 3.7021% (**−33%**) | 1.016e-04 → 1.049e-04 (**+3.2%**) | same, smaller amplitude |

⇒ "error <1%" **implies nothing** at token level; any accuracy claim must report both columns;
**KL is not monotone in gc** ⇒ the placement rule **must not eat KL as input** (the "quality leg" in blog2 §5 runs on slot
counts, not on KL).

**③ The measurement basis must be printed on the table** (methodology 23). Gate ① of P0-③ originally differed by
**4.77e-07**, and it was not a bug: the same piece of network takes **a different GEMM tiling / reduction order** at T=1 and
T=512. The fix was to build both into the rig: every cell captures freshly at the same length via `capture_from(ids_row)`, and
the archived copy was **demoted to a "pairwise-difference reference column"**.
The worst option is to pretend it is 0 — the relRMSE of the first 30 cuts fell into exactly that hole, using one measurement
basis as the denominator of another.

---

## 4. Ten gates: each one must be able to say what it once caught

| Gate | Assertion | What it actually caught (not decoration) | Source |
|---|---|---|---|
| **⓪ cross-bench reconciliation** | the new code recomputes, cell by cell, **the 60 KL values the old code archived in `results/`** | first time it was switched on: **0 cells** disagreed, largest relative deviation **2.9e-04** (the archive keeps only 4 significant digits, threshold 2e-3) ⇒ the timing / memory windows did not pollute the numeric path, which is what licenses that ④a table to exist | P0-④a |
| **① splice bitwise** | the bitwise agreement across the layer-8 MLP output before/after the overwrite must be 1 | it is what dragged out the **4.77e-07** between freshly captured x and archived x and demoted it to a reference column (a measurement-basis accident) | P0-③④a |
| **② idx differences all 0** | dtype must not leak into the routing / quantisation front section | the old pit from 乙-1: `tf32` **changes the routing assignment**, it is not "slightly lower accuracy" | P0-③④a / 乙-2c |
| **③ three modes differ by 0** | sync / async / three modes share one front-section graph | 🚨 **it cannot catch a broken front-section graph** (all three modes share it ⇒ NaN in the front section would still print all-green). The only referee for the front section is **a head-to-head against eager's `m.route()` / `_quantize_act()`** (methodology 20) | P0-③ |
| **④ cross-round KL difference 0** | even a deterministic quantity has to prove its determinism | it proved core-stealing does not affect the quality tables (still 0 at loadavg 1.47) | P0-③ |
| **⑤ dtype zero-traffic self-check** | with 0 slots on the GPU leg, changing dtype must be bitwise 0 | what it proves is that **values** have no entry point — **this does not extrapolate to the timing columns** (methodology 25, see below) | P0-③④a |
| **⑥ co-residency difference** | with both variants resident at once in the paired section, values are still bitwise 0 | the whole table 0.000000000 ⇒ the paired section borrows time only, it does not pollute values | P0-④a |
| **red line H · poisoned neighbour** | `D∈{100,576,640}` non-divisible + `D=1024` divisible control, with the tail lying right next to a NaN buffer | the D loop in `route_fused` masked rows but not columns: **old-code bad values = 128**, `D=1024` bad values = 0 (which proves the trigger is **geometry**, not luck). This mine hid for more than ten rounds, because every cut and every red line had only ever tested divisible values | P0-① |
| **idle alarm** | idle >2MB inside a memory window alarms on the spot | "an extra +92MB of high-water after the peak column" got labelled **the referee's own cost** (`(E−gc)×10.1MB`), not the engine's | P0-②③④a |
| **[pairing usable]** | both legs' L3-stepping within ±10% **and** the faster round 7/7 or 0/7 | before this column existed I nearly read bs=128 A `gc=3`'s **+29.9%** as "bf16 is actually slower". Of ④a's **30 rows only 15 were usable** | P0-④a |

**One general discipline about gates** (methodology 4/6/8): **gates before latency** — a new switch may not be looked at in a
latency table until it proved bitwise equivalence (or until a tolerance is stated explicitly);
**do not write self-vs-self red lines** (when two names point at the same memory the difference is identically 0 — T12 nearly
shipped one); write "snapshot before the overwrite vs read after it" instead; **a gate must use the same parameters as the model
instance being compared** — 乙-1 compared against an fp32 reference taken at some other gc and reported a
**fake 12~15% error** (different gc = different weight draw = comparing two models).

---

## 5. Attribution discipline: when does a Δ count as "assigned to a leg"

| Discipline | Counter-example (my own) | Measured |
|---|---|---|
| **"a leg got more expensive" may only be read from the difference between [this leg has traffic] and [it has none]** (§4-12) | reading a +0.3ms that has nothing to do with slots as "the 18MB swiglu working set is standing on the byte wall" | at bs=1 the A variant's CPU leg does not read a single byte, and A→C differs by only **+6us** ⇒ the byte wall was hidden by the overlap of the two legs |
| **do not use a tight loop on an isolated kernel as the cost split** (§4-5) | what `bench_halves.py` measures is the warm-cache steady state | non-monotone across gc: **8MB→0.729 but 6MB→0.809** (a warm-cache phantom where CPU residency ≈ L3) |
| **the measured half of a cost-split table ≠ the input of the placement rule** (§4-14) | concluding the rule is broken because a diagnostic printed "model too small" | the `end-to-end − max(two halves)` residual is +0.150 at gc=2 and +0.09 at gc≥3 ⇒ what the rule eats is **calibrated bandwidth + a tie rule** |
| **before splitting operators, first ask "what dtype is this tensor", and take byte counts from the single entry point** (§4-13) | I nearly treated the GPU half as "kernels not optimised enough" and went off tuning layout | **415us / 126MB = 303GB/s**, which is already this card's read ceiling |
| **measure "fixed cost + slope" first, then decide which term to cut** (§4-15) | assuming "half the bytes ⇒ half the time" | the bare-mm `gu` variant gives −49%, the 4MB-class `down` only −28%, and the whole GPU half only −19%~−39% (inside it sits **≈0.05ms** of pre+D2H graph that does not move with dtype) |
| **an attribution table must pin placement (`FIXGC`)** (§4-9) | "everything on vs baseline" smuggled in a change of the placement rule | bs=32/A's `+T5` went from **−24.4% (mixed) to −1.4% (clean)** |
| **with different measurement bases, both numbers of the same conclusion can be true** (§1 note + LADDER 🚨) | assuming one of T14's −3.4% (ladder) and −12.4% (dedicated table) must be wrong | the ladder builds all 28 models in GPU memory and runs only 30×3 per variant ⇒ **long-tail gains get washed out**; both numbers are true, **a citation must name its basis** |

---

## 6. Five moments that were nearly written wrong (one sentence each)

1. **When a default changes, historical variants must be pinned to the old value** (§4-3, tripped over **4 times**):
   T11 `pre_impl` / T12 `l2_stream` / T13 `auto` / T14 `cpu_sched`. Pin it or the "T4 baseline" variant quietly eats the new cut's gain.
2. **Touching a `static thread_local` by symbol name inside an OpenMP parallel region** (§4-11, a real bug): worker threads get
   **their own empty copy**, and if it does not crash it **silently takes the wrong scale**. A parallel region may only accept a
   raw pointer that the main thread computed and passed in.
3. **perf is unavailable (paranoid=4)** (§4-7): there is no profiler ⇒ the timeline has to be rebuilt from same-process
   rotation + a pure-CPU calliper + pointer probes.
4. **An assertion written backwards** (the ④a smoke run): `E > max(GC_LIST)` disabled `gc=E`, the **pure-GPU upper-bound
   anchor**, outright (it should have been `E >=`) — the author's 10:09 paste-back was stopped at the starting line by exactly
   that. **Disabling the anchor row = the table loses its upper bound.**
5. **A red line must be able to prove it still bites** (the [tooth check] inside `red`): re-run the same cell with
   **TAIL forcibly off (= the old code)**; bad values must be ≠ 0. Otherwise "ALL PASS" only says you never tested that geometry.

---

## 7. Commit thresholds (a checklist you can copy straight into CI)

| Item | Threshold | What violating it cost (all of these happened) |
|---|---|---|
| header loadavg | **< 0.8** or a timed round may not enter the record | one whole P0-② paste-back was voided; the bs=1 cells and the ④pure-CPU leg are the most sensitive to core-stealing |
| rounds | `ROUNDS ≥ 7`, per-round pairing, report p50 and σ% per round | `ROUNDS=2` manufactured a fake 27% |
| aggregation | p50 / median, **min-of-N forbidden** | T8's fake −16% |
| effect size | above the noise floor (A 2~3%, C 8~18%) **and** a majority of rounds in the same direction | C's −1.4% was taken as a win for v2 |
| memory | `reset_peak` + idle baseline + `requires_grad_(False)`, and **say whether it is peak or steady state** | the same 405.5MB on four paths; the whole column shifted by a row; the 200MB phantom |
| equivalence | a new switch passes the bitwise gate before latency; the gate uses the same parameters as the compared instance | a fake 12~15% error |
| absolute values | any absolute value entering the text needs a **second-process replication round**; cross-process gaps >8% are flagged bimodal and quoted as intervals | the 1.35/1.74ms bimodality was quoted as a single point |
| pairing | only rows marked [pairing usable] ✓ are read | +29.9% was read as "bf16 is slower" |
| archiving | `results/` + md5 + date line + machine state in the header; anything run by a model is labelled "not part of the record" | `/tmp` is cleared at boot; a paste-back with no credential is the same as not having that table |
| criteria | **written down before the run** in the docstring and in OPTLOG, not changeable afterwards; the landing is evaluated by code, which prints the ex-ante prediction | afterwards picking a threshold that explains things = turning measurement into storytelling (§9) |
| ruler pre-check | if the empty control group's `底_pair ≤ TARGET` does not pass, the three questions behind it **may not be read**; the only landing is "booked" | "no effect was measured" misread as "there is no effect"; all four unsettled bookings here are exactly this |
| more than one ruler | the primary ruler is **designated before the run** (e.g. RULER=sum); the corroborating ruler is printed too but **does not participate in the landing** | accepting after the run whichever of the two rulers' tree fits = choosing the basis after the fact |

---

## 8. Cost accounting (the honest version)

**What discipline spent**: one whole P0-② paste-back voided plus one re-run; a whole round of the ④a appendix just to pin the
bs=128 absolute value (the conclusion did not change, but **methodology 26 was bought with it**); every new switch runs bitwise
equivalence before latency (about half the time); every red line added is a permanent subscription.

**What discipline bought**: **4 real bugs** (int16 saturation `max_abs 3.003→1.095`, the out-of-bounds tail-block read in
`route_fused`, the empty `static thread_local` copy, autograd holding gc×8MB); **2 memory overstatements**; **3 self-refutations**
(T8 moving D2H, T14 swapping loop order, the 乙-2 "swiglu stands on the byte wall" prediction); plus one thing that now looks
like common sense — **per-frame latency ≈ `max(CPU leg, GPU leg) + fixed serial host overhead`** (its price list is in blog2).

> This list is not virtue, it is a **price list**. And it still has a blind spot: it only governs "may this number enter the
> record", not "is this ruler entitled to answer this question" — that is what §9 is for. Read §9, then go to blog2: only there
> does this system itself get discussed, the two walls and the placement rule.

---

## 9. The pre-registered decision tree and that ≈8% offset (and its reclassification) — how a machine gets proven unable to resolve 5%

This section is the only place in this piece where **failure itself is the measurement object**. It answers a question every
empirical project runs into: **"I did not measure an effect" and "this machine cannot measure this effect" are two different
sentences — how do the same gates tell them apart?**

### 9.1 What a pre-registered decision tree is

Before the run, three things are written down for good in the docstring of `bench_*.py` and in OPTLOG; the code
**evaluates them automatically** at the end of the table and prints the landing:

| Part | Role |
|---|---|
| pre-checks (a/b/c/d/e) | **whether the ruler itself is fit**. Fail ⇒ none of the questions behind it may be read, and the landing can only be "booked" |
| questions (K1/K2/K3) | the actual scientific questions (does it keep up / does it lag / did it ever fire) |
| the ex-ante prediction | before the run, stake one landing; after the run print "delivered / refuted", with no changing one's story |

The key design: **pre-checks separated from questions**. When a pre-check fails, the question's numbers may not enter the text
however pretty they are — this is what blocks the most common self-deception, "the effect was not significant, let me re-read it
against a different threshold". This project paid four unsettled bookings for that (`c299b2b9` K0, `96ecadf1` M0 second time,
`48499173` M0″ third time, `1f69a7cb` M0‴ fourth time), and what it bought is: **the claim "adaptive" has still not been validly
assessed once to this day** — that sentence has evidence behind it, it is not modesty.

### 9.2 Four unsettled bookings, each dying somewhere else

| Cut | The single changed variable | Landing | Where the only red item was |
|---|---|---|---|
| r1 | build the rig | K0 | empty control `底_pair 6.91%` > 5%, and the oracle anchor did not pass the star |
| r3 | change only the ruler (★ star → ★2: both clusters must agree in sign) | M0, 2nd time | oracle ★2 blocks = empty; `底_pair 4.18%` passed but `底_序 16.00%` ⇒ threshold 32.0% |
| r4 | change only the estimator (per-block p50 → block sum) | M0″, 3rd time | `底_pair 8.41%` > 5% (n=48 ≥ N_MIN=45) ⇒ threshold 16.8% |
| r5 | change only the replication unit (pass → process) + randomise arm build order | M0‴, 4th time | median `底_pair 5.05%` > 5% (**short by 0.05pp** ⇒ the landing may not be ratified) |

r4 is the cut where **the change of basis worked everywhere it was supposed to**: bimodal cells **11/21 → 0/21**, the order term
`底_序` **16.00% → 6.94%**, the oracle anchor green for the first time. The only red item was still the same one. Only now is it
allowed to ask: **why do neither more passes nor a new estimator cure it?**

r5 changed one more dimension: **the replication unit went from "pass" to "process"** (5 processes × 48 passes each, arm build
order randomised), `底_pair` per process `[5.81/5.03/7.42/5.05/4.73]%`, median **5.05%** — 3.36pp below r4's 8.41%, **but it did
not cross 5%**, so the fourth booking; the difference is that this time a‴/b‴/d‴/e‴ were green simultaneously for the first time
and the only red item left was `c‴`.

### 9.3 The arm-identity fixed effect (r4's wording, **reclassified in r5**): an 8% **unrelated to ground truth**

Taking A1/A2/A3 (three engines with **literally identical configuration that never migrate at the same time**) as the empty
control group should read ≈0%. r4 read:

| Evidence | Value | What it rules out |
|---|---|---|
| **甲** A2/A1 over three blocks **−8.4 / −7.5 / −4.1%**, and the forward and reverse rounds of each block **agree in sign** | forward −8.9/−8.2/−7.5, reverse −7.9/−6.7/−0.6 | not order (forward and reverse in equal halves; agreement in sign means it is not order) |
| **乙** A3/A1 negative in all three blocks −2.6/−0.7/−1.6 | the second instance agrees in direction (🚨 per-block sign agreement holds only for A2) | not one machine's fluke |
| **丙** net-gain column A2 **+23.0**, A3 **+4.5 ms/pass** | two arms that never migrate "gain for free" | physically impossible ⇒ this 8% is not real acceleration |
| **丁** after dividing out each arm's **own** cross-block geometric mean the order curve leaves only ±2~3.6% | there is no 8% left in the residual | the 8% is not on the order axis ⇒ it travels with the **arm instance** |

Conclusion: the offset is drawn **at the instant the arm is built** (allocator / page table / graph pool / residency placement —
uncontrollable items); it **does not respond to the number of passes** (n=25→48 does not move), **not to the estimator**
(p50 and block sum are the same size), and **same-process pairing cannot cancel it** (both sides of the pair are the same
instance to begin with).

⇒ So "5% cannot be resolved" was **qualified** into a testable sentence: **at r4 this machine could not resolve arm-to-arm
differences below ≈8%**, and TARGET=5% sits exactly below that line. That is not a "the noise is high" excuse — it gives a
magnitude, a direction and a way to falsify it (change the replication unit from passes to processes and randomise arm build
order; if the 8% collapses, the qualification is right; if it does not, change machines).

> Two reminders about superseded wording: ① "arm identity" in the title of this section is r4's wording; ② that "≈8%" in the
> paragraph above is also an r4 value — r5 (`1f69a7cb`) settled the bet: **direction right, attribution changed, magnitude
> narrowed to ≲5~7%** ⇒ the current basis is always §9.4 and that row of README §7 — see §9.4.

### 9.4 Settling the §9.3 bet: the 8% halved, but the attribution changed

The bet, written verbatim (`1f69a7cb`, NP=5 × ROUNDS=48, header loadavg of all five processes <0.8), was "if the 8% collapses
when the replication unit changes, the qualification is right; if not, change machines". What came back is a **third kind of
answer**:

| Observation | Value | How to read it |
|---|---|---|
| median `底_pair` r4 `8.41%` → r5 **`5.05%`** | −3.36pp; per process `[5.81/5.03/7.42/5.05/4.73]%` | halved, **but it did not cross `TARGET=5%`** (short by 0.05pp ⇒ the landing stays booked, the same precedent as r2) |
| per-process worst-cell `\|Δ%\|` range `4.73~7.42%` | calibration of the same-configuration instance offset `σ_inst≈1.6%` | the offset is **not on arm identity**: shuffling + changing processes only **redistributes** it between instances, it does not change its distribution |
| hypothesis 乙 "the offset follows arm build position" | before A1 `+0.02%` (n=4) vs after A1 `−1.13%` (n=6), eaten by the range band `[−4.32~3.89]%` | it does not follow build position ⇒ randomising build order is not the cure |

So the qualification was reclassified from "**arm-identity fixed effect ≈8%**" into a sentence that is harder and less
comfortable: the offset is **a per-instance one-off** (the block of memory that the k-th allocation happened to land on —
allocator / page table / graph pool / residency placement), and it **does not respond to the number of passes, to the estimator,
to arm build order, and now not even to the number of processes**.
The "change machines" branch was therefore formally reached — and before spending the money for a new machine, a
**zero-run-cost accounting** was done first (`OPTLOG` §5.2i appendix 4; these figures were computed offline by the model,
**there is no verdict table to archive ⇒ not part of the record**): can folding instances into `k` replication layers buy
`c‴` green? The answer is **split**: **the pre-check can be bought** (at `k=3` the predicted median of `底_pair` is `3.39%`,
`P(green)≈98%`), **the landing cannot** (the question's `±1σ=0.127` is **5.8x** the decision-point spacing `0.022`, and still
3.9x after `k=3`; deciding it needs `k≈47` ⇒ `7 arms × 47 = 329` resident instances ⇒ RSS ≈57.9GB = physically impossible).
**What kills it is not the working set** (21 instances are only 5.8GB against 22GB available) but the division in item 4 of §9.5.

### 9.5 Four rules this section hands you

1. **Keep pre-checks and questions apart, and if a pre-check fails the question cannot be read.** Otherwise by the fifth round
   you will have changed the threshold to 10%.
2. **The empty control group needs more than two instances.** One instance's offset gets read as "this round was unlucky"; only
   two instances agreeing in direction, plus a net gain that comes for free, hold up the diagnosis that "the offset is not on the
   ground truth" (at the time called the identity fixed effect; the r5 reclassification is in §9.4).
3. **The cause of death has to be measured too.** "Landing booked" is not a wasted run: what r4 really delivered was the
   qualification "8% = the offset between same-configuration instances", which closed off the "add more passes" road outright and
   specified the rig change for the next cut. r5 then took it apart: **the magnitude half survived, the attribution half
   ("arm identity") died** ⇒ reclassified as a per-instance one-off. **The value of a negative result depends on whether you
   measured its cause of death.**
4. **Division inflates relative error — "is this ruler fit to decide this landing" has to be computed beforehand.** r5's question
   was `T = (1−GAIN[H1]) / (1−GAIN[O1])`: the numerator `1−GAIN[H1]` is only `0.1535`, so the **absolute** scatter of `GAIN`
   itself, `3.8%`, becomes a **20.7% relative error** once divided in; the denominator `1−GAIN[O1]=0.2715` only inflates by
   `3.68x`, a relative `7.1%`. Numerator and denominator are still two instances with **literally identical configuration**
   ⇒ this division does not remove noise, it amplifies. The rule: **compute "error band ÷ decision-point spacing" first; if the
   ratio is >1 the landing may not be read.** It is far cheaper than adding samples — and the road it killed would have run 329 instances.

> Methodology 26 says absolute values need second-process evidence; this section is its dual: **paired differences need
> "second instance" evidence too** — instances with the same configuration should not differ, and if they do, that difference is
> part of the ruler.
