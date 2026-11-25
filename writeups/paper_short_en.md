# Unit-price expert placement for heterogeneous MoE — measurements, price tags and one negative result on a 12-thread desktop

> Short paper, self-contained English version of `paper_short.md`. Deeper links (Chinese): derivations of the criterion and of the
> price tags in `blog2_two_walls_placement.md` §8/§9; the measurement discipline and "why this ruler fails" in
> `blog1_measurement.md` §9; index and reproduction in `README.md` §4/§8.
> **Every number carries the `md5` prefix (first 8) of the archived verdict table it came from.** All archives are in `results/`,
> all were run by hand and committed in full, with machine state in the header line.

---

## Abstract

On an 8-expert / top-2 SwiGLU MoE, with 3 experts resident on GPU and the rest resident on CPU as **int8 weights, fetched by
routing**, on an i5-10400 (6C/12T, dual-channel DDR4, measured 42GB/s, 12MB L3, no AVX-VNNI) + RTX3060, we show:

1. **Use "unit price", not "heat", as the placement criterion.** `u_e` = the KL measured when expert *e* alone is put on the CPU leg.
   0 free parameters; on 3/3 corpus segments the resident set it selects **is** the measured-KL-minimal one (`2e7f1cb2`, `3ce6836d`).
2. **The CPU side has two walls and the crossover must be measured.** The byte wall (weight bytes / 42GB/s) and the compute wall
   (AVX2 `maddubs` ceiling) cross at ≈5 CPU-resident experts; the GPU side pays by weight bytes (`ebf87c3b`, `0bdd9f58`).
3. **Zero-traffic experts still pay (the dense tax), and it can be removed.** After traffic-sparse batching:
   **35.8→3.6us (fp32) / 20.6→1.8us (bf16)** per frame, so `bs=1` decode becomes **34.1% faster than the pure-CPU path** of the same build (`64436f8a`).
4. **Negative result (the headline; scope = this paper's four attempts)**: **dynamic placement — moving experts as traffic drifts —
   was never validly evaluated once across four pre-registered, pre-committed decision trees.** All four trees died at the *ruler
   pre-check*. The final qualitative cause, called "arm-identity fixed effect ≈8%" in r4 and **reclassified in r5 as a
   per-instance one-off offset** (`σ_inst≈1.6%`, worst-cell per-process range 4.73~7.42%): engine instances built from literally the
   same configuration carry a stable offset unrelated to ground truth, and it **does not respond to the number of passes, to the
   estimator, to arm build order, or to the number of processes** ⇒ **arm-vs-arm differences of ≲5~7% are not resolvable on this
   machine** (`48499173`, `1f69a7cb`). The escape hatch "**buy one more layer of replication**" was **killed offline**: `k=3` resident
   instances raise the pre-check to `P(green)≈98%` but still cannot decide the outcome (the `±1σ` on the question's `T` remains
   **3.9x** the spacing between decision points; deciding needs `k≈47` = 329 resident instances = RSS 57.9GB = physically impossible),
   and what kills it is **not** the working set but the pair "cross-instance pairing + denominator inflation".

🚨 **This paper contains no "adaptive placement works" claim.** Every citable gain comes from **one-shot placement at start-up**.

---

## 1. System under test and the boundary of the claim

- Model: SmolLM shapes (`D=576, N=1536`), the 8 experts of layer 8's MLP used as 8-of-2, **real weights, real shapes**.
- **Referee** = a same-architecture 8-of-2 **fp32 dense MoE**, not the original model ⇒ quality is reported as **relative quantities
  only**, and only next to the reference row: "replacing a dense FFN with an MoE at all" already costs
  **668x / 871x / 1585x** (three bs settings, `fba72657`, `13f8740c`) — inside that reference frame, KL differences between engine
  cells can only be read as "between gc values / between dtypes".
- Memory is always reported as **peak / steady state** in two columns (high-water and steady state differ by ~200MB, `5bb8a840`).
- All constants are machine-specific (42GB/s, 250GB/s, 12MB L3, 6C/12T, a desktop steals cores).
  **What transfers is the method and the conclusion, not the numbers.**

---

## 2. Measurement method (the "rig" of this paper)

| Part | How | The accident it prevents |
|---|---|---|
| Same-process rotation pairing | all arms of one cell alternate **inside one process**, one forward and one reverse round | cross-process drift in absolute values being read as a gain |
| Aggregation | **p50 / block sum**; 🚨 **min-of-N forbidden** | `ROUNDS=2` manufactured a fake 27%; T8 a fake −16% |
| Commit threshold | header `loadavg < 0.8`, `ROUNDS ≥ 7`, per-round σ% | a core-contended round entering the record |
| Empty control group | 2~3 arms with **literally identical configuration that never fire** | — (the protagonist of §6) |
| Positive control | one arm that is **known to be slow** (member set deliberately wrong) | "the ruler is broken" being read as "there is no effect" |
| Upper-bound anchor (oracle) | the probe knows every block's correct answer, zero reaction latency, but **pays the same migration cost** | conflating "it tracks slowly" with "the ceiling was never there" |
| Gates ⓪~⑨ | bitwise equivalence, slot↔expert mapping re-checked, graph not re-recorded, counters and routing identical, placement independence, zero migration in the static arm … | caught **4 real bugs** |
| **Pre-registered decision tree** | pre-conditions + question + prediction, **written into the docstring before the run**, evaluated automatically by code, landing printed | picking, afterwards, whichever threshold explains the result |

**The key discipline**: pre-conditions (is the ruler fit) and question (the science) are **separate**. Pre-check fails ⇒ the question
may not be read, and the landing can only be "booked unsettled".

---

## 3. Evidence for the three legs

### 3.1 Two walls on the CPU side

The marginal cost of one CPU expert is `max(weight bytes / 42GB/s, compute)`, and the two walls cross at
**≈5 slots/expert** (`ebf87c3b`). **The exact caliber of `3.07x`** 【that round ran on the demo shape `D=1024/N=2048`:
gemm one matrix 2MB/expert, SwiGLU three matrices 6MB/expert】: taking the slope between the C-column points `bs=32→128`
⇒ gemm `8.2us/slot`, swiglu `25.2us/slot` ⇒ **3.07x marginal cost per slot** ⇒ the three matrices hide **no extra serialisation**,
they honestly pay 3x. 🚨 It is neither a whole-frame ratio nor "real shape vs demo shape": the whole-frame `swiglu/gemm` moves with bs —
rotation caliber, C column bs=1/4/32/128 = **2.25/2.79/2.89/3.02** (`ebf87c3b`); solo caliber =
**2.241/2.310/2.690/3.018** (`3840ec53`) ⇒ **≈3x holds only at bs=128; the low-bs truth is 2.2~2.4**.
For the real shape (SmolLM `D=576/N=1536`, w1/w3/w2 = **2.53MB**/expert) the byte wall is **63us/expert**, not the 147us of the 6MB case.

### 3.2 The GPU side pays by weight bytes

Taking the hot experts' weights on the GPU half from fp32 to bf16: end-to-end **−15.4% (gemm) / −33.9% (swiglu)** (`ee301770`),
but **calibration values must be measured**: the effective bandwidth of a bare `mm` and the end-to-end realised gain are not the
same thing (the `dtrate` probe).

### 3.3 Dense tax and decode

A zero-traffic expert still pays **35.8us (fp32) / 20.6us (bf16)** per frame (`a1dabef9` conclusion ④); traffic-sparse batching
compresses it to **3.6 / 1.8us** (÷9.9 / ÷11.4), so the best `bs=1` cell is **0.184ms** < the same build on pure CPU **0.279ms**
= **−34.1%** ⇒ the intuition "under decode you must fall back to pure CPU" is void (`64436f8a`).
🚨 Batching lands only at `bs=1`: at `bs=128` the traffic path loses **+238.3%** (`25746271`).

---

## 4. The unit-price criterion (this paper's transferable output)

**Definition** `u_e` = the KL measured when only expert *e* is placed on the CPU leg (`gc = E−1`); `kappa_e = u_e / hit_rate_e`.
**Criterion** resident set = the one that minimises `Σ_{e∈CPU} hit_e × u_e` under the capacity budget; swap threshold `thr_rel = 5%`.
**Cost** 0 free parameters, one offline table of **8 floats**.

| Check | Result | Archive |
|---|---|---|
| Within-segment rank stability | Spearman **+1.00 in all three segments** (n=9/10/6) | `2e7f1cb2` |
| Cross-segment rank stability | 23 cells **0.92~1.11** | `2e7f1cb2` |
| Additivity (verifiable at the endpoints) | predicted / measured = **0.902 / 1.070 / 0.920** | `2e7f1cb2` |
| **Is the selected resident set the measured optimum** | **3/3 segments: yes** | `3ce6836d` |
| Versus the old criteria (coverage / heat) | coverage-optimal is **1.60x more expensive** on the drift segment | `3ce6836d` |
| One checkable prediction | predicted drop **8.0%** ↔ measured **7.99%** | `27f8c744` |
| Cross-domain | `kappa_e` cross-segment Spearman only **+0.16**, `u_e` **+0.62/+0.88/+0.81** ⇒ look up the `u_e` table, **one copy per domain** | `2e7f1cb2` |

⇒ "who is hotter" is a proxy; "who makes int8 pay more KL" is the bill. And that ledger is 8 floats on this machine.

> 🚨**The header-load caliber of `2e7f1cb2`** (the only archive in this repository committed despite a header of
> `loadavg 0.82/2.63/2.68` above the `<0.8` threshold): ① everything the text takes from it is **ranks and ratios**
> (Spearman +1.00 in all three segments, cross-segment 0.92~1.11, additivity 0.902/1.070/0.920, `kappa_e` cross-segment +0.16) —
> **not one absolute µs**; ② `u_e` itself is a **KL = deterministic quantity**: recomputing on the same input is bitwise identical, and
> core contention changes the **elapsed time**, not the **reading**; ③ line 5 of the archive carries a downgrade notice **printed by the
> rig itself** ("with `loadavg > 1.5` absolute values are inflated overall — a desktop is stealing cores — read only relative values
> within one run"); ④ the original scope of the `<0.8` threshold was written for "**P0-② style timed rounds**".
> ⇒ it **remains a valid verdict table**, but **only ranks/ratios may be quoted**. This boundary is also in the freshness-credential
> column of the corresponding row of `writeups/README.md` §4.

---

## 5. Three cuts at the migration price tag (each cut overturned the previous caliber)

| Cut | How it was measured | Result | Landing |
|---|---|---|---|
| 1 (`6eabc1de`) | wall-clock difference of the flip window ÷ frame length | sync **19.30 frames/flip × 3.911ms = 75.5ms**; the flip itself p50 15855µs ≈ 4.1 frames ⇒ **only 1/5** | booked unsettled |
| 2 (`0e305a89`) | window split into C/H/**M**/P and attributed | sync **M/T = 81%** ⇒ the money is the **steady-state placement penalty while flipped**, not the switch itself | **I2**: table 3's price-tag caliber void; `thr_rel` must not be recalibrated on it |
| 3 (`22ff0967`) | add a control that is **flipped for whole segments** ⇒ `once=(tag+tagf)/2` cancels the steady term **by construction** | sync **once = 4.89 frames = 19.9ms**; async_staged once 1.09 frames ⇒ **ratio_once = 0.22**; the steady term **4.645 ms/frame** reported separately | **J2** (prediction hit) |

Self-consistency: after normalising both rulers to **ms/frame**, the self-consistency gap for sync is **9%** (<50%)
⇒ "the cancellation holds", so `19.9ms` may be used.
⇒ **what `thr_rel` needs is "how many frames does one switch cost" (a one-off); the steady term has the opposite sign in a real
decision and cannot be the threshold.**

---

## 6. Negative result: dynamic placement was not evaluated once in this project's four attempts, plus a quantitative proof that the ruler is unfit

### 6.1 Observation: the timed rounds never fired

`d9d54f5b` (n=25) reads the **decision counters** (immune to bimodality): **zero decisions per arm in the timed rounds**
(A1/A2/A3/C1/H1/H2 all 0 times / 0ms). ⇒ the "saves 20%~27%" seen in earlier dyn tables happened **outside the timed rounds**
(at start-up), and under the timed caliber it can only be read as **【one-shot placement at start-up】**.

### 6.2 The rig: a corpus whose membership really drifts + four trees, four unsettled bookings

| Cut | The single changed variable | Landing | The only red item |
|---|---|---|---|
| r1 `c299b2b9` | Build the rig (7 arms × 3 blocks × 24 frames; the oracle sets of different blocks are pairwise different; all gates green) | **K0** | empty control `底_pair 6.91% > TARGET=5%`; oracle star-passing blocks = [] |
| r3 `96ecadf1` | Change only the ruler (★ → ★2: both clusters must individually pass and agree in sign) | **M0, 2nd time** | oracle ★2 blocks = []; `底_pair 4.18%` passed but **`底_序 16.00%`** ⇒ threshold 32.0% |
| r4 `48499173` | Change only the estimator (per-block p50 → **block sum**; criterion values and rig code untouched) | **M0″, 3rd time** | `底_pair 8.41% > 5%` (n=48 ≥ N_MIN=45) ⇒ threshold 16.8% |
| r5 `1f69a7cb` | Change only the replication unit (**pass → process**, NP=5) and randomise arm build order | **M0‴, 4th time** | cross-process median of `底_pair` **5.05%** > 5% (short by **0.05pp**) — a‴/b‴/d‴/e‴ green simultaneously for the first time |

(r2 `3efbe364` is the voided-rig round: gate ⑥ 1027/1029 turned out to be a **false red** from an un-updated predicate — measured hard
reds 0 + transient 2 ⇒ lands at M0, **not ratified**.)

r4 worked exactly where it was supposed to: bimodality **11/21 → 0/21**, the order term `底_序` **16.00% → 6.94%**, the oracle anchor
green for the first time. The single red item was still the same one ⇒ one must answer "why neither more passes nor a better
estimator fixes it". r5 changed the **unit of replication**: the median of `底_pair` went **8.41% → 5.05%**
(per-process `[5.81/5.03/7.42/5.05/4.73]%`) = halved, but **did not cross TARGET=5%**, by 0.05pp ⇒ fourth unsettled booking.

### 6.3 Qualification: arm-identity fixed effect ≈8% → **per-instance one-off offset** (this paper's technical contribution)

| Evidence (labels 甲/乙/丙/丁 = i–iv are the ids used in the Chinese archives) | Value | What it rules out |
|---|---|---|
| **甲** (i) A2/A1 over three blocks **−8.4/−7.5/−4.1%**, forward and reverse rounds of each block **agree in sign** (forward −8.9/−8.2/−7.5, reverse −7.9/−6.7/−0.6) | sign agreement | order effects (forward and reverse contribute equally, so agreement cannot be an order effect) |
| **乙** (ii) A3/A1 negative in all three blocks **−2.6/−0.7/−1.6** (🚨 per-block sign agreement holds only for A2) | two instances agree | a single machine's fluke |
| **丙** (iii) net-gain column A2 **+23.0** / A3 **+4.5 ms/pass** | an arm that never migrates "gains for free" | real acceleration (physically impossible) |
| **丁** (iv) after dividing out each arm's **own** cross-block geometric mean, the order curve leaves only **±2~3.6%** | the 8% is not on the order axis | order, number of passes, estimator |

⇒ the offset is drawn **at the moment the arm is built** (allocator / page tables / graph pool / where the resident set lands —
uncontrollable items): it **does not eat passes** (n=25→48 leaves it unmoved), **does not eat the estimator** (p50 and block sum agree
in magnitude), and **same-process pairing cannot cancel it** (both sides of the pair are the same instance).
⇒ **r4's judgement at the time: arm-vs-arm differences below ≈8% are unresolvable here**, and TARGET=5% sits just below that line
【🚨 the current caliber is not 8% — after r5 changed the rig it narrowed to **≲5~7%**; everywhere in this paper §7's
"per-instance one-off offset" row is authoritative】. Disposition, following the pre-committed hint: **more passes are blocked**; next
cut changes the rig = **process-level replication + randomised arm build order**.

**r5 (`1f69a7cb`) reclassified the attribution.** After changing the rig and randomising build order: the median of `底_pair` fell to
5.05% (still red by 0.05pp), but the competing hypothesis "the offset follows the arm's build position" was swallowed by its own
range band `[−4.32~3.89]%` (before A1 `+0.02%` vs after A1 `−1.13%`), and the same-configuration offset calibration came out at
**`σ_inst≈1.6%`** ⇒ the offset is **not bound to arm identity** but is **per-instance and one-off** (the k-th block of memory handed
out): shuffling and switching processes only **redistribute** offsets between instances without changing their distribution.
⇒ the quantitative conclusion "**5% is not measurable on this box**" is **unchanged**, but its cause is downgraded from "attributable
to identity" to "**not attributable**" — which is a harder negative result.

### 6.4 The upper-bound anchor: how much is edible ≠ you can eat it

`O1` (zero reaction latency, pays the same flip) `GAIN = 0.6866` ⇒ the placement gain on this drift corpus has an
**upper bound of 31.3%**; edible per frame **blk1 3.168 / blk2 3.022 ms/frame**, blocks are 24 frames ⇒ **sync pays back in
6.3/6.6 frames, async in 1.4 frames** ⇒ **"it is too expensive so you should not act" does not hold (mathematically reachable)**.
`H1 T = 0.64` (across rounds 0.86→0.84→0.64→**0.578**, the last being r5's cross-process median); r5's `O1 GAIN = 0.7285`
⇒ upper bound **27.2%**. 🚨 **all of these come from the ruler that failed its pre-check** ⇒ corroboration only, never a claim.

### 6.5 Why the control "turn the drift up and judge again" does **not** have to be run (zero measurement, a constructive answer)

The control outside reviewers ask for most often is: "what if the ruler fails only because the stimulus is too small? Turn the
synthetic drift up until the effect beats the noise floor, then judge again." On this rig that sentence **structurally hits nothing**,
for three reasons:

1. **The pre-check ruler is independent of stimulus amplitude**: `底_pair` measures the **empty control arm** — configuration
   literally identical, **never firing, never deciding**. However large the stimulus, it does not change "how far apart two same-build
   instances read" ⇒ turning the drift up **cannot buy a green pre-check** (item c is red by construction).
2. **"Can it see a large effect" already has an answer: it can.** Positive control C1 (member set deliberately wrong, known to be
   slow) passed the star in **3/3 blocks** in both r2 and r3 (`3efbe364`, `96ecadf1`) ⇒ the ruler is not blind, it simply cannot read `≲5~7%`.
3. **And the net effect of this question falls exactly in the unreadable band**: the drift amplitude here is already the ceiling
   (pairwise-different oracle sets per block + zero-latency oracle, `O1 GAIN 0.7285` ⇒ upper bound 27.2%), yet after subtracting the
   flip price tag (sync `once 19.9ms` ≈ 4.89 frames, `22ff0967`) the **net** difference between dyn and static drops into the noise
   band. Pushing it out of the band would require a stimulus **unreachable by real text** — and "what is it worth on real text" is a
   different matter, registered in §7 as an **unmeasured item**, on which **this paper gives no sign, positive or negative**.

⇒ Turning the amplitude up would indeed manufacture a scenario where "dynamic placement pays", but **the existence of that scenario
is not this paper's question**. Registered as *will not do*.

---

## 7. Limitations

- **Corpus**: out-of-domain readme patchwork, **no held-out ppl**; quality is reported only in relative terms (`CORPUS=` swaps it).
  ✅ **The relative conclusions were re-measured on a second corpus** ⇒ separate subsection §7.1.
- **Referee**: a same-architecture fp32 dense MoE, not the original model; and the referee can be **born-dirty** (one stage-2 run with
  `|P|absmax=5.85e+19` pushed the reference KL to `2.564e+11` while top-1 did not move) ⇒ the quality columns of some segments are void
  as a whole; root cause not settled (probe attached).
- **Sample resolution**: the top-1 denominator is that cell's bs; the bs=512 setting was added (`13f8740c`, resolution **0.195pp**),
  bs=1 is still a single-token coin.
- **Placement rule v2**: the default. Mean overpricing **2.8%** at bs=32; at bs=128 **4/4 optimal, 0.0%**;
  🚨 the two bs=1 cells are **unfixed** (`A fp32 +7.0%`, `C fp32 +8.1%`).
- **Noise floor / per-instance offset**: A 2~3%, C 8~18%; between same-configuration instances there is also `σ_inst≈1.6%` and a
  worst-cell range of 4.73~7.42% ⇒ **no "who is faster" statement of ≲5~7% may be read on this machine**; buying replication layers was
  killed by calculation (deciding needs `k≈47` = 329 instances = physically impossible).
- **Dynamic placement**: see §6. Asynchrony can compress the one-off price tag to **0.22x** (`22ff0967`), but **the default remains
  sync** (changing a default requires the author's approval).
- **Unmeasured item (not a negative result)**: whether natural drift on real text can feed a dynamic placement loop — **not measured**.
  One zero-cost pre-scan only (reads gate histograms, builds no arm, times nothing ⇒ not part of the record, earns no realisation),
  on which **no work order was issued** ⇒ this paper gives **no sign, positive or negative**, on what dynamic placement is worth on
  real text; the committed upper-bound anchor belongs to **synthetic drift** only and its round failed the pre-check (corroboration only).
- **Scale and parallelism**: every measurement is single-node single-GPU with `E=8`/top-2 and a 135M shape; no multi-GPU, no NCCL, no
  expert-parallel all-to-all, no 64+ experts, no downstream tasks ⇒ this is **single-node heterogeneity metrology and a placement
  criterion**; for larger scale **not one of these numbers applies**. Nor do we compare numbers with any published MoE inference
  system (not run ⇒ not in a table).
- **Reproduction on another machine**: the only way to overturn "≲5~7% is unresolvable here" is **a different physical instance**; none
  to hand ⇒ **not done** (all four layers of replication buyable on one machine — passes / estimator / arm build order / processes —
  were spent in r1~r5; the candidate-machine probe is committed) ⇒ "this ruler is unfit" is a statement about **this machine plus this
  rig**, **not a universal negation**.
- **P2, frozen**: int4, biased packing, VNNI, register tiling, large-bs tuning — all in the disproof list or gated behind P0.
- **Data that is not in `results/`**: the LADDER 14-cut cumulative ledger (`A/d0 1.758→0.300ms` / `C/d0 6.109→0.756ms`) and the two grey
  diamonds in `fig1` (`+T8` disproof / `+T12b` default-off) come from **the LADDER table in `OPTLOG §1`** ⇒ they **have no md5** (a
  cumulative ledger is not a single verdict table), but `writeups/make_ledger.py`'s `verify()` **reconciles every cell against the
  `results/` archives** (the line "all 11 LADDER rows agree cell by cell with the archives" in its output is that check). ⇒ the **only**
  exception to "§4 index lists the source of every number in the text", registered explicitly under the table in `writeups/README.md` §4.

### 7.1 Cross-corpus robustness (reorganising data already held — **not a new measurement**)

The quality section originally stood on a single corpus (this repository's readme patchwork) ⇒ we ran one control that changed
**only the corpus axis** (`0e0694aa`: second corpus = the dialogue of the English play *Coriolanus*, 7314 tokens; same as §4c shape,
same gates, `BS=512`, 20 cells green):

- **What did not change**: the "wrong placement ÷ right placement" tax **5.9x / 20.9x / 309x** (gc=3/5/7) **reproduced** on the second
  corpus; the bf16 cut stayed negligible (top-1 cell-by-cell 100.00, same as fp32; absolute KL `≤6.7e-06`); the engine's own 10 cells
  `|ΔNLL| ≤ 3.95e-04`, ppl ratio ∈ [0.9996, 1.0001] ⇒ still inside noise.
- **What did change**: the hit histogram reordered (hottest/coldest `433x → 201x`) ⇒ **the gc=3 resident member set went
  `{0,2,1} → {0,5,2}`**, and only for gc≥5 do the two corpora agree ⇒ at the small-gc end, "static placement done right" is
  **a function of the corpus**. This is the same axis as §4's "`kappa_e` cross-segment Spearman only +0.16 ⇒ one price table per domain".
- 🚨**Caliber**: this corpus was very likely seen during pre-training ⇒ this subsection claims only **robustness to a corpus change in
  relative terms**, and is **not written up as a strict held-out**; neither corpus gives an absolute scale.

---

## 8. Reproduction

```bash
cd labml_nn/transformers/glu_variants
PY=/home/adomlans/miniconda3/envs/dl_lab/bin/python   # 🚨 this interpreter, not base (its 3.14 profiler is broken)
ROUNDS=3 ./run_all.sh swapcost      # the three-cuts rig
ROUNDS=25 ./run_all.sh dynpair      # ruler calibration (empty control + both orders)
(date; ROUNDS=48 NP=5 ./run_all.sh drifttrackN) 2>&1 | tee /tmp/p1i_drifttrack_r6.txt   # ≈40min
# commit threshold: header loadavg<0.8; verdict tables are run by the author and committed in full, model runs are labelled "not part of the record"
```

**The discipline of this paper is itself part of the deliverable**: criteria written down before the run, landings evaluated by code,
a failed pre-check blocking the question, the primary ruler designated before the run (no picking a favourite between two rulers),
and not one old table touched. The four unsettled bookings are the price list of that discipline.
