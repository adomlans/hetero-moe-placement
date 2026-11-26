# Wrap-up overview — running a SwiGLU MoE on two legs, CPU+GPU (on this machine every single item pays for the same model)

> **This file is the English translation of `README.md`** (the same `writeups/` index). It is a translation, not a second
> text: every number, every `md5`-style evidence pointer, every archive filename and every table cell comes from the
> Chinese original verbatim; where the two disagree, **the Chinese text wins**. Kept in Chinese inside backticks are
> literal printed identifiers only — the OPTLOG section ids (`§5.2i附` …), the archive appendix markers
> (`P0-④a附`, `P1-②附`), the printed column marker `配对可用` (pairing-usable) and the archive series labels
> `甲/乙/丙/丁` = i–iv.

> **Writing red lines shared by the four documents** (pinned in every header, also written into OPTLOG §4/§5.1.8):
> ① **First-order and second-order quantities in two columns, always together**: relRMSE/L2rel and top-1 agreement/KL/TV
> must appear side by side; "error <1%" is not allowed to stand as a sentence on its own (methodology 22: where L2
> differs by 11.6x, KL differs by 303x, and in the gc=0→1 cell the two columns point in opposite directions).
> ② **Quality is reported only as a relative quantity, and the reference row must be attached**: swapping the real
> layer-8 MLP for "an 8-choose-2 fp32 dense MoE" is itself worth KL 1.897e-01 (bs=32) / 8.854e-02 (bs=128), top-1
> 84.38%/86.72% ⇒ **668x / 871x** dearer than the harshest engine setting. No KL in this project may be used as
> absolute ppl.
> ③ **Memory must be labelled either [peak MB] or [steady-state MB]**: at gc=5 both dtypes peak at the same 186.1MB,
> while steady state is 169.3(fp32) vs 109.8(bf16) (methodology 24). The two columns may not cross-reference each other.
> ④ **The three-way exchange uses only the two P0-④a archives** (same cell, same weights, same judge; gate ⓪ has
> already reconciled them); **every number points back to a verdict table the user ran themselves** (`results/` carries
> md5/date/header-line loadavg).
> ⑤ **In paired segments only rows whose `配对可用` (pairing-usable) is ✓ are trusted** (both legs land on L3 within
> ±10% and the faster round is 7/7 or 0/7); Δ on a ✗ row is the noise floor, not an effect size.
> ⑥ **Cross-process bimodal cells get an interval, never a single point**: A bs=128 `gc=3 bf16` = **1.35 / 1.74ms
> bimodal** (within-round σ=1%); it enters no denominator in the text (methodology 26).

---

## 1. The claim (one sentence) and its three legs

> Running a MoE on CPU+GPU heterogeneity: hot experts stay resident on the GPU, cold experts run on the CPU in
> int8/AVX2, **the two legs overlap**, and per-frame latency ≈ `max(CPU leg, GPU leg) + fixed serial host overhead`.
> Every measurable item on this machine pays for exactly this model.

| # | Who proves this leg | The single hardest number |
|---|---|---|
| ① **the model itself holds** | 乙-1 / P0-② (once `delay=3ms` pins the CPU leg down, every knife aimed at the CPU leg goes flat) | **the price of overlap is positive in 16/16 cells**: +8.5%~+67.6% (the gemm A setting rises monotonically with bs) |
| ② **the cost structure holds (two walls)** | T8/T14/乙-2 + P0-②③④a | CPU byte wall 49us/expert(gemm 2MB)~63us(swiglu 2.53MB)@42GB/s, compute wall 7us/slot@288GMAC/s, **crossing point ≈5 slot/expert**; GPU skinny calibration **250GB/s** |
| ③ **the gain decomposes and is attributable** | LADDER 14 knives + 乙-1/乙-2/乙-2b/乙-2c + P0-①②③④a | A/d0 **1.758 → 0.300ms (−82.9%)**, C/d0 **6.109 → 0.756ms (−87.6%)**; the −60~−76% under decode all comes from host-side serialization |

**The multiples that may be quoted externally** (P0-②; that table with A placed right, same machine, toy weights is the
only legal source): swiglu bs=128 **vs pure-CPU 5.47x / vs pure-GPU 1.51x**, bs=32 2.61x / 1.40x; gemm bs=128
4.18x / 1.25x.
🚨 **A multiple must always be reported together with its bs**: at bs=4 it is only 1.11~1.63x, and at bs=1 pure CPU
wins (heterogeneity does not pay off under decode).
🚨 The sentence external models like to write, "3.5x speed-up when CPU latency is 3ms", measures **1.0x**.

**The three hardest cells on real weights + three-way exchange** (`P0-④a`/`P0-④a附`; memory is reported as
**steady-state**, baseline = gc=0 **of the same dtype**):

| cell | baseline gc=0 | switched to | peak/steady MB | p50 | KL(P‖Q) | what it buys |
|---|---|---|---|---|---|---|
| bs=1 placed right, bf16 | 0.273ms / 0.1MB / 4.928e-03 | gc=3 | 111.4 / **65.9** | **0.215ms** | **1.353e-05** | spend 66MB ⇒ 21.2% faster **and** 364x better quality |
| bs=32 placed right, bf16 | 1.378ms / 0.3MB / 2.841e-04 | gc=5 | 186.1 / **110.0** | **0.390ms** | **1.180e-05** | spend 110MB ⇒ 71.7% faster **and** 24x better quality |
| bs=128 placed right, bf16 | 3.536→0.510 / 3.427→0.493 | gc=7 | 262.1 / **155.3** | **0.510 / 0.493ms** | **1.532e-06** | spend 155MB ⇒ **85.6% faster (same value in both rounds)** and 66x better quality |

---

## 2. What is being measured (the common conditions behind both figures and every table)

| item | value | source |
|---|---|---|
| **real source weights** | SmolLM-135M: D=576, N=1536, 30 layers, vocab 49152 | all of P0-①②③④a |
| **expert set** | the MLPs of layers [0,4,8,12,16,20,24,28] used as 8 experts, **top_k=2**, routing = embed row | same as above |
| **real routing hits** | hit% = [42.3, 13.2, 17.7, 6.8, 1.4, 12.6, 6.0, 0.1], **hottest/coldest = 433x**; A order [0,2,1,5,3,6,4,7] | `P0-④a` header line |
| **activation** | x = layer-8 MLP input (512, 576), corpus = this repository's readme pastiche (**out of domain**, only Δ is comparable) 7244 token | P0-③④a |
| **the 14 LADDER knives** | demo shapes bs=32/hidden=1024/D=1024/N=2048/E=8, random std=1 weights | OPTLOG §0/§1 |
| **the two legs** | `L2_zero-CUDA+pinned`: zero CUDA calls on the CPU side + pinned buffers, overlapped with the worker | same as above |
| **configuration A / C** | A = the genuinely hottest gc go to the GPU (placed right, hitting the byte wall); C = the genuinely coldest gc go to the GPU (placed wrong, hitting the compute wall) | same as above |
| **machine** | i5-10400 6C/12T, L3=12MB, DDR4 effective **42GB/s**; RTX3060 skinny GEMM measured **250GB/s**; no AVX-VNNI (uses `maddubs`) | T14/乙-2/T13 |

---

## 3. The two ledger figures (`writeups/`)

| figure | what it draws | data source |
|---|---|---|
| `fig1_ladder.png` | the cumulative bill of the first 14 knives (A/d0 and C/d0 as two polylines, log-y); grey diamonds draw the two steps outside the main line (+T8 refutation / +T12b off by default) | the LADDER table in OPTLOG §1 |
| `fig2_exchange.png` | the three-way exchange: x=**steady-state MB**, y=p50(log), circle colour=KL, hollow=fp32/filled=bf16, thick ring=Pareto frontier, green dashed line = this table's claim, purple box = cross-process bimodality | the two `P0-④a` archives |

```bash
/home/adomlans/miniconda3/envs/dl_lab/bin/python writeups/make_ledger.py
# ✅ reconciliation passed: all 36 exchange cells + the frontier set + both bimodal peaks + the 11 LADDER rows match the archives cell by cell
```

🚨 The script **reconciles itself**: `verify()` parses the `results/` archives and the OPTLOG LADDER table directly,
and **on any mismatch it `SystemExit`s and refuses to draw the figure**. Not one number in the figures is a second
hand-entered copy.

---

## 4. Verdict-table index (the source of every number in the text; all of them run by the user)

| archive file (`results/`) | md5 first 8 | freshness evidence | what question it answers | the single hardest number |
|---|---|---|---|---|
| `2026-11-21_P0-④a_swap_3way.txt` | `a1dabef9` | date 10:41:44, header line 4000.035MHz loadavg 0.22/0.23 | memory↔latency↔quality exchange in the same cell (72 rows) + the dtype paired segment | the price of lower precision rises with gc to **−48~−53%** (only cashed in where bytes are the bound) |
| `2026-11-21_P0-④a附_bs128_crossproc.txt` | `a6b759d7` | date 11:35:42, header line 4000.971MHz loadavg **0.17/0.08** | the second-process evidence for the bs=128 absolute values (127 rows) | **−85.6%, same value in both rounds**; brings out the cross-process bimodality 1.35/1.74ms |
| `2026-11-21_P0-③_metrics_top1-kl.txt` | `fba72657` | date 09:28:34, loadavg 0.07/0.26, gates ①②③④⑤ all green | moving the quality claim from relRMSE to the token level (60 rows) | the int8 bill is **274~364x** dearer than bf16; bf16 is free at the token level |
| `2026-11-21_P0-②_base_4paths_user7r.txt` | `5bb8a840` | date 08:38:54 (clean window, md5 ≠ my own 11-20 run) | the first comparison **against somebody else**: heterogeneous / non-overlapping / pure-GPU / pure-CPU, four paths + gc sweep | overlap price positive 16/16; winner map 6/6/4; rule 8/8 dearer by +16~95% |
| `2026-11-21_P0-①_real_swiglu_bs1-128.txt` | `6a34d45c` | header line 4149.147MHz loadavg **0.81** (timing not admitted, gates are) | the full chain with real weights and real shapes | real routing is naturally skewed **433x**; D=576 exposed an out-of-bounds read in the tail block |
| `2026-11-21_red_after_P0-①.txt` | `6a37a7e1` | same as above | all red lines green after P0-① + a new poisoned-neighbour tooth check | a red line must be able to prove it still bites |
| `2026-11-21_乙-2c_dtype_gc5-bs1.txt` | `ee301770` | header line 4236.320MHz loadavg 0.22/0.27 | the price of dropping precision on the GPU half (gemm/swiglu × A/C × bs) | end-to-end **−15.4%(gemm)/−33.9%(swiglu)**, not 2x |
| `2026-11-20_乙-2_swiglu_realshape.txt` | `ebf87c3b` | header line 4175.608MHz loadavg 0.25/0.90 | the three SwiGLU matrices (demo shape 6MB/expert): the two walls re-calibrated + four precision bills | **per slot** marginal cost ratio **3.07x** (column C `bs=32→128` slope: gemm 8.2 / swiglu 25.2 us/slot ⇒ no hidden serialization in the three matrices); 🚨the whole-frame ratio moves with bs **2.25→3.02**, the low-bs true value is 2.2~2.4 (solo gauge `3840ec53`); re-quantising h is only 1/5 |
| `2026-11-20_乙-2b_halves_bs1.txt` / `…乙-2b附_halves_bs32.txt` | `0bdd9f58` / `3725cc18` | attribution table (diagnostic gauge) | that +0.3ms which does not depend on slots is attributed to the fp32 bytes of the GPU half | 415us/126MB = **303GB/s** is already the card's read ceiling |
| `2026-11-19_T14_ab.txt` / `…_T14_red.txt` | `946b290a` / `8fb8f1c3` | header line 4091.091MHz loadavg 0.27/0.47, 11 paired rounds | the straggler tax: `static,32` → `guided` | one core-grabber attached: 839→**569us (−32%)**; end-to-end A −6.8%/C −12.4% |
| `2026-11-20_dig3_pointer_probe.txt` | `ecc8d623` | probe | inside an OpenMP parallel region `static thread_local` gets an empty copy | silently the wrong scale (a real bug, fixed) |
| `2026-11-22_P1-②c_err单价_bs512.txt` | `2e7f1cb2` | date 11:20:31, header line 4082.779MHz **🚨loadavg 0.82/2.63/2.68 = above this project's `<0.8` load gate** (the gate's own scope statement reads only "**P0-② style timing rounds**"; this file is a **deterministic quantity** of KL/ranking ⇒ **admitted in downgraded form** per what line 5 of the apparatus prints itself: "with loadavg>1.5 absolute values are uniformly inflated, only relative values inside one and the same run are meaningful" ⇒ 🚨**only rankings and ratios may be quoted, not one single absolute µs**), 66 cells + 27 archive anchors, 196s, average 2.07 shots/cell | the int8 **unit price** `u_e` of every expert + three models with **0 free parameters** fighting each other | the unit-price pole's within-segment Spearman **+1.00 in all three segments**, coverage traffic is left with **+0.60** on drift |
| `2026-11-22_P1-②d_dyn_gc3_bs512_priced.txt` | `3ce6836d` | date 12:18:12, header line 4105.837MHz loadavg 0.07/0.44/0.99, 33 rows, 83s, gate ⑨ 33/33 quotable | placement criterion switched from "hot" to "unit price": member set / coverage / KL / migration price tag **in the same cell** | the price-optimal roster is **3/3 segments = that segment's smallest measured KL**; coverage-optimal on drift is **1.60x dearer** |
| `2026-11-22_P1-②d附_dyn_gc3_bs512_r3_coolfix.txt` | `27f8c744` | date 13:04:42, header line 3999.273MHz loadavg **0.08/0.02/0.20**, re-run with a byte-identical command to the row above, 86s, gate ⓪ mismatch **0 cells** max `7.41e-05` | is the cool-fix's blast radius just the prediction row + can the price pole report a "value" | prediction down **8.0%** ↔ measured down **7.99%**; 🚨same round `\|P\|absmax=5.85e+19` ⇒ the phase-2 quality column is void |
| `2026-11-21_P0-④c_metrics_bs512.txt` | `13f8740c` | date 13:32:54, header line 4058.989MHz loadavg **0.19/0.10**, 88 rows | the catch-up run for the sample-resolution debt: the quality table re-measured at BS=512 | top-1 resolution **0.78pp → 0.195pp**; reference row KL `8.306e-02`/top-1 `90.43%` = **1585x** the harshest engine setting `5.240e-05` (the reference row of red line ②) |
| `2026-11-21_乙-2d_swap_bs1_traffic.txt` | `64436f8a` | date 14:24:53, header line 4083.863MHz loadavg **0.10/0.35**, 220 rows | traffic-sparse batching: can the dense tax really be eaten away | zero-traffic experts per frame **35.8→3.6us(÷9.9) / 20.6→1.8us(÷11.4)**; bs=1 best **0.184ms** < the same slice pure CPU `0.279ms` = **−34.1%** ⇒ "decode can only be carried by pure CPU" is void |
| `2026-11-21_乙-2d边界+P1v2_swap_bs32_rulev2.txt` | `f17fc0d0` | date 15:27:02, header line 4062.391MHz loadavg **0.04/0.32**, 271 rows | the 乙-2d boundary (bs=32) + the first setting of placement rule v2 | rule v2 is on average **2.8%** dearer over four cells at bs=32 (v1 61.3%), worst **+8.4%** (v1 **+206.9%**); on the traffic-worse side all 9/9 ✓ rows lose |
| `2026-11-21_P1v2_swap_bs128_rulev2.txt` | `25746271` | date 19:38:57, header line 4143.665MHz loadavg **0.16/0.11**, 281 rows | the second setting of placement rule v2 (bs=128, the only one with a cross-process axis) | v2 **hits the global optimum in 4/4 cells, average excess 0.0%** (v1 114.6%, worst **+303.7%**); 🚨in the same round traffic loses badly at bs=128, **+238.3%** ⇒ batching only lands at bs=1 |
| `2026-11-22_P1-②_dyn_gc3_bs512.txt` | `bd8eae3b` | date 09:18:38, header line 4075.886MHz loadavg **0.10/0.11**, 139 rows | 11 warm-start strategies in the same cell (dyn round one) | shuffle with the right members KL/A **3.87**, static_C **5.11**; hyst_1x with one migration 4.0 frames/**14.23ms** ⇒ **0.59** |
| `2026-11-22_P1-②b_dyn_gc3_bs512_startshuffle.txt` | `8f33ba36` | date 09:32:06, header line 4070.805MHz loadavg **0.20/0.48**, 149 rows | re-run after shuffling the start order (autocorrelation check of the position term) | 3.87 → **0.96 = 4.03x** (response **3.2 frames**) ⇒ half of that round's gain was the [start position], not the policy |
| `2026-11-22_P1-②附2_dyn轮转配对_dynpair_r1.txt` | `528be085` | date 19:35:46, header line 4000.029MHz loadavg **0.18/0.71**, 169 rows, n=7 | the first time dyn is given a [same-process rotation pairing] ruler | lands on D, on account: bottom_pair **8.88%**/bottom_seq **16.43%** ⇒ threshold **32.9%** > effect size 27%, bimodality 10/18 |
| `2026-11-22_P1-②附2附_dyn发数标度_dynpair_r2.txt` | `d9d54f5b` | date 20:45:02, header line 4052.066MHz loadavg **0.15/0.10**, 350 rows, n=25 | can adding shots sharpen the ruler down to 5% | lands on E1 [`甲` refuted, the stake paid off]: cumulative bottom n=7/13/19/25 = 10.70/2.22/3.14/**2.75%**, k=1.07, bottom_seq 7.53% ⇒ threshold 15.1%; ⭐**the scoring round made 0 decisions per arm** ⇒ "saves 20%~27%" re-reads as [a one-off placement at start-up] |
| `2026-11-23_P1-②附3_迁移价签重定_swapcost_r1.txt` | `6eabc1de` | date 07:05:35, header line 4055.857MHz loadavg **0.32/0.14**, 84 rows | what a migration price tag actually costs (measured for the first time) | sync `tag_main 19.30 frames/run × frame-length ruler 3.911ms = 75.5ms`; the flip itself p50 `15855µs` (≈4.1 frames) is only **1/5**; ratio 0.738 ⇒ lands on account (the G1 stake loses) |
| `2026-11-23_P1-②附4_价签事后归因_swapcost_r2.txt` | `0e305a89` | date 12:39:33, header line 3999.966MHz loadavg **0.00/0.04**, 111 rows | where those 75.5ms are spent | C 0.30 / H 21.76 / **M 64.80** / P −6.92 / T 79.94 ⇒ **M/T = 81%**, T=40.0ms/run, F=16.1ms/run ⇒ lands on **I2** = table 3's wall-clock price-tag gauge is void, `thr_rel` may not be re-calibrated from this table |
| `2026-11-23_P1-②附5_一次性价签_swapcost_r3.txt` | `22ff0967` | date 16:59:49, outer header line 800.000MHz (idle down-clocking, a single sample; inside the process **4129.68** = no violation) loadavg **0.16/0.06**, 164 rows | prising the price tag back from "steady-state penalty" into "one-off cost" | **sync once_main 4.89 frames = 19.9ms**, async_staged once **1.09 frame** ⇒ **ratio_once = 0.22**; steady-state P `sync 4.645 / async_direct 5.230 / async_staged 4.436 ms/frame`; frame-length ruler 4.057ms ⇒ lands on **J2** |
| `2026-11-23_P1-②附6_真漂移追踪_drifttrack_r1.txt` | `c299b2b9` | date 18:30:13, header line 4000.051MHz loadavg **0.01/0.05**, 388 rows, n=25 | the first of the pre-set decision tree K: can dyn keep up on a drifting corpus | 🚨lands **K0** (red c+e): bottom_pair 6.91%/bottom_seq 7.80% ⇒ threshold 15.6%, `O1 paired [0.9974, 0.5443, 0.6585]`, stars passed = **[]**, the K2 stake loses |
| `2026-11-23_P1-②附6附_磨尺子_drifttrack_r2.txt` | `3efbe364` | date 19:52:08, header line 4029.871MHz loadavg **0.45/0.43**, 654 rows | the ruler-sharpening round (only the ruler changes, not the question) | 🚨**apparatus-void round**: gate ⑥ 1027/1029 turned out to be false red from the predicate of fix ⑰ not being updated (measured hard red 0 + transient 2); all three sharpening items paid off (c′ 0.97%, d′ C1 3/3, e′ O1 ★2=[1,2]); lands M0, **not ratified** |
| `2026-11-23_P1-②附6附_磨尺子_drifttrack_r3.txt` | `96ecadf1` | date 20:26:15, header line 4000.449MHz loadavg **0.44/0.56**, 660 rows | re-run of the same K′ tree after fix ⑰ | all gates green (⑥' hard red 0 + transient 1/1029) ⇒ precondition a **green for the first time**; 🚨lands **M0 for the second time** (red e′: O1 ★2 blocks = [], paired `[1.0291, 0.5748, 0.6811]`); bottom_pair 4.18% meets the bar but bottom_seq **16.00%** ⇒ threshold **32.0%** |
| `2026-11-23_P1-②附6附2_换口径_drifttrack_r4.txt` | `48499173` | date 22:11:10, header line 4000.912MHz loadavg **0.01/0.03**, 686 rows, 305s, ROUNDS=48 ⇒ n=48, main ruler RULER=sum | the only independent variable = the estimator (per-block p50 → block sum): can changing the gauge sharpen the ruler | 🚨lands **M0″ for the third time**: main ruler bottom_pair **8.41%**/bottom_seq **6.94%** ⇒ threshold 16.8% > TARGET 5%; ⭐the main ruler's ★2 detail is empty (bimodality **0/21**), e″ green for the first time (O1 ★2 blocks [1,2]) ⇒ the only red item moves to c″; `O1 GAIN 0.6866` ⇒ ceiling **31.3%** (corroboration only), `H1 T=0.64` |
| `2026-11-24_P1-②附6附3_B1换装置_drifttrack_r5.txt` | `1f69a7cb` | date 13:01:17, 3511 rows, **five-process** loadavg **0.57/0.74/0.79/0.74/0.72 all <0.8** (idle-gate waits 0/110/110/110/120s), NP=5 ROUNDS=48, wall clock ≈32min | the only independent variable = the replication unit (shots → processes) + randomised arm-creation order: can changing the apparatus sharpen the ruler | 🚨lands **M0‴ for the fourth time**: `bottom_pair` per process `[5.81/5.03/7.42/5.05/4.73]%`, median **5.05%** (vs r4 8.41% = **−3.36pp**) **but still > TARGET=5.0%, short by 0.05pp** ⇒ the landing point may not be ratified; ⭐preconditions a‴/b‴/d‴/e‴ **green simultaneously for the first time** ⇒ only c‴ remains red; `O1 GAIN 0.7285` ⇒ ceiling **27.2%**, `T` cross-process median **0.578** (🚨corroboration only) |

---

> 🚨**Data sources this table cannot cover (they are not `results/` archives ⇒ no md5)**: the **LADDER 14-knife
> cumulative bill** (`A/d0 1.758→0.300ms −82.9%` / `C/d0 6.109→0.756ms −87.6%`, cited by leg ③ in §1 and by `fig1`)
> and the two grey diamonds on `fig1` (`+T8` refutation / `+T12b` off by default) = their source is **the LADDER table
> in `OPTLOG §1`** [a cumulative bill ≠ a single verdict table].
> ⇒ it is reconciled **cell by cell back into the `results/` archives** by `make_ledger.py`'s `verify()` (any mismatch
> immediately `SystemExit`s and refuses to draw the figure)
> ⇒ "not in this table" = not a single verdict table, **not** "no evidence"; the header sentence "the source of every
> number in the text" has exactly **this one exception**.

### 4.1 The 10 of the 42 `results/` archives that [did not enter the text] — booked explicitly

> The 8-character short codes appearing in the deliverables are measured in both directions [script gauge: the first 8
> md5 characters of `results/*.txt` against the full text of `writeups/*.md`]: after this round filled in the table
> below = **42 short codes, 0 dangling, all 42 files accounted for** [32 cited in the text + 10 booked in this
> section] ⇒ **zero unbooked**; **before the fill-in it was 32 short codes, with 10 files neither cited in the text
> nor booked**. [Wording finalised: **"not cited in the text" is a deliberate choice, "unbooked" is the leak** — the
> two may not be written interchangeably, and the two-meaning phrase "zero citations" may not be used again.]
> Those 10 files are not lost, they were deliberately not cited — but "deliberately not cited" still has to be written
> down, otherwise a reader cannot tell **screened** from **never looked**.

| archive file | md5 first 8 | what it is | why it did not enter the text |
|---|---|---|---|
| `2026-11-20_P0-②_base_4paths.txt` | `85281ac9` | the first four-path run of P0-② (header line `loadavg=0.86`) | **superseded by the next day's clean window** ⇒ the text cites only `5bb8a840` [`date 08:38:54`, `loadavg=0.28`, fuller ROUNDS]; not one row of the old table is touched, it is simply not cited |
| `2026-11-21_P0-③补_metrics_corpus-input_txt_bs512.txt` | `0e0694aa` | **the change-of-corpus robustness round** (same shape as ④c, only the corpus axis swapped) | 🚨**a genuine missed citation**: the finalised conclusion "the tax `5.9x/20.9x/309x` reappears under another corpus, the bf16 knife stays negligible, but the gc=3 member set goes `{0,2,1}→{0,5,2}`" is both the robustness evidence of the quality section and the case for opening P1-② ⇒ it has been added to the "corpus" row of §7 and promoted into its own sub-section **`paper_short §7.1` (cross-corpus robustness)** |
| `2026-11-21_red_after_P1-②_dyn.txt` | `bb210c99` | a red-line regression file (**not a verdict table**) | it answers "do the gates still bite", and the text has no matching claim to hang it on; the red-line section of OPTLOG §3 does have it |
| `2026-11-22_乙-1b_nt扫描_swiglu_bs1-128.txt` | `5088fec0` | the `n_threads` adaptation verdict round (that 6MB/expert setting, 8 cells) | landing point = **⚪ write no rule** [in 7/8 cells the tax of fixed `nt=6` is ≤0.2%] ⇒ the code does not change ⇒ the text has nothing to cite; this round explicitly registered in `blog3 §7` that "this item is finalised ≠ future work" |
| `2026-11-22_乙-1b_nt扫描_decode_bs1-128_clean.txt` / `..._loadavg2.5.txt` | `0c6f77bd` / `556d5e9b` | the same round at the toy shape (2MB/expert), clean run / dirty run | same as above; the dirty run exists only to demonstrate [how a core-grabbing round is blocked by the gates], it produces no conclusion itself |
| `2026-11-22_乙-1c_配对税探针_pairtax_bs1-128.txt` | `38e09710` | the pairing-tax / cross-measurement water-level probe | **different level**: it compares [the same cell under two measurement gauges], not end-to-end ⇒ a different level from the two ledgers in §1/§3, mixing them would invent a false comparison; the conclusion "the tax is in [running the other setting], not in [being present]" is booked in OPTLOG §5.2b |
| `2026-11-22_乙-1d_镜像税_pairtax_gemm_bs1-128.txt` | `8185eadf` | the mirror tax (run in the reverse direction) | same as above; and its "the published C-column ratio is high by `6%~+29%`" has been **narrowed** to `−0.4%~−4.2%` by the same-gauge re-measurement `3840ec53` ⇒ 🚨the text may not cite the pre-narrowing version |
| `2026-11-22_乙-1e_solo口径比值_solopair_bs1-128.txt` | `3840ec53` | the `swiglu/gemm` ratio on the solo gauge (8 cells) | 🚨**a genuine missed citation**: it is the only evidence for "the low-bs true value is 2.2~2.4", and that `3.07x` sentence in §3.1 never carried it ⇒ **this round added it to the §4 index row and to `paper_short §3.1`** |
| `2026-11-22_乙-1f_双峰归因_bimodal_bs4-128.txt` | `fafa6a26` | the bimodality attribution probe (`ROUNDS=21`) | landing point = **D, on account** [bimodality is unrelated to core-grabbing, pool capacity and shot count] ⇒ no positive number to cite; but it **refuted one of this text's own mechanism hypotheses** (the bet on "resident steady state" lost), booked in OPTLOG §5.2e |

　One sentence on how to read it: **1 superseded + 1 non-verdict table + 6 different-level/landing-on-account + 2 genuine
missed citations already added this round** (`0e0694aa`, `3840ec53`) = all 10 files accounted for; "finalised" and
"entered the text" are two different things, but **if you do not write it down nobody can tell which one it was**.

## 5. The P0 wrap-up checklist (OPTLOG §5.1)

| item | status | deliverable |
|---|---|---|
| P0-① real weights | ✅ | the `real_src=` trio + `./run_all.sh real`; the D=576 out-of-bounds read fixed + a poisoned-neighbour check added to the red lines |
| P0-② external baselines | ✅ | four paths × 16 cells + an 8-cell gc sweep (7 clean-window rounds by the user) |
| P0-③ real metrics | ✅ | `bench_metrics.py` → token-level top-1/top-5/KL/TV/ΔNLL/ppl |
| P0-④a data source | ✅ (+ appendix) | `bench_swap.py` three axes in one cell + gate ⓪ cross-code reconciliation over 60 anchors + the second-process evidence |
| **P0-④b writing** | **✅ this directory** | README + 3 blogs + 2 ledger figures (every number points back to an archive) |

---

## 6. P1 follow-up: the placement criterion swapped from [hot] to [unit price] (three verdict tables in the archive)

**In one sentence**: "who is hotter" is a proxy, "who makes int8 pay more KL" is the bill; and that ledger
**is only 8 floats** on this machine — one offline calibration is enough, online re-pricing **changes no decision**
(⇒ the originally planned shadow probe P1-②e is not opened).
The full development is in `blog2_two_walls_placement.md` §8; only the index and the hardest numbers live here.

### 6.1 price = `u_e`, 0 free parameters, endpoints verifiable (`2e7f1cb2`)

`u_e` = the KL measured when only expert e is put on the CPU leg (`gc=E−1`); `kappa_e = u_e / hit_e` = the expert's own
body error. Three poles that **fit no coefficient at all** are compared on "predicted KL ÷ measured KL" (23 cells
within a segment):

| pole | within-segment Spearman (readme / input.txt / drift) | cross-segment 23-cell ratio |
|---|---|---|
| M0 coverage traffic % (the old criterion) | +0.98 / +0.98 / 🚨**+0.60** | geometric mean 1.03, range **0.15~1.86** (22/23 inside [0.5,2]) |
| **M1 unit price `Σu_e`** | **+1.00 / +1.00 / +1.00** (n=9/10/6) | geometric mean **1.01**, range **0.92~1.11**, **23/23** |
| M2 old-gauge L2rel% | +0.98 / +1.00 / +0.71 | — |

The additive endpoint (harder than a correlation coefficient): **all-CPU measured KL ÷ `Σu_e` = 0.902 / 1.070 / 0.920**.
The reference row (swapping to the MoE itself, red line ② must print it together): KL `8.306e-02 / 8.586e-02 /
1.265e-01` = **1585x / 2263x / 2798x** that segment's worst setting.
The harshest cell: when the CPU leg holds only `{7}` (0.1% of traffic) M0 gives just **0.15x** of the measurement
(underestimating by 6.7x) ⇒ for a counting-type quantity the error shifts systematically with member identity,
**and no amount of coefficient tuning saves it**.

### 6.2 The verdict after swapping the pole (`3ce6836d`, 11 policies × 3 phases = 33 rows in one cell)

| segment | price-optimal roster (measured KL) | coverage-optimal roster (the old "floor" setting) | difference |
|---|---|---|---|
| readme | `{0,2,3}` = 9.370e-06 | `{0,1,2}` = 9.711e-06 | coverage-optimal is **3.6% dearer**, while it covers 73.1% > 66.8% of traffic |
| input.txt | `{0,2,5}` = 6.145e-06 | the same set | tie |
| drift | `{0,2,3}` = 1.301e-05 | `{0,5,6}` = 2.079e-05 | 🚨**1.60 times dearer** |

⇒ **in all three segments the price-optimal roster is exactly the gc=3 roster with the smallest measured KL**; the
"floor" in the old documents should be renamed [coverage-optimal setting], and `KL/floor < 1` is now a
**legitimate reading** (no longer the illusion of "the engine broke through the lower bound").
⇒ start-up / god's-eye `{0,2,1}` is **not** price-optimal in any of the three segments: `price gap` =
**8.6% / 44.1% / 21.7%**.

**Quantitative self-proof** (`27f8c744`): for that `{0,2,5}→{0,2,3}` migration the criterion printed a
**predicted 8.0% drop** before firing, and the measured drop was **7.99%**; `price_cal@input.txt` printed a price gap
of **30.3%**, while the archive arithmetic says **30.43%** was available; the price-optimal cell's `M1÷measured` =
**0.92 / 1.01 / 1.01**.
⇒ `thr_rel=5%` (a single migration measures 13.8~22.8ms = **2.6~4.4 frames**) became a cell-by-cell verifiable
numerical criterion instead of a slogan.

**Cross-domain**: the mean cross-segment Spearman of `kappa_e` is **+0.16** (−0.38/+0.76/+0.10) ⇒ **you cannot look up
one kappa table across corpora**; the `u_e` ranking is far more stable (+0.62/+0.88/+0.81) ⇒
**look up `u_e`, never bare kappa**.
`price_cal ≡ price_hit` (final member set and KL identical bit for bit) ⇒ online re-pricing changes no decision; the
only gap is the change-of-corpus cell (KL/A 0.85 vs oracle 0.59), and the replacement =
**keep one 8-float price table per domain, offline**.
🚨 cross-policy p50 in dyn rounds is **unreadable** (noise floor 15~39%: no rotation between policies + desktop core
grabbing); only the mechanism column and the quality column may be read.

### 6.3 Drift and migration: in **this project's four attempts** dynamic placement was never validly assessed even once (an honest account)

**Conclusion first**: every **quotable** gain this project has about placement comes from **a one-off placement at
start-up** (choosing the resident set by the unit price `u_e`), not from "moving experts along with the traffic drift".
That conclusion was not designed in, it was forced out by two tables:

| archive | the fact | how to read it |
|---|---|---|
| `d9d54f5b` (dynpair r2, n=25) | **0 decisions per arm in the scoring round** (A1/A2/A3/C1/H1/H2 all 0 times/0ms) | that "saves 20%~27%" in the earlier dyn tables happened **outside the scoring round**, and can only be read as [a one-off placement at start-up] |
| `48499173` (drifttrack r4, n=48) | the scoring round did fire (H1 144 times/2208ms, H2 96 times/1489ms), but **the three questions may not be read** | having decisions ≠ being assessed; assessment requires the ruler to pass its preconditions first |

**Four pre-set decision trees, four times all died on [the ruler's preconditions]; the question was never read once**
(every criterion was written before the run and never changed afterwards):

| knife | archive | landing point | the only red item (all other preconditions green) |
|---|---|---|---|
| r1 `§5.2i` "K tree" | `c299b2b9` | **K0** on account | c the null control `bottom_pair 6.91%` > TARGET=5.0% (and e also red: O1 stars passed = []) |
| r3 `§5.2i附` "K′ tree" (sharpening the ruler, only the ruler changes, not the question) | `96ecadf1` | **M0 second time** on account | e′ O1 ★2 blocks = [] (paired `[1.0291, 0.5748, 0.6811]`); bottom_pair 4.18% meets the bar but **bottom_seq 16.00%** ⇒ threshold 32.0% |
| r4 `§5.2i附2` "K″ tree" (changing the gauge: per-block p50 → block sum) | `48499173` | **M0″ third time** on account | c″ `bottom_pair 8.41%` > TARGET=5.0% (n=48 ≥ N_MIN=45) ⇒ pairing threshold **16.8%** |
| r5 `§5.2i附3` "K‴ tree" (changing the apparatus: replication unit shots→processes + randomised arm-creation order) | `1f69a7cb` | **M0‴ fourth time** on account | c‴ cross-process `bottom_pair` **median 5.05%** > TARGET=5.0% (short by **0.05pp**) — the other a‴/b‴/d‴/e‴ green together for the first time |

(r2 `3efbe364` is the **apparatus-void round**: gate ⑥ 1027/1029 was in fact false red from the predicate of fix ⑰ not
being updated, measured hard red 0 + transient 2, lands M0 **not ratified**, kept only as a process file.)

**r4's real output = the first time the cause of death was refuted from "sampling noise" and shown to be [a fixed
arm-identity effect]**. Four pieces of evidence:

| no. | evidence | numbers |
|---|---|---|
| `甲` | the null-control arm A2 is systematically faster than A1, and **the forward and reverse rounds agree in sign in every block** | A2/A1 across three blocks **−8.4 / −7.5 / −4.1%** [forward −8.9/−8.2/−7.5, reverse −7.9/−6.7/−0.6] |
| `乙` | the second null-control arm A3 points the same way | A3/A1 across three blocks **−2.6 / −0.7 / −1.6%** (all three blocks negative; 🚨the per-block forward/reverse sign agreement only holds for A2) |
| `丙` | two arms that **never migrate** "collect for free" in the net-gain column | A2 **+23.0** / A3 **+4.5 ms/shot** (physically impossible: they never moved a single byte) |
| `丁` | once each arm's **own** cross-block geometric mean is divided out of the position curve, only the position term remains | residual **±2~3.6%** ⇒ that 8% is not on the position axis, it is a **fixed offset** drawn when the arm instance was created |

⇒ this 8% **does not eat shots** (r2 up to n=25, r4 up to n=48, neither moves it), **does not eat the estimator** (both
the p50 and the block-sum ruler sit near 8%), and **same-process pairing cannot remove it** (both sides of the pair are
the same arm instance). ⇒ a third round of added shots must still die, so after r4 **the added-shots road was closed**.

**r5 pushes the characterisation one step further: the attribution changed.** r5 (`1f69a7cb`) swapped the replication
unit from "shots" to "processes" (NP=5, 48 shots each) and randomised the arm-creation order: **`bottom_pair` median
8.41% → 5.05% (halved, but not through 5%)**, per process `[5.81/5.03/7.42/5.05/4.73]%`, same-configuration
instance-offset calibration **`σ_inst≈1.6%`**; diagnostic `乙` "the offset follows the arm-creation position" was eaten
by the range `[−4.32~3.89]%` (before A1 `+0.02%` vs after A1 `−1.13%`). ⇒ so it is not **arm identity** but
**one-off per instance** (the piece of memory allocated on the k-th go): **it does not eat shots, does not eat the
estimator, does not eat the arm-creation order, and does not eat the process count** — shuffling and switching
processes only **redistribute** the offsets between instances. (This round's `甲`/`乙`/`丙` diagnostics are the
corroboration column at the end of the combiner, **they do not enter the criterion**.)

**What changing the gauge won and did not win** (r4's only independent variable = the estimator; criterion numbers and
apparatus unchanged to the letter):

| item | r3 (per-block p50) | r4 (block sum, main ruler) | reading |
|---|---|---|---|
| bimodal cells | 11/21 | **0/21** (the main ruler's ★2 detail is empty) | ✅ it worked: migration spikes get spread over the whole block sum |
| bottom_seq (position term) | 16.00% | **6.94%** | ✅ it worked: ÷2.3 |
| bottom_pair (arm-identity term) | 4.18% | **8.41%** | 🚨 does not move: threshold 16.8% still > 2× TARGET |
| the two rulers in the same round, side by side | — | sum 8.41%/6.94% ↔ p50 8.10%/8.13% | changing the gauge compressed the position term, not the arm-identity term |

**The ceiling anchor is corroboration only, it does not enter the claim**: `O1` (a probe that knows the answer with zero
reaction latency, but pays the same flip) `GAIN = 0.6866` ⇒ **ceiling 31.3%** for placement gains on that drift
corpus; `H1 T = 0.64` (3.00 decisions/shot, `T_k = 0.88/0.67/0.63`, reaction latency 16/24 frames, flip 45.99ms/shot,
true net +51.68ms/shot), `H2 T = 0.34`. Cross-round T = 0.86 → 0.84 → 0.64 → **0.578** (r5's cross-process median);
r5's `O1 GAIN = 0.7285` ⇒ ceiling **27.2%** (the same order as 31.3% = the same corroboration reproducing at the
cross-process layer).
🚨 every one of those numbers comes from **the ruler whose preconditions did not pass**; they can only be read as
"machine health + a mechanism clue", never as "adaptation works".

**Disposition** (per hint ④ written before the run in the K″/K‴ trees): on account for the third time ⇒ this machine's
"BLK=24 frames + TARGET=5%" ruler is ruled [**unusable**]; r5, which changed the apparatus, landed on account **a
fourth time** ⇒ that verdict is **restated for the fourth time**. The direction of B1 (change the apparatus) is
supported by the data (−3.36pp) but is **not enough**, and the remaining gap is already outside what "buying a bit
more repetition" can solve: **B5 (folding instances into a repetition layer) was killed by offline arithmetic in `OPTLOG`
`§5.2i附4`** — `k=3` resident machines can buy the precondition to `P(green)≈98%`, but the `±1σ` of the question's `T`
is still **3.9x** the distance between decision points (resolving it needs `k≈47` ⇒ 329 resident machines ⇒
RSS ≈57.9GB = physically impossible), and **the killing point is not the working set** (21 machines ≈5.8GB / 22GB
available) but the structure "cross-instance pairing + denominator inflation". **B2 (a real drift corpus) was killed by
a zero-cost pre-check in `OPTLOG` `§5.2i附5`** — the pre-check probe only reads gate histograms, builds no arm, times
nothing ⇒ **it enters no record**, and its route verdict is "the stimulus of a real corpus may itself not exist"
⇒ **no ticket was filed and no measurement was produced**. ⇒ to touch measurement again only **B3 (change machine) /
B6 (same-instance switch, newly registered, touching it = touching the ruler for the sixth time)** remain, and both
need the user's nod.
🚨 Boundary: no added shots, no lowering of `TARGET=5.0`/`T_LO=0.60`/`thr_rel=0.05`, not one old table touched,
`swapper` still defaults to `"sync"`.

---

## 7. Limitations and future work (the honest version, no sugar-coating)

| category | specifics | impact |
|---|---|---|
| **corpus** | out-of-domain readme pastiche, **no held-out ppl** (`CORPUS=/path/to/wt2.txt` and it gets swapped the moment it arrives). ✅**The relative conclusions have been re-measured under a second corpus** (`0e0694aa`: second corpus = Shakespeare dialogue, 7314 token; the placed-wrong/placed-right tax `5.9x/20.9x/309x` reappears, the bf16 knife stays negligible); but the gc=3 resident member set changes with the corpus from `{0,2,1}` to `{0,5,2}` ⇒ on the small-gc end "static placement done right" is a function of the corpus; neither corpus is **strictly held-out** | absolute ppl is meaningless; the quality section reports relative quantities only (red line ②); across corpora only the **relative quantities** survive, the absolute scale does not; the standalone sub-section is **`paper_short §7.1`** |
| **the judge** | the judge is a same-architecture 8-choose-2 **fp32 dense MoE**, not the real SmolLM model | "swapping to the MoE itself" already costs 668~871x ⇒ this table's KL can only be read between gc settings and between dtypes |
| **the judge can be born dirty** | the judge `P` is born dirty: r3 phase-2 `\|P\|absmax=5.85e+19` ⇒ the reference KL was pushed to `2.564e+11` while **top-1 did not move a character**, the in-process gate ④ cross-difference is 0, and gate ⑨ still printed "33/33 quotable" | **the quality column of individual segments is void as a whole**; only the [reference-side absolute red line] + cross-bench archive reconciliation can catch it (methodology 36). The root cause is not settled; a probe is attached |
| **unit price across domains** | the mean cross-segment Spearman of `kappa_e` is **+0.16** (`u_e` is +0.62/+0.88/+0.81) | going live means **keeping one 8-float price table per domain**; re-pricing by online hits changes no decision ⇒ a shadow probe has no payoff |
| **sample resolution** | ✅**catch-up run is in the archive**: the `BS=512` quality table `13f8740c`. The remaining problem is only in the bs=1/128 cells: the top-1 denominator is that cell's bs, T=128 gives only **0.78pp**, T=1 is a **single-token coin** | the bs=512 setting's resolution **0.195pp** is usable; those bs=1 rows are still only readable for the order of magnitude of KL/TV |
| **the cross-process axis** | methodology 26 was just established: absolute values have second-process evidence only at bs=128, not at bs=1/32 | the 3 bimodal cells (all in bf16, where the CPU leg is heavy) get an interval, not a point |
| **the noise floor** | A 2~3%, C 8~18%; point-to-point exclusivity ⇒ small differences between adjacent gc are not readable | the only pairing in the tables is that dtype knife, and it still has to pass `配对可用` (pairing-usable) |
| **one-off per-instance offset** (in r4 it was called "fixed arm-identity effect", **r5 reversed the verdict**) | r4: different arm instances of the same configuration differ stably by **≈8%** (`48499173`); r5 (`1f69a7cb`): cross-process + randomised arm-creation order gives median **5.05%**, `σ_inst≈1.6%`, per-process worst cell `\|Δ%\|` range **4.73~7.42%** ⇒ it does not eat shots (n=25→48), does not eat the estimator (p50/block sum), does not eat the arm-creation order, **and does not eat the process count** | 🚨**this machine cannot resolve arm-to-arm differences below ≈5~7%** ⇒ on **this machine** anything smaller than that claiming "X is faster than Y" may not be read; the "BLK=24 + TARGET=5%" ruler is ruled [unusable], **verdict (fourth time)**. **B5 is killed by offline arithmetic**: `k=3` can buy precondition green (`P≈98%`) but the landing point stays unresolvable (the ±1σ of `T` = 3.9x the decision-point spacing; resolving needs `k≈47` = 329 instances = physically impossible) ⇒ all three ways of buying (more shots / more processes / more instances) are closed together. What transfers is the **measurement method** (pre-set decision tree + null control + per-instance offset check), not the numbers on this machine |
| **the dense tax** | ✅**fixed** (乙-2d traffic-sparse batching landed): zero-traffic experts per frame **35.8→3.6us(fp32, ÷9.9) / 20.6→1.8us(bf16, ÷11.4)**; after batching the bs=1 best **0.184ms** is **34.1%** faster than the same slice pure CPU `0.279ms` | landing scope = segmented: bs=1 uses traffic, bs≥32 uses cat (at bs=128 traffic loses badly, **+238.3%** `25746271`); evidence `64436f8a` / `f17fc0d0` / `25746271` |
| **placement rule v2** | ✅delivered and it is the default (`v2 = unit price + compute wall`): over four cells at bs=32 on average **2.8%** dearer, at bs=128 **4/4 hits on the global optimum, average excess 0.0%** (v1 was 61.3% / 114.6%). 🚨**still not fixed**: the two bs=1 cells `A fp32 +7.0%`, `C fp32 +8.1%` (the bf16 side +0.0%) — bs=1 has only 2 slots so the rule's inputs degenerate, and widening the tie_band is not the antidote; also the GPU leg lacks a bs compute term | the old price of a placed-wrong setting (v1 era) measures short by **+101~106%** for fp32; the two bs=1 cells must be cited as [known to be non-optimal for the rule], never as the engine's price |
| **unmeasured item: natural drift on real text** | "how much would picking the best member per segment actually save on a real corpus" **has never been measured** — only a **zero-cost pre-check** was done (`scan_drift.py`: reads gate histograms only, builds no arm, times nothing ⇒ **enters no record, cashes in nothing**), and its route verdict is "the stimulus itself may not exist, and the one-off price tag far exceeds the amount that could be eaten" ⇒ **no ticket filed** | 🚨this text gives **no conclusion, positive or negative,** on "the gain of dynamic placement on real text"; the ceiling anchor that is in the archive **belongs to synthetic drift only**, and its verdict round did not pass its preconditions (corroboration only) ⇒ this line is registered as an **unmeasured item**, not written up as a negative result |
| **scale and parallelism extrapolation** | every measurement = **one machine, one card, `E=8`/top-2, the 135M shape, 7244 tokens of out-of-domain activation**; no multi-GPU, no NCCL, no all-to-all of expert parallelism (EP), no 64+ experts, no downstream task | this text is **single-machine heterogeneity metrology + a placement criterion**; for "tens to hundreds of experts / multi-card EP / much larger models" these numbers **do not apply by even one word**; 🚨this text **compares no number against any published MoE inference system** (not run = not in a table); things like KTransformers are to be read as **motivation** only |
| **reproducing on another machine** | the **only** way to turn the cause of death "this machine cannot resolve arm-to-arm differences of ≲5~7%" is **to change the physical instance**; there is no second machine at hand ⇒ **not done**. The four layers of repetition this machine can buy (shots `n=25→48` / estimator `p50→block sum` / randomised arm-creation order / process count `NP=5`) **were all bought out in r1~r5**; the preliminary probe `probe_b3_machine.sh` is in the archive | "this ruler fails" is a conclusion about **this machine + this apparatus**, **not a universal negation**; for the real gain of dynamic placement see the **unmeasured item** row |
| **this machine's constants** | 42GB/s, 250GB/s, 12MB L3, no VNNI, 6C/12T, desktop core-grabbing | the numbers belong to this machine; what transfers is the **conclusions and the method**, not the numbers |
| **P2 frozen** | int4, biased packing, VNNI, register blocking, the gc-times-bmm fused graph for `down`, large-bs/prefill tuning | all of it is either on the refutation list or only done with a P0 backstop, see OPTLOG §3/§5.3 |
| **fp16** | at the toy scale the swiglu intermediate magnitude overflows to nan=223; the overflow did not reproduce on real weights | the only candidate left is bf16 (has an exponent width, 8-bit mantissa) |
| **the amplitude of the synthetic drift is fixed** (it is not a free parameter); "raise the amplitude and re-judge" is registered as **will not do** | synthetic drift = [the two blocks' god's-eye members all differ] + a zero-reaction-latency oracle ⇒ it is already the ceiling of that question. The reason for not doing this control is **constructive**, not an omission: ① the precondition `bottom_pair` measures **the null-control arms that never fire**, so it is **independent of** stimulus amplitude — bigger drift cannot buy precondition green; ② "would a large effect be resolvable" was already answered by the positive control C1 **passing the stars in 3/3 blocks** (`3efbe364`/`96ecadf1`) | this text **does not claim** "dynamic placement never pays at any amplitude", it is answerable only for **this apparatus at this amplitude** (an amplitude real text does not reach); the derivation is in `paper_short.md` §6.5 |
| **data sources not in `results/`** | the LADDER 14-knife cumulative bill and the two grey diamonds on `fig1` come from **`OPTLOG §1`** (a ledger table, **no md5**) | the **only exception** to the §4 index header's "the source of every number in the text", registered explicitly below the §4 table; reconciled cell by cell back into the archives by `make_ledger.py verify()` |

---

## 8. Reproducing (hard constraint: only this one machine, only this one interpreter)

```bash
cd labml_nn/transformers/glu_variants
# short names: red | real | base | metrics | swap | swiglu | dtype | t14 | dyn | err | decode | dynpair | swapcost | drifttrack
BS=1,32,128 GC=0,1,3,5,7,8 DTYPES=fp32,bf16 ROUNDS=7 ./run_all.sh swap   # the three-way exchange table
BS=128 GC=0,1,3,5,7,8 DTYPES=fp32,bf16 ROUNDS=7 ./run_all.sh swap          # second-process evidence (must be a separate process)
BS=1,32,128 GC=0,3,5,7,8 DTYPES=fp32,bf16 ROUNDS=3 ./run_all.sh metrics    # quality table (deterministic quantity, ROUNDS only proves gate ④)
# P1-②c the unit-price table (66 cells, slow) / P1-②d the criterion verdict round (33 rows, fast) —— both need loadavg<0.8
BS=512 PHASES=readme,input.txt,drift ROUNDS=1 ./run_all.sh err
GC=3 BS=512 FRAMES=64 PHASES=readme,input.txt,drift POLICIES=static_A,shuffle,static_p1,static_C,naive,hyst_1x,hyst_2x,hysteresis,price_cal,price_hit,price_orc ./run_all.sh dyn
# the `P1-②附` series (each needs header-line loadavg<0.8 to be admitted; verdict tables are run by a human who pastes the full text)
ROUNDS=25 ./run_all.sh dynpair      # ruler calibration: three arms on the same placement rotated + forward and reverse order (≈2min)
ROUNDS=3  ./run_all.sh swapcost     # migration price tag: sync vs async_staged (≈3min)
# the `§5.2i` series (drift tracking, 7 arms x 3 blocks x 24 frames, n=ROUNDS per cell; this is exactly the one r4 used)
(date; ROUNDS=48 NP=5 ./run_all.sh drifttrackN) 2>&1 | tee /tmp/p1i_drifttrack_r6.txt   # ≈40min; **per-process** header-line loadavg<0.8 to be admitted (next round number r6; B2/B5 are killed ⇒ it runs again only if B3/B6 get the nod)
```

- The interpreter **must** be `/home/adomlans/miniconda3/envs/dl_lab/bin/python`; **no matplotlib, no network** ⇒ the two figures are hand-drawn with PIL.
- Admission gate for timing tables: header line **loadavg < 0.8**, `ROUNDS ≥ 7`, same-process rotation pairing + p50, **min-of-N forbidden**.
- Verdict tables are **run by a human who pastes the full text**; anything the model runs itself (smoke/attribution/diagnostic) is marked "does not enter the record".

---

## 9. Going deeper (3 blogs + 1 short paper)

| piece | topic | in one sentence |
|---|---|---|
| `blog1_measurement.md` | 26 rules of measurement + pre-set decision trees | the cost of "telling the truth" on this machine: three error axes, two kinds of memory, two kinds of scale; 🚨§9: how a pre-set decision tree separates "no effect was measured" from "no effect could be measured", and how that ≈8% offset was reversed into "one-off per instance" by our own data |
| `blog2_two_walls_placement.md` | the two walls and the placement rule | on the CPU side the byte wall and the compute wall cross at ≈5 slot, on the GPU side you pay by weight byte ⇒ the rule needs both legs + a tie-breaking rule; 🚨§8: the fourth leg is the **unit price**, hit% is only its proxy; 🚨§9: the fifth leg is the **migration price tag**, and once it is measured the gain collapses to "a one-off placement at start-up" |
| `blog3_real_shape_bf16.md` | real shapes and reduced precision | the price of reduced precision rises with gc to −52%, the two separate bills int8:bf16 = 274~364x, zero-traffic experts 35us per frame |
| `paper_short.md` | the short paper (self-contained version) | four sections: the unit-price criterion (`u_e`, hitting the measured optimum in 3/3 segments) → the two walls + the dense tax → three knives on the migration price tag → 🚨**the negative result for dynamic placement** [= none of this project's four attempts passed its preconditions + the quantitative proof that the ruler is unusable, **not** a conclusion that "dynamic placement is ineffective"]; every number carries its archive md5 in parentheses |
