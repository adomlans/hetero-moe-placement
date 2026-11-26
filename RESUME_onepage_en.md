# Zhixuan Cao (`曹志瑄`) · Target role: AI Infra internship (LLM inference systems / performance engineering)

> **This file is the English translation of `RESUME_onepage.md`.** It is a translation, not a second text: every number,
> every evidence pointer and every archive identifier is copied verbatim from the Chinese original; where the two
> disagree, **the Chinese text wins**. Chinese is kept inside backticks only where it is a literal printed identifier.

Nanjing, Jiangsu · github.com/adomlans/hetero-moe-placement · Jiangsu Police Institute · Cybersecurity and Law Enforcement · sophomore

> One sentence: on a desktop i5-10400 + RTX3060 I **hand-wrote a CPU-GPU heterogeneous MoE inference engine**, and then,
> with a measurement system built out of **pre-registered decision trees + empty controls + gates**,
> **I falsified the very gain I had originally set out to prove**, and added a quantitative proof of **why this machine's ruler does not qualify**.
> All of the evidence (42 verdict tables + 5 process dumps + 4000+ log lines) is open source.

---

## 0. Education · Intent

**Jiangsu Police Institute** · Cybersecurity and Law Enforcement · **sophomore** ｜ based in Nanjing, Jiangsu ｜ intent: **AI Infra internship** (LLM inference systems / performance engineering)
Availability: **winter and summer breaks only** (full time inside a break)
My major is not in the CS track ⇒ so every claim below is built as a **reproducible command + raw archive**, not a spoken conclusion:
one i5-10400 + RTX3060, no team, no server, **everything checkable is in the repository**.

---

## 1. A negative result: **four pre-registered, pre-committed decision trees, and not one of them ever passed the "ruler pre-check"** ⇒ dynamic expert placement **was never validly evaluated** (🚨this is **not** a claim that "it does not work")

- Built a rig whose membership really does drift (7 arms × 3 blocks × 24 frames, with an **empty control** / a **positive control** / an **oracle upper-bound anchor**);
  **before each run** the pre-conditions, the question and the prediction were written into the docstring, and the landing is evaluated by code.
- Four trees, four unsettled bookings, and the only red item was always the empty-control floor: `底_pair` 6.91% → 4.18% → 8.41% → **5.05%**
  (cross-process NP=5, short by **0.05pp** of `TARGET=5%`) ⇒ the landing is **never ratified**.
- The root cause was located: **a one-off per-instance offset** (`σ_inst≈1.6%`, per-process worst-cell range `4.73~7.42%`) — it
  **does not respond to more passes** (n=25→48), **not to the estimator** (p50 / block sums), **not to arm build order**, **not to the process count**
  ⇒ on this machine an inter-arm gap of ≲5~7% is not resolvable.
- The escape route "buy one more level of replication and it becomes decidable" is **killed by offline arithmetic**: `k=3` buys the pre-check up to `P(绿)≈98%`,
  but the landing stays undecidable (the question's `±1σ` is **3.9x** the spacing between decision points; deciding needs `k≈47` = 329 instances = physically impossible).
- And a reading of my own got **reversed** on the way: the decision counters show **0 decisions per arm in the timed rounds** ⇒ the "saves 20%~27%" was
  **a one-off placement at start-up**, not a dynamic one.

## 2. A transferable output: the placement criterion moved from "hotness" to "unit price"

- `u_e` = the KL measured when expert e alone is put on the CPU leg; the criterion = minimise `Σ 命中_e × u_e` under a budget.
  **0 free parameters, one offline table of 8 floats.**
- Checks: within-segment Spearman **+1.00 in all three segments**; across segments 23 cells 0.92~1.11; additivity predicted/measured 0.902/1.070/0.920;
  **in 3/3 corpus segments the resident set it selects is the segment's measured-KL minimum**; the old hotness/coverage criterion is **1.60x more expensive** on the drifting segment.
- Its boundary was measured as well: cross-domain `kappa_e` has an inter-segment Spearman of only **+0.16** ⇒ in production you **carry one price table per domain**.

## 3. One cost map + two real gains

- **The two walls**: CPU byte wall 49~63us/expert @42GB/s, compute wall 7us/slot @288GMAC/s, **crossing point ≈5 CPU-resident experts**
  ⇒ "how many experts stay resident in the CPU" is not a capacity parameter, it is a **wall-choice** parameter. The GPU leg pays by weight byte (fp32→bf16 end to end **−15.4%/−33.9%**).
- **The dense tax** (a zero-traffic expert still pays 35.8us/20.6us every frame) → **traffic-sparse batching** pushes it down to **3.6us/1.8us** (÷9.9 / ÷11.4)
  ⇒ the best `bs=1` decode is **0.184ms** < the same share of **pure CPU** at 0.279ms = **−34.1%**, overturning "decode has to fall back to pure CPU".
  🚨The boundary is stated too: batching only lands at `bs=1`; at `bs=128` it loses **+238.3%**.
- **The price of overlap is positive in 16/16 cells** (+8.5%~+67.6%); `swiglu bs=128` vs pure CPU **5.47x** / vs pure GPU **1.51x**
  (🚨a ratio must always carry its bs: at `bs=4` it is only 1.11~1.63x, at `bs=1` pure CPU wins; external phrasings like "CPU at 3ms ⇒ 3.5x" measure **1.0x**).
- **Three knives on the migration price tag** (each knife's measurement definition overturned the previous one): synchronous one-off **19.9ms**, asynchronous staged **0.22x**
  ⇒ but **the default is not changed** (changing a default needs the user's nod).

## 4. Engineering surface / skills

- **CPU**: AVX2 `maddubs` int8 GEMV (no VNNI path), int8 symmetric quantisation and re-quantising `h`, OpenMP `guided` scheduling and the straggler tax (1 contended core: 839→569us).
- **GPU**: CUDA Graph capture and the "graph must not be re-recorded" gate, pinned buffers, D2H/H2D overlap, traffic-sparse batching, skinny GEMM calibrated at 250GB/s.
- **Measurement**: same-process rotation pairing (forward/reverse order), p50 / block sums (**min-of-N is banned**), empty control / positive control / oracle anchor, gates ⓪~⑨
  (bitwise equivalence, slot↔expert mapping re-checked, placement independence, zero migration on the static arm) — **the gates caught 4 real bugs**
  (`D=576` tail-block out-of-bounds read, `static thread_local` taking an empty copy inside an OpenMP parallel region, the referee born-dirty, the fake 27% produced by `ROUNDS=2`).
- Languages / stack: Python, C++/AVX2, CUDA/Triton, PyTorch (`torch.utils.cpp_extension`), performance profiling and statistics (pairing, Spearman, scaling-law fits).

## 5. The three sentences I volunteer as "do not cite"

1. "Dynamic placement saves 20%~27%" — **void**; that was a one-off placement at start-up.
2. "3.5x when the CPU latency is 3ms" — **measured 1.0x**.
3. "A 5% gap on this machine can be measured" — **it cannot**, and the arithmetic already shows the fix cannot be bought (329 instances).

> Repository: <https://github.com/adomlans/hetero-moe-placement> — every verdict table carries date / MHz / loadavg in its header line;
> the plotting scripts verify themselves against the archives and refuse to draw a figure when a cell does not match.
