#!/usr/bin/env python
"""判决表 T14: CPU kernel 的 OpenMP 行调度 —— static,32(预先平摊) vs guided(自取递减块)。

【这一刀的来历, 两次改判】
  ① 最初的假设是"每满 8 个 token 就把 2MB 权重整趟重读, 白搬 14MB DRAM" => 换循环顺序。
     换序做完, 逐位闸门过了, 但延迟基本没动(C/d0 -15% 但 σ18%, C/d3 反而 +4.4%)。
  ② 纯 CPU 卡尺(/tmp 探针, 不碰 CUDA)量出来的真相:
       单核 13.5 MAC/cyc/core, 6 线程 288 GMAC/s, 而且【MAC/s 与权重重扫几遍无关】
       ([8]=1趟 与 [64]=8趟 的 MAC/s 一样) => 这个 kernel 是 AVX2 maddubs 的【算力墙】,
       重扫的权重全在 L2/L3 里不要钱 => 换序否证。
  ③ 那帧内 811us 是怎么来的? 同一份输入, 只在 CPU 上挂【1 个抢核进程】:
       static,32: 独占 478/559us -> 有抢核 839/905us   ← 帧内量的就是 811us, 对上了
       guided   : 独占 453/505us -> 有抢核 569/590us   (-32%)
     机制: 帧内主线程要同时做 graph replay / D2H 入队, 它和绑在 core0 的 OMP 线程0
     同核 => 线程0 半速 => static 下【整个并行区等它一个】。guided 让慢者自动少拿活。

【口径】同进程轮转 + 逐轮配对 + 中位数(min 会奖励方差大的变体, T8 的教训)。
🚨 本表刻意多测一列"外挂抢核进程": 真实帧里【永远】有这个条件(主线程 + 桌面),
   不测这行就只会看到"guided 和 static 打平", 从而把这一刀否证掉。
🚨 上一版这张表的协议是错的: gc_of() 里先建模型再抽 x => x 与主表不同 => 64 个槽位
   样本抖动 => 放置规则选到 gc=5(CPU 侧几乎没活), 整张表在判一个空载 kernel。
   现在 x 在建任何模型之前抽, 并把命中率打印出来跟主表对账。
"""
import os, sys, time, atexit, statistics, subprocess, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

ROUNDS = int(os.environ.get("ROUNDS", 11))
D, E, N, K, BS = 1024, 8, 2048, 2, 32
CFG = [("A", [0, 1]), ("C", [5, 6])]
# 主表 delay=0 的命中率, 用来给本表的 x 对账(不一致 = 协议又漂了)
EXPECT_HITS = {"A": [40.6, 46.9, 3.1, 0.0, 0.0, 0.0, 1.6, 7.8],
               "C": [1.6, 4.7, 1.6, 0.0, 0.0, 45.3, 40.6, 6.2]}
# (名字, ctor kwargs)
VARS = [("static,32(原状)", dict(cpu_loop=0, cpu_sched=0)),
        ("guided(新)", dict(cpu_loop=0, cpu_sched=1)),
        ("guided+换序", dict(cpu_loop=1, cpu_sched=1))]
NAMES = [v[0] for v in VARS]

# 🚨 x 必须在【建任何模型之前】抽, 且和主表同一次 seed —— 否则放置规则看到的
#    槽位样本和主表不是同一批, 会选出别的 gc(上一版就是这么把 gc 从 3 测成 5 的)
torch.manual_seed(42); X = torch.randn(BS, D).cuda()


def gc_of(label, hot):
    """和主表同协议地跑一遍放置规则, 顺便把命中率/每专家 slot 数打出来对账。"""
    torch.manual_seed(42)
    probe = H.HeteroMoE(D, E, N, 2, K, hot_experts=hot, cpu_delay_ms=0.0).cuda()
    hits = H.measure_hit_rates(probe, X)
    # 🚨 字节数要传【每个专家的权重字节】: int8 = N*D = 2MB, fp32 = 4*N*D = 8MB。
    #    上一版误传 E*D(8KB): 两条腿同比缩小 => 选出的 gc 没错, 但打印的"预测 us"
    #    小了 256 倍(0us/1us), 是个假数 —— 记录在 OPTLOG.md 的"本表自查抓到的坑"
    gc, pred = H.plan_gpu_capacity(hits, N * D, 4 * N * D, gpu_expert_count=2)
    slots = [round(h / 100 * BS * K, 1) for h in hits]
    ok = all(abs(a - b) < 0.15 for a, b in zip(hits, EXPECT_HITS[label]))
    cpu = " ".join(f"e{e}={slots[e]}" for e in range(gc, E) if slots[e] > 0)
    print(f"  {label}: gc={gc}(预测{pred:.0f}us) 命中率对账 {'OK' if ok else '🚨和主表不一致!'}"
          f"  CPU侧有流量的专家 slot 数: {cpu}")
    del probe
    return gc


print("=== 放置规则 v2 (和主表同 seed 同 x) ===")
GC = {l: gc_of(l, h) for l, h in CFG}

M = {}
for label, hot in CFG:
    for d in (0.0, 3.0):
        for nm, kw in VARS:
            torch.manual_seed(42)
            # 🚨 形参是 (hidden, num_experts, expert_size, gpu_expert_count, top_k) ——
            #    上一版这里把 N/E 写反了, 建成 2048 个专家, 建模型就要几分钟且判的是废 kernel
            M[(label, d, nm)] = H.HeteroMoE(
                D, E, N, GC[label], K, hot_experts=hot, cpu_delay_ms=d,
                n_threads=6, tok_block=8, gpu_impl="cat", graph_scope="full",
                pre_impl="fused", l2_stream="single", **kw).cuda()

# ---------- 0a. 正确性闸门(kernel 级, 便宜且指哪打哪) ----------
#   sched 只决定"哪条线程算哪一行", loop/tb 只决定"哪些 token 共用一次权重载入"
#   —— 每个 (行 i, token t) 的点积和同一 (b,i) 上的累加顺序都不该变 => 必须逐位 0
def caliper(m):
    """把真帧的 CPU 半边输入(权重 + 量化激活 + 路由)取出来, 供 kernel 级对拍/卡尺用。"""
    ts, ti, xq, xs = m._prepare(X)
    torch.cuda.synchronize()
    args = (xq.cpu().contiguous(), xs.cpu().contiguous(), ti.cpu().contiguous(),
            ts.cpu().contiguous(), m.cpu_w, m.cpu_experts_scale)
    cnt = [int((ti == e).sum().item()) for e in range(m.gpu_expert_count, E)]
    nslot = sum(cnt)
    npass = sum(-(-c // 8) for c in cnt)     # tb=8 时这张表要整趟重读几次权重
    return args, nslot, npass

print("=== 正确性闸门 A: kernel 级 loop×sched×tb 全部必须与【原状 loop0/sched0/tb8】逐位相同 ===")
bad = 0
for label, _ in CFG:
    args, nslot, npass = caliper(M[(label, 0.0, NAMES[0])])
    ref = torch.zeros(BS, N)
    H.avx_gemv_ext.avx2_moe_cpu_experts_int8_into(*args, ref, M[(label,0.0,NAMES[0])].gpu_expert_count,
                                                  True, 6, 8, -1, 0, 0)
    for loop in (0, 1):
        for sc in (0, 1):
            for tb in (1, 2, 4, 8):
                out = torch.zeros(BS, N)
                H.avx_gemv_ext.avx2_moe_cpu_experts_int8_into(
                    *args, out, M[(label,0.0,NAMES[0])].gpu_expert_count, True, 6, tb, -1, loop, sc)
                dd = (out - ref).abs().max().item()
                bad += dd != 0.0
                print(f"  {label} loop={loop} sched={sc} tb={tb}  差 = {dd:.9f}  "
                      f"{'OK' if dd == 0 else '🚨FAIL'}")
if bad:
    print("🚨 调度/换序把结果改错了 —— 停止, 不许看延迟"); sys.exit(1)

# ---------- 0b. 端到端逐位闸门(顺带证明 graph/worker 那条链也没被带动) ----------
print("\n=== 正确性闸门 B: 端到端 L2 输出, guided / 换序 vs 原状 逐位 ===")
bad = 0
for label, _ in CFG:
    base = M[(label, 0.0, NAMES[0])](X, async_mode=True, mode="L2_零CUDA+pinned").clone()
    torch.cuda.synchronize()
    for nm in NAMES[1:]:
        got = M[(label, 0.0, nm)](X, async_mode=True, mode="L2_零CUDA+pinned")
        torch.cuda.synchronize()
        dd = (got - base).abs().max().item()
        bad += dd != 0.0
        print(f"  {label} {nm:<20} 差 = {dd:.9f}  {'OK' if dd == 0 else '🚨FAIL'}")
if bad:
    print("🚨 端到端输出变了 —— 停止, 不许看延迟"); sys.exit(1)


# ---------- 1. kernel-only 卡尺: 同一批权重/同一批 slot, 只换调度 ----------
# 🚨 out 必须【外面预分配】: 在计时循环里 torch.zeros(32,2048) 是 ~10us,
#    对配置A(整核才几十 us)来说是 20% 的系统误差, 会把"guided 略慢"这种假结论造出来。
def one_caliper(args, gc, loop, sched, out, iters=30):
    f = lambda: H.avx_gemv_ext.avx2_moe_cpu_experts_int8_into(
        *args, out, gc, True, 6, 8, -1, loop, sched)
    for _ in range(3): f()
    ts = []
    for _ in range(9):
        t0 = time.perf_counter()
        for _ in range(iters): f()
        ts.append((time.perf_counter() - t0) / iters)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def spin(cores):
    ps = [subprocess.Popen([sys.executable, "-c", "while True: pass"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                           preexec_fn=lambda c=c: os.sched_setaffinity(0, {c})) for c in cores]
    atexit.register(lambda: [p.kill() for p in ps])
    time.sleep(1.0)
    return ps


print("\n=== kernel-only 卡尺(纯 CPU, 真帧的权重+slot 分布; 每格 = min/p50 us) ===")
print(f"{'配置':<5}{'抢核':<10}{'slot':>5}{'原状趟数':>7}"
      + "".join(f"{n:>24}" for n in NAMES))
for label, _ in CFG:
    m0 = M[(label, 0.0, NAMES[0])]
    args, nslot, npass = caliper(m0)
    macs = nslot * N * D
    out_buf = torch.zeros(BS, N)
    for ctag, cores in [("无", []), ("1@cpu0", [0]), ("2@cpu0,1", [0, 1])]:
        ps = spin(cores)
        cells, raw = [], []
        for nm, kw in VARS:
            mn, md = one_caliper(args, m0.gpu_expert_count, kw["cpu_loop"],
                                 kw["cpu_sched"], out_buf)
            cells.append(f"{mn*1e6:>14.0f}/{md*1e6:<9.0f}")
            raw.append(mn)
        for p in ps: p.kill()
        for p in ps: p.wait()
        # 算力上限对照 = 本行【原状无抢核】那格的 min, 不额外再跑一遍(白烧 0.5s)
        print(f"{label:<5}{ctag:<10}{nslot:>5}{npass:>5}" + "".join(cells)
              + f"   算力上限对照 {macs/raw[0]/1e9:.0f} GMAC/s")


# ---------- 2. 端到端帧延迟: 无抢核 / 有抢核 各一遍 ----------
def sweep(tag, cores, delays=(0.0, 3.0)):
    keys = [(l, d, nm) for l, _ in CFG for d in delays for nm in NAMES]
    res = {k: [] for k in keys}
    ps = spin(cores) if cores else None
    for r in range(ROUNDS):
        for l, _ in CFG:
            for d in delays:
                for nm in NAMES:
                    res[(l, d, nm)].append(
                        H.bench(M[(l, d, nm)], X, True, "L2_零CUDA+pinned")[0])
    if ps:
        for p in ps: p.kill()
        for p in ps: p.wait()
    la = open('/proc/loadavg').read().split()
    print(f"\n=== {tag} | loadavg={la[0]}/{la[1]} 轮转 {ROUNDS} 轮 | L2 端到端 p50 ms ===")
    print(f"{'配置':<4}{'delay':>6}{'变体':<20}{'p50':>8}{'σ%':>5}{'min':>8}"
          f"{'vs原状':>9}{'更慢轮数':>10}")
    for l, _ in CFG:
        for d in delays:
            base = res[(l, d, NAMES[0])]
            for nm in NAMES:
                v = res[(l, d, nm)]
                worse = sum(1 for u, w in zip(base, v) if w > u)
                sd = statistics.pstdev(v) / statistics.median(v) * 100
                print(f"{l:<4}{d:>6.1f}{nm:<20}{statistics.median(v):>8.3f}{sd:>5.0f}"
                      f"{min(v):>8.3f}"
                      f"{(statistics.median(v)/statistics.median(base)-1)*100:>8.1f}%"
                      f"{worse:>8}/{len(v)}")


sweep("端到端 无抢核(桌面安静时≈这一行)", [])
sweep("端到端 外挂 1 个抢核进程@cpu0 ← 真实帧的条件", [0])
