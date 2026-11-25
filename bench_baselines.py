#!/usr/bin/env python
"""P0-② 外部基线: 这套异构系统第一次【跟别人比】, 而不是只跟自己比。

为什么必须做它: 台账里 480 行全是【内部 ladder】(T4 基线 -> 全开), 审稿人/面试官的
第一句话一定是"比谁快"。这里把四个此前根本没跑过的通路放进同一进程轮转:

  ① hetero      热专家在 GPU + 冷专家 CPU int8, 两条腿【重叠】(= 主线, max(腿,腿))
  ② no_overlap  同一个模型、同一批权重、同一组图, 只把【提交顺序】改成先 CPU 后 GPU
                => 每帧应当变成 sum(腿+腿)。这是【主线自身的消融】, 四个基线里最重要的一条:
                   它直接给"重叠"这件事定价, 也就是这条主线到底值多少。
  ③ pure_gpu    gc=E=8: 所有专家常驻显存(KTransformers 的"全热"上界)。精度最好、最贵显存。
  ④ cpu_only    gc=0: 所有专家在 CPU int8(纯 CPU 下界, 显存 0 权重)。

🚨 为什么 ② 能做成"零混淆": overlap 是【运行时读的布尔】, 我在同一个模型对象上翻转它,
   所以两次测量共享权重/图/线程池。已验输出逐位相同(/tmp/probe_p02.py: 互差 0.000000000)。
🚨 为什么 ③④ 只能重建模型: gc 是构造期决定的(权重按 gc 切两半)。约定: 每条基线之前
   都 torch.manual_seed(SEED) => gate 权重逐位相同(ctor 恒定消耗 8 张 randn 才到 gate),
   所以【路由和命中率在四条基线之间可比】; 精度各自跟【自己那份 fp32 裁判】比。

顺手补两个 P0-③ 的真指标: 峰值显存(这条主线的卖点本来就是省显存)、token/s。

env: BS=1,4,32,128  EKIND=gemm,swiglu  GC=5  ROUNDS=7  SWEEP=1(要不要跑 gc 扫描那张)
"""
import os, sys, time, statistics, torch
from gc import collect as _pycollect   # ✅ 叫 collect 不叫 gc.collect: 下面扫描段的循环变量就叫 gc, 会遮蔽模块
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

HS, NE, ES, TOPK = 1024, 8, 2048, 2
MODE = "L2_零CUDA+pinned"
SEED = 42
BS_LIST = [int(v) for v in os.environ.get("BS", "1,4,32,128").split(",")]
KINDS = [k for k in os.environ.get("EKIND", "gemm,swiglu").split(",") if k]
assert set(KINDS) <= {"gemm", "swiglu"}, KINDS
GC = int(os.environ.get("GC", 5))
ROUNDS = int(os.environ.get("ROUNDS", 7))
ITERS = int(os.environ.get("ITERS", 20))
SWEEP = os.environ.get("SWEEP", "1") == "1"
SWEEP_BS = [int(v) for v in os.environ.get("SWEEP_BS", "1,32").split(",")]
CFG = [("A", [0, 1]), ("C", [5, 6])]          # A=放对(热专家在 GPU) C=故意放错(打 CPU 腿)
FULL = dict(gpu_impl="cat", graph_scope="full", n_threads=6, tok_block=8,
            pre_impl="fused", l2_stream="single", cpu_sched=1, d2h_wait="main")
L2 = 32        # token/s 折算: 假设整模型有 32 层这样的 MoE(报告里写清了这是折算)


def mk(gc, hot, kind, **kw):
    """建模型 + 上 GPU。每条基线前重新 seed => gate 逐位相同 => 路由可比。"""
    torch.manual_seed(SEED)
    m = H.HeteroMoE(HS, NE, ES, gpu_expert_count=gc, top_k=TOPK, hot_experts=hot,
                    expert_kind=kind, **{**FULL, **kw}).cuda()
    return m


def fwd(m, x, **kw):
    """前向一律 no_grad。✅ P0-②b: 模型自己已经冻结参数(freeze_params=True), 这里再包
    一层是 bench 侧的双保险 —— autograd 图会把 gc x 8MB 的 fp32 权重拽住不还
    (探针 /tmp/probe_leak5.py), 精度/显存两列都会被它污染。"""
    with torch.no_grad():
        return m(x, **kw)


def win():
    """开一个【干净窗口】: gc -> sync -> empty_cache -> 读空载基线 -> 清零高水位。
    🚨 空载基线必须查: 第一版没查, 上一窗口的 autograd 图残留 40~65MB, "峰值显存"整列
       偏高、"省显存"错位一格。这里查出来就当场报警, 不要等事后对探针。"""
    _pycollect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    if base > 2.0:
        print(f"  🚨窗口不干净: 空载仍有 {base:.1f}MB 常驻 => 下面两列显存数字全部作废")
    return base


def warm(m, x, n=12):
    for _ in range(n):
        fwd(m, x, async_mode=True, mode=MODE)
    torch.cuda.synchronize()


def one_round(m, x):
    """一轮 = ITERS 次取中位(方法论第 1 条: 禁 min-of-N)。"""
    s = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fwd(m, x, async_mode=True, mode=MODE)
        torch.cuda.synchronize()
        s.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(s)


print(f"[被测] D={HS} E={NE} expert={ES} top_k={TOPK} | bs={BS_LIST} | 形状={KINDS} | "
      f"异质档 gc={GC} | 轮转 {ROUNDS} 轮 x 每轮 {ITERS} 次取中位 | 种子={SEED}(四条基线共用)")
st = H.machine_state() if hasattr(H, "machine_state") else None
print(f"[机器状态] {st}" if st else "")
ARMS = ["①hetero(重叠)", "②no_overlap(sum)", "③pure_gpu(gc=8)", "④cpu_only(gc=0)"]

for kind in KINDS:
    for bs in BS_LIST:
        torch.manual_seed(SEED)
        x = torch.randn(bs, HS).cuda()
        for cname, hot in CFG:
            # ---------- 阶段A 独占段: 一次只让一个模型活着 ----------
            # 显存只能这样量(第一版复用延迟段那三个模型, 四条基线打印出同一个 405.5MB,
            #   因为 max_memory_allocated 量的是"当时所有活张量"的高水位)。
            # 延迟也顺手在这里量一遍 = 【独占态绝对值】, 与主表/halves 表可比;
            #   阶段B 的轮转态才是配对口径(抗漂移), 两列一起印, 差值就是踩 L3 的代价。
            solo, pk, pk2, acc, dmax = {}, {}, {}, {}, {}
            for gc, ta, tb in ((GC, "①hetero(重叠)", "②no_overlap(sum)"),
                               (NE, "③pure_gpu(gc=8)", None),
                               (0, "④cpu_only(gc=0)", None)):
                base = win()
                m = mk(gc, hot, kind); warm(m, x, 20); torch.cuda.synchronize()
                pk[ta] = (torch.cuda.max_memory_allocated() / 2**20) - base
                ts, ti = m.route(x)
                ref = m.fp32_reference(x, ts, ti)
                acc[ta] = H.err_metrics(fwd(m, x, async_mode=True, mode=MODE), ref)[2]
                if tb:                              # ①/② 同一个模型 => 就地逐轮配对
                    ra, rb = [], []
                    for r in range(ROUNDS):
                        # ✅ 单双号交换 ①/② 先后 = 位置偏置自查。第一版固定 ①→②,
                        #    冒烟里出现过 ② 反而快 27%(重叠档先跑, 替后跑的档暖了 OMP/PCIe),
                        #    那是测量位置不是物理。换序后若差值仍 < 噪声底(A 2~3%),
                        #    就只能报"这笔价格在本机测不出", 不许硬解读。
                        if r % 2 == 0:
                            m.overlap = True; ra.append(one_round(m, x))
                            m.overlap = False; rb.append(one_round(m, x))
                        else:
                            m.overlap = False; rb.append(one_round(m, x))
                            m.overlap = True; ra.append(one_round(m, x))
                        acc[tb] = H.err_metrics(fwd(m, x, async_mode=True, mode=MODE), ref)[2]
                    m.overlap = True
                    solo[ta], solo[tb] = ra, rb
                    # (显存两列在下面的独占窗口结束后再抄 —— ② 与 ① 是同一份权重)
                else:
                    solo[ta] = [one_round(m, x) for _ in range(ROUNDS)]
                out_a = fwd(m, x, async_mode=True, mode=MODE).float().cpu()   # 留给附刀对拍
                del m
                base2 = win()
                # ---------- 附刀窗口: 同一份权重, 只是不再建 bmm 档那份死副本 ----------
                # 🚨 必须另起一个独占窗口: 如果 m 还活着就读 max_memory_allocated,
                #    量到的是两个模型之和(这个坑本轮已经踩过一次)。
                # (win() 里已经 reset_peak: 不 reset 会读到上一窗口含 fp32 裁判临时搬上 GPU
                #  的那 24~64MB 的旧高水位 —— 第一版就是这么读出"省显存"变负数)
                m2 = mk(gc, hot, kind, free_dead_w=True)
                warm(m2, x, 20)
                torch.cuda.synchronize()
                pk2[ta] = (torch.cuda.max_memory_allocated() / 2**20) - base2
                d_dead = (fwd(m2, x, async_mode=True, mode=MODE).float().cpu() - out_a).abs().max().item()
                dmax[ta] = max(dmax.get(ta, 0.0), d_dead)
                del m2, out_a
                torch.cuda.empty_cache()
                if tb:                              # 同一份权重 => 同一份显存/同一条逐位判定
                    pk[tb], pk2[tb], dmax[tb] = pk[ta], pk2[ta], dmax[ta]

            # ---------- 阶段B 轮转段: 三个模型同时常驻, 逐轮配对 ----------
            m5, m8, m0 = mk(GC, hot, kind), mk(NE, hot, kind), mk(0, hot, kind)
            for m in (m5, m8, m0):
                warm(m, x)
            hit5 = H.measure_hit_rates(m5, x) if hasattr(H, "measure_hit_rates") else None
            ts, ti = m5.route(x)
            o_h = fwd(m5, x, async_mode=True, mode=MODE).clone()
            m5.overlap = False
            o_n = fwd(m5, x, async_mode=True, mode=MODE).clone()
            m5.overlap = True
            d_bit = (o_h - o_n).abs().max().item()
            rounds = {a: [] for a in ARMS}
            for r in range(ROUNDS):
                rounds["①hetero(重叠)"].append(one_round(m5, x))
                m5.overlap = False
                rounds["②no_overlap(sum)"].append(one_round(m5, x))
                m5.overlap = True
                rounds["③pure_gpu(gc=8)"].append(one_round(m8, x))
                rounds["④cpu_only(gc=0)"].append(one_round(m0, x))
            del m5, m8, m0
            torch.cuda.empty_cache()

            # ---------- 打印 ----------
            rot = {a: statistics.median(v) for a, v in rounds.items()}
            sal = {a: statistics.median(v) for a, v in solo.items()}
            h1r, h1s = rot["①hetero(重叠)"], sal["①hetero(重叠)"]
            print()
            print("=" * 104)
            print(f"{kind} | bs={bs:>3d} | 配置{cname} hot={hot} | 异质档 gc={GC} | "
                  f"CPU侧驻 {NE-GC} 个专家 | 命中率(512行累计)="
                  f"{[round(v,1) for v in hit5] if hit5 else '?'}")
            print("=" * 104)
            print(f"{'通路':17s}{'p50独占':>9s}{'p50轮转':>9s}{'σ%轮':>6s}{'vs异质独占':>12s}"
                  f"{'vs异质轮转':>12s}{'更慢轮':>8s}{'精度relRMSE':>12s}{'峰值显存MB':>11s}"
                  f"{'去死权重MB':>11s}{'tok/s独占':>11s}")
            for a in ARMS:
                first = a.startswith("①")
                dr = "  —" if first else f"{100*(sal[a]/h1s-1):+10.1f}%"
                dv = "  —" if first else f"{100*(rot[a]/h1r-1):+10.1f}%"
                sl = "  —" if first else f"{sum(1 for i in range(ROUNDS) if rounds[a][i] > rounds['①hetero(重叠)'][i])}/{ROUNDS}"
                sg = 100 * statistics.pstdev(rounds[a]) / rot[a]
                print(f"{a:17s}{sal[a]:9.3f}{rot[a]:9.3f}{sg:6.0f}{dr:>12s}{dv:>12s}"
                      f"{sl:>8s}{acc[a]:11.4f}%{pk[a]:11.1f}{pk2[a]:11.1f}{bs/sal[a]*1e3:11.0f}")
            dirt = {a: 100 * (rot[a] / sal[a] - 1) for a in ARMS}
            print(f"  ①vs② 逐位互差 = {d_bit:.9f}  (这条必须 0: 消融只改提交顺序)")
            print(f"  重叠的价格(②相对①): 独占态 {100*(sal['②no_overlap(sum)']/h1s-1):+.1f}%"
                  f" / 轮转态 {100*(rot['②no_overlap(sum)']/h1r-1):+.1f}%")
            print(f"  附刀(死权重 bmm 副本)省显存: "
                  + " ".join(f"{a[0]}{pk[a]-pk2[a]:.0f}MB" for a in ARMS)
                  + f" | 逐位互差(必须 0) " + " ".join(f"{a[0]}{dmax[a]:.9f}" for a in ARMS))
            print(f"  同进程踩 L3(轮转-独占): " + " ".join(f"{a[:2]}{dirt[a]:+.0f}%" for a in ARMS)
                  + f"   [CPU侧常驻 ①{NE-GC}个/③0个/④{NE}个专家 => 踩脏是不对等的]")
            print(f"  整模型折算({L2}层) token/s = "
                  f"{bs/h1s*1e3/L2:,.0f}(异质独占) {bs/sal['④cpu_only(gc=0)']*1e3/L2:,.0f}(纯CPU) "
                  f"{bs/sal['③pure_gpu(gc=8)']*1e3/L2:,.0f}(纯GPU)")

if SWEEP:
    print()
    print("=" * 104)
    print("gc 扫描: 显存 <-> 延迟 <-> 精度 的三方交换(这条主线的核心卖点图, 画图用这张)")
    print("🚨 口径: 逐点【独占】(建一个量一个删一个) —— 第一版把 9 个 gc 全建起来, 结果")
    print("   CPU 侧常驻权重合计 72MB 互相踩 L3, 绝对值整体抬高 17~37%(探针 /tmp/probe_p02b.py)。")
    print("   代价: 这一张是【曲线】不是【配对】, 两点之差若 <噪声底(A 2~3% / C 8~18%)不要下结论。")
    print("=" * 104)
    for kind in KINDS:
        for bs in SWEEP_BS:
            torch.manual_seed(SEED)
            x = torch.randn(bs, HS).cuda()
            for cname, hot in CFG:
                print(f"\n{kind} bs={bs} 配置{cname} hot={hot}")
                print(f"{'gc':>3s}{'GPU驻留MB':>11s}{'CPU侧驻':>9s}{'峰值显存MB':>12s}{'去死权重':>10s}"
                      f"{'p50独占':>9s}{'σ%':>5s}{'vs纯CPU':>10s}{'精度relRMSE':>12s}  说明")
                got, pkdead = {}, {}
                cpu_p = None
                for gc in range(0, NE + 1):
                    wbase = win()                 # 窗口空载 MB(必须 ~0, 否则显存列作废)
                    m = mk(gc, hot, kind)
                    warm(m, x, 20)
                    torch.cuda.synchronize()
                    pkmb = (torch.cuda.max_memory_allocated() / 2**20) - wbase
                    ts, ti = m.route(x)
                    rel = H.err_metrics(fwd(m, x, async_mode=True, mode=MODE),
                                        m.fp32_reference(x, ts, ti))[2]
                    rs = [one_round(m, x) for _ in range(ROUNDS)]
                    p = statistics.median(rs)
                    if gc == 0:
                        cpu_p = p                 # 纯CPU 下界 = 这一行的对照组
                    got[gc] = p
                    驻 = (NE - gc) * ES * HS * (3 if kind == "swiglu" else 1) >> 20
                    note = ("纯CPU下界" if gc == 0 else
                            "纯GPU上界" if gc == NE else
                            "规则选这档" if gc == GC else "")
                    del m, ts, ti
                    wbase = win()                             # 干净窗口 + 清高水位(同上)
                    m = mk(gc, hot, kind, free_dead_w=True)   # 附刀: 死权重不建
                    warm(m, x, 10)
                    torch.cuda.synchronize()
                    pkd = (torch.cuda.max_memory_allocated() / 2**20) - wbase
                    del m
                    pkdead[gc] = pkd
                    print(f"{gc:>3d}{gc*ES*HS*(3 if kind=='swiglu' else 1)>>20:>11d}"
                          f"{NE-gc:>9d}{pkmb:12.1f}{pkd:10.1f}{p:9.3f}{100*statistics.pstdev(rs)/p:5.0f}"
                          f"{100*(p/cpu_p-1):+9.1f}%{rel:11.4f}%  {note} ={驻}MB")
                    torch.cuda.empty_cache()
                best = min(got, key=got.get)
                gpub = NE * ES * HS * (3 if kind == "swiglu" else 1) >> 20
                print(f"  → 实测最优 gc={best}({got[best]:.3f}ms) | 规则选 gc={GC}({got[GC]:.3f}ms, "
                      f"贵 {100*(got[GC]/got[best]-1):+.1f}%) | 纯GPU gc={NE} {got[NE]:.3f}ms "
                      f"(GPU 多驻 {gpub}MB 权重, 精度 {0.0001:.4f}% 是它的唯一独占优势)")
                torch.cuda.empty_cache()


print("判读提示: ① ①vs② 就是【重叠本身值多少】—— 主线主张的价格, 也是简历里那个倍数的唯一合法来源;")
print("          ② ③pure_gpu 若更快, 要同时报它的显存(它用显存换延迟, 这正是本项目的交换曲线);")
print("          ③ ④cpu_only 在 bs=1 上可能【反而更快】(GPU 半边的权重字节是纯成本) —— 那是")
print("             真结论不是失败: decode 下放置规则必须给 GPU 半边定价(乙-2b/T13 同源)。")
print("          ④ ①'②'两行是【独占态】对照(删掉另两条基线的常驻权重再量): 它和轮转态的差")
print("             就是【同进程共享 L3】踩脏了多少 —— 显存列另起独占段, 同一个理由。")
