#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0-① 真权重 / 真激活: 把【玩具分布】换成 SmolLM-135M 的真东西, 看前 15 刀的结论还站不站得住。

这一刀要回答的三个问题(全是台账里没证过的):
  Q1 精度: 台账里每一条 relRMSE 都建立在"随机 std=1 权重 + std=1 激活"上。真权重 std≈0.25、
     真激活行范数只有 2~3(玩具是 ~sqrt(D)=32) => int8 的 amax/127 格点完全在另一个尺度上。
     所以每格同时量【三种激活】: 真激活 / 同尺度高斯 / std=1 玩具 —— 权重、形状、模型三档全同,
     唯一变的是激活分布, 这样才能把"精度差异"归给分布而不是归给形状。
  Q2 命中分布: 以前的 A/C 两档是 gate.bias=1.5 手工拧出来的热度。真路由(lm_head 取 E 行)
     偏不偏斜? 偏斜来自真方向还是我们造的? => 同时跑 gate=embed 与 gate=randn 两档对照。
     A/C 不再用 bias, 改成【把专家列表按真命中率排序】(gate_w 的行跟着走, 否则 id 和方向错位)。
  Q3 价格: 真形状 D=576/N=1536 比玩具 D=1024/N=2048 小 => 每专家 int8 从 6.29MB 掉到 2.65MB
     (swiglu), CPU 常驻工作集可能重新挤回 L3 => 字节墙高度变了, 于是把"异质 vs 纯GPU vs 纯CPU"
     这条主线价格搬到真分布上重量一遍(放置规则本身要不要重标定 = 下一刀, 这张表里 gc 钉死)。

延迟口径照旧: 同进程轮转 + 逐轮配对 + p50, 🚨 禁 min-of-N。

env: LAYERS=0,4,8,12,16,20,24,28  GATE=embed|randn  CAP_LAYER=8  HIT_BS=512
     BS=1,4,32,128  EKIND=swiglu  GC=5  ROUNDS=7  ITERS=15  T4=1
"""
import os, sys, time, statistics, torch
from gc import collect as _pycollect
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H
import real_smollm as R
import bench_swiglu as BSW          # 复用精度四笔(它有 __main__ 闸, import 不会跑表)

LAYERS = tuple(int(v) for v in os.environ.get("LAYERS", "0,4,8,12,16,20,24,28").split(","))
GATE = os.environ.get("GATE", "embed")
CAP_LAYER = int(os.environ.get("CAP_LAYER", 8))
HIT_BS = int(os.environ.get("HIT_BS", 512))
BS_LIST = [int(v) for v in os.environ.get("BS", "1,4,32,128").split(",")]
KINDS = [k for k in os.environ.get("EKIND", "swiglu").split(",") if k]
assert set(KINDS) <= {"gemm", "swiglu"}, KINDS
GC = int(os.environ.get("GC", 5))
K = 2
E = len(LAYERS)
ROUNDS = int(os.environ.get("ROUNDS", 7))
ITERS = int(os.environ.get("ITERS", 15))
T4_ROUNDS = int(os.environ.get("T4_ROUNDS", 3))
DO_T4 = os.environ.get("T4", "1") == "1"
PROBE_BS = int(os.environ.get("PROBE_BS", 64))
SEED = 42
MODE = "L2_零CUDA+pinned"
FULL = dict(gpu_impl="cat", graph_scope="full", n_threads=6, tok_block=8,
            pre_impl="fused", l2_stream="single", cpu_sched=1, d2h_wait="main")
T4CFG = dict(gpu_impl="sparse", graph_scope="off", n_threads=1, tok_block=1,
             pre_impl="torch", l2_stream="sep", cpu_sched=0, tail_graph=False)

print(f"[机器状态] {H.machine_state()}")
dims = R.dims()
D, N, NL = dims["D"], dims["N"], dims["layers"]
print(f"[真源] SmolLM-135M: D={D} N={N} 层数={NL} | 取 {list(LAYERS)} 层的 MLP 当 {E} 个专家"
      f" | top_k={K} | 路由={GATE} | 激活=第 {CAP_LAYER} 层 MLP 入口(真英文文本跑出来的)")
print(f"[被测] bs={BS_LIST} | 形状={KINDS} | gc={GC}(钉死, 不跑 T13 规则) | "
      f"轮转 {ROUNDS} 轮 x 每轮 {ITERS} 次取中位")

BASE = R.real_src(LAYERS, gate_mode=GATE)
XREAL, _ = R.capture(layer=CAP_LAYER, bs=HIT_BS)
assert XREAL.shape[1] == D, (XREAL.shape, D)
torch.manual_seed(SEED)
ROW = XREAL.norm(dim=1).mean().item()
XS = torch.randn(HIT_BS, D) * (ROW / D ** 0.5)      # 同尺度高斯: 换分布不换幅度
XU = torch.randn(HIT_BS, D)                          # std=1: 玩具那套激活
ACTS = [("真激活", XREAL), ("同尺度高斯", XS), ("std=1玩具", XU)]


def fwd(m, x):
    with torch.no_grad():
        return m(x, async_mode=True, mode=MODE)


def perm_src(order):
    """按 order 重排专家: order[新 id] = 旧 id; gate_w 的行必须跟着走。"""
    return dict(experts=[BASE["experts"][i] for i in order],
                gate_w=BASE["gate_w"][list(order)].contiguous(),
                gate_b=BASE["gate_b"].clone(),
                note=BASE["note"] + f" perm={list(order)}")


def hits_of(gate_w, rows):
    """按帧累计 top-k 命中(放置决定的是"哪些专家长期热", 与单帧大小无关)。"""
    W = gate_w.float()
    cnt = torch.zeros(E)
    for i in range(0, rows.shape[0], PROBE_BS):
        sc = rows[i:i + PROBE_BS] @ W.t()
        cnt += torch.bincount(sc.topk(K, dim=1).indices.flatten(), minlength=E).cpu()
    return cnt


# ================================================================ 段0 统计对照
print("\n== 段0 权重/激活统计: 真 vs 玩具(int8 吃的就是 amax/127 这个步长) ==")
print(f"{'来源':<22}{'矩阵':>4}{'std':>10}{'max|w|':>10}{'int8步长':>12}")
for r in R.weight_stats():
    print(f"{r[0]:<22}{r[1]:>4}{r[2]:>10.4f}{r[3]:>10.3f}{r[4]:>12.3e}")
torch.manual_seed(SEED)
toy = torch.randn(4096, 4096)
print(f"{'玩具(入库 randn std=1)':<19}{'w':>4}{toy.std().item():>10.4f}"
      f"{toy.abs().max().item():>10.3f}{toy.abs().max().item()/127:>12.3e}")
print(f"{'激活':<22}{'':>4}{'std':>10}{'max|x|':>10}{'行范数mean':>11}")
for nm, xt in ACTS:
    print(f"{nm:<22}{'':>4}{xt.std().item():>10.4f}{xt.abs().max().item():>10.3f}"
          f"{xt.norm(dim=1).mean().item():>11.2f}")
mb = lambda n: n / 2**20
print(f"  每专家 int8: 真 swiglu 3x{D}x{N}={mb(3*D*N):.2f}MB / 真 gemm {D}x{N}={mb(D*N):.2f}MB"
      f"   (玩具 swiglu {mb(3*1024*2048):.2f}MB / gemm {mb(1024*2048):.2f}MB)"
      f" => 真形状是玩具 swiglu 的 {D*N/(1024*2048):.2f}x")
print(f"  字节墙/专家: 真 swiglu {3*D*N/H.CPU_BW_EFF*1e6:.0f}us (玩具 149us); "
      f"CPU 常驻 {E-GC} 个 = {mb(3*D*N*(E-GC)):.1f}MB vs L3 12MB")

# ================================================================ 段1 命中分布
print(f"\n== 段1 真路由命中分布({HIT_BS} 行, 每 {PROBE_BS} 行一帧累计; 对照 = 同一批真激活) ==")
cnt_e = hits_of(BASE["gate_w"], XREAL)
cnt_r = hits_of(R.router_rows(E, D, "randn"), XREAL)
pct = lambda c: [round(float(v) / float(c.sum()) * 100, 1) for v in c]
print(f"  路由={GATE:<6} 命中% = {pct(cnt_e)}  最热/最冷 = {float(cnt_e.max())/max(float(cnt_e.min()),1.0):.1f}x")
print(f"  路由=randn  命中% = {pct(cnt_r)}  ← 对照组: 这档若接近均匀(1/{E}={100/E:.1f}%),"
      f" 说明上面那档的偏斜来自真投影方向, 不是我们 bias 拧的")
order_A = torch.argsort(cnt_e, descending=True).tolist()
order_C = torch.argsort(cnt_e).tolist()
CFGS = [("A", order_A, "放对: 真最热的 gc 个进 GPU"),
        ("C", order_C, "放错: 真最冷的 gc 个进 GPU")]
print(f"  A 序(新id0=最热旧层)= {order_A}   C 序(新id0=最冷旧层)= {order_C}")
print(f"  旧 id 下前 8 帧的 top_k = {(XREAL[:8] @ BASE['gate_w'].t()).topk(K, dim=1).indices.tolist()}")


def mk(gc, order, kind, cfg=None):
    torch.manual_seed(SEED)
    return H.HeteroMoE(D, E, N, gpu_expert_count=gc, top_k=K, expert_kind=kind,
                       real_src=perm_src(order), cpu_delay_ms=0.0,
                       **{**FULL, **(cfg or {})}).cuda()


def win():
    """干净显存窗口(方法论第 17 条): gc+sync+empty_cache+读空载+清高水位。"""
    _pycollect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    if base > 2.0:
        print(f"  🚨窗口不干净: 空载仍有 {base:.1f}MB => 本行显存列作废")
    return base


def warm(m, x, n=12):
    for _ in range(n):
        fwd(m, x)
    torch.cuda.synchronize()


def rounds(m, x, n):
    """n 轮, 每轮 ITERS 次取中位。"""
    out = []
    for _ in range(n):
        s = []
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fwd(m, x)
            torch.cuda.synchronize()
            s.append((time.perf_counter() - t0) * 1e3)
        out.append(statistics.median(s))
    return out


def acc(m, x, acts):
    """整条 L2 vs 本实例 fp32 裁判。acts: [(名字, CPU 上的行块)] """
    res = {}
    ts, ti = m.route(x)
    for nm, xt in acts:
        xx = xt[:x.shape[0]].cuda()
        _ts, _ti = m.route(xx)
        ref = m.fp32_reference(xx, _ts, _ti)
        o = fwd(m, xx)
        mx, _, rel = H.err_metrics(o, ref)
        # 🚨 顺带记下【这一档激活自己把多少 slot 派到了 CPU 腿】: 换激活分布会同时换路由,
        #    不记它就分不清"误差变大是量化变差"还是"流量换腿了"。
        res[nm] = (rel, mx, int(torch.isinf(o).sum()) + int(torch.isnan(o).sum()),
                   int((_ti >= m.gpu_expert_count).sum()))
    return res, ts, ti


def legs_four(m, x, ts, ti):
    """精度四笔 + 中间量幅度(只量 CPU 腿)。返回 (文本行列表, 最大|g|/|u|/|h|)。"""
    n_slot = int((ti >= m.gpu_expert_count).sum())
    if n_slot == 0:
        return ["  精度四笔: 🚨跳过 —— 本帧 CPU 腿 0 个 slot, 整条 relRMSE 不是量化的代价"], None
    hot = m._hot_compute(x, ts, ti).double()
    leg_kernel = (m(x, async_mode=True, mode=MODE).double() - hot)
    w = BSW.prep_w(m)
    L = BSW.cpu_legs(m, x, ts, ti, w)
    rms = L["ref"].pow(2).mean().sqrt().item()
    r1, r2, r3 = (BSW.rel(L["s_w"], L["ref"]), BSW.rel(L["s_wx"], L["ref"]),
                  BSW.rel(L["s_all"], L["ref"]))
    d = (leg_kernel - L["s_all"]).abs().max().item()
    scale = L["s_all"].abs().max().item()
    ok = "OK(≤1 LSB 级)" if scale and d / scale < 1e-4 else "🚨 对不上, kernel 有 bug"
    lines = [f"  精度四笔(只量 CPU 腿 {n_slot} 个 slot, 分母=本实例 CPU 腿 fp32 真值 rms={rms:.3g}):",
             f"    ①只有 W 量化      relRMSE={r1:.4f}%",
             f"    ②W+x 量化        relRMSE={r2:.4f}%   (+x 这一笔 {r2-r1:+.4f}pp)",
             f"    ③W+x+h 再量化     relRMSE={r3:.4f}%   (+h 这一笔 {r3-r2:+.4f}pp)",
             f"    ④kernel 实测      relRMSE={BSW.rel(leg_kernel, L['ref']):.4f}%   "
             f"④vs③ 绝对={d:.3g} (③量级 {scale:.3g} => 相对 {d/scale if scale else float('nan'):.2g}) {ok}"]
    # 中间量幅度 —— fp16 溢出的判据(上限 65504), 乙-2c 那 223 个 inf 就是这么来的
    mg = mu = mh = 0.0
    with torch.no_grad():
        xq = x.double()
        for e in range(len(m.cpu_experts_t)):
            mask = (ti == m.gpu_expert_count + e)
            b = mask.nonzero(as_tuple=True)[0]
            if b.numel() == 0:
                continue
            g = xq[b] @ w[f"g{e}"]; u = xq[b] @ w[f"u{e}"]
            h = torch.nn.functional.silu(g) * u
            mg = max(mg, g.abs().max().item()); mu = max(mu, u.abs().max().item())
            mh = max(mh, h.abs().max().item())
    del w
    lines.append(f"    中间量幅度(判 fp16 上限 65504): max|g|={mg:.3g} max|u|={mu:.3g} max|h|={mh:.3g}"
                 + ("  🚨超 fp16" if max(mg, mu, mh) > 65504 else "  => fp16 不溢出"))
    return lines, (mg, mu, mh)


# ================================================================ 逐格测量
ARMS = ["⓪T4 基线", "①异质 gc=%d" % GC, "③纯GPU gc=%d" % E, "④纯CPU gc=0"]
for kind in KINDS:
    for bs in BS_LIST:
        x = XREAL[:bs].cuda()
        for label, order, note in CFGS:
            print("\n" + "=" * 100)
            print(f"{kind} | bs={bs:>3d} | 配置{label} {note} | gc={GC} | CPU 驻 {E-GC} 个专家"
                  f" | 新序命中率%={pct(cnt_e[order])}")
            print("=" * 100)
            # ---- 墙判定(用真形状自己的字节数)
            probe = mk(GC, order, kind)
            ts, ti = probe.route(x)
            per = [int((ti == GC + e).sum()) for e in range(E - GC)]
            eb = probe.expert_bytes()
            mx = max(per) if per else 0
            wall = "算力墙" if mx >= 5 else "字节墙"
            n_slot = int((ti >= GC).sum())
            print(f"    本帧 CPU 腿 slot={n_slot} 每专家 slot={per} 最大 {mx} → 【{wall}】"
                  f"(交叉点≈5) | 每专家 {mb(eb):.2f}MB 字节墙≈{eb/H.CPU_BW_EFF*1e6:.0f}us"
                  f" | 常驻 {mb(eb*(E-GC)):.1f}MB vs L3 12MB")
            # ---- 正确性闸门 + 三种激活的精度(都在 ① 这一档量)
            o_sync = None
            with torch.no_grad():
                o_sync = probe(x, async_mode=False, mode=MODE)
                a1, a2 = fwd(probe, x), fwd(probe, x)
            dmode = max((o_sync - a1).abs().max().item(), (a1 - a2).abs().max().item())
            res, ts, ti = acc(probe, x, ACTS)
            print(f"  闸门 三模式互差={dmode:.9f}  inf/nan="
                  f"{sum(v[2] for v in res.values())}  本帧 CPU 腿 slot={int((ti >= GC).sum())}")
            print("  整条 L2 vs 本实例 fp32 裁判(权重/形状三档全同, 只换激活): " + "  ".join(
                f"{nm} relRMSE={v[0]:.4f}%/max_abs={v[1]:.3g}/CPU腿{v[3]}slot"
                for nm, v in res.items()))
            if kind == "swiglu":
                ln, _ = legs_four(probe, x, ts, ti)
                for l in ln:
                    print(l)
            # 🚨 探针模型必须在开【独占显存窗口】之前删掉: 它自己那 ~50MB 权重会留在
            #    空载基线里, 于是每个窗口的"峰值显存"都含一份别人的模型(乙-②冒烟踩过)。
            del probe
            torch.cuda.empty_cache()
            solo_pk, solo_t = {}, {}
            # ---- 独占窗口: 一次只让一个模型活着(显存只能这样量)
            for arm, gc, cfg in ((ARMS[0], GC, T4CFG), (ARMS[1], GC, None),
                                 (ARMS[2], E, None), (ARMS[3], 0, None)):
                if arm == ARMS[0] and not DO_T4:
                    continue
                base = win()
                m = mk(gc, order, kind, cfg); warm(m, x, 20)
                torch.cuda.synchronize()
                solo_pk[arm] = (torch.cuda.max_memory_allocated() / 2**20) - base
                solo_t[arm] = rounds(m, x, T4_ROUNDS if arm == ARMS[0] else ROUNDS)
                if arm in (ARMS[2], ARMS[3]):
                    r2, _, _ = acc(m, x, ACTS[:1])
                    res[arm] = r2[ACTS[0][0]]
                del m
                win()
            # ---- 轮转窗口: 三条通路同时常驻, 逐轮配对
            m1, m3, m4 = mk(GC, order, kind), mk(E, order, kind), mk(0, order, kind)
            for m in (m1, m3, m4):
                warm(m, x)
            rot = {ARMS[1]: [], ARMS[2]: [], ARMS[3]: []}
            for r in range(ROUNDS):
                rot[ARMS[1]].append(rounds(m1, x, 1)[0])
                rot[ARMS[2]].append(rounds(m3, x, 1)[0])
                rot[ARMS[3]].append(rounds(m4, x, 1)[0])
            del m1, m3, m4
            torch.cuda.empty_cache()
            print(f"{'通路':<14}{'p50独占':>9}{'p50轮转':>9}{'σ%轮':>6}{'vs①轮转':>10}"
                  f"{'更慢轮':>8}{'relRMSE真':>11}{'峰值显存MB':>11}{'tok/s轮转':>10}")
            r1m = statistics.median(rot[ARMS[1]])
            for arm in ARMS:
                if arm not in solo_t:
                    continue
                if arm == ARMS[0]:
                    # 🚨 T4 只在独占段量 => 它的对比对象是【①的独占 p50】, 不是轮转 p50
                    t4 = statistics.median(solo_t[arm]) / statistics.median(solo_t[ARMS[1]]) - 1
                    print(f"{arm:<14}{statistics.median(solo_t[arm]):>9.3f}{'—':>9}{'—':>6}"
                          f"{t4:>+9.1%}{'—':>8}{'—':>11}"
                          f"{solo_pk[arm]:>11.1f}{bs/statistics.median(solo_t[arm])*1e3:>10.0f}")
                    continue
                v = rot[arm]
                p50 = statistics.median(v)
                sg = statistics.pstdev(v) / p50 * 100
                slower = sum(1 for i in range(ROUNDS) if v[i] > rot[ARMS[1]][i])
                accv = res[arm][0] if arm in res else res[ACTS[0][0]][0]
                print(f"{arm:<14}{statistics.median(solo_t[arm]):>9.3f}{p50:>9.3f}{sg:>6.0f}"
                      f"{p50/r1m-1:>+10.1%}{slower:>7d}/{ROUNDS}{accv:>10.4f}%{solo_pk[arm]:>11.1f}"
                      f"{bs/p50*1e3:>10.0f}")
            # 🚨 路由随激活变 => 最后这三行只和【上面 ①异质 那一行】同路由可比, 别和 ③④ 混读
            print("  ①异质 换激活(路由也跟着换): " + "  ".join(
                f"{nm} {res[nm][0]:.4f}%/CPU腿{res[nm][3]}slot"
                for nm, _ in ACTS if nm in res))
            print(f"  ①异质 轮间: " + " ".join(f"{t:.3f}" for t in sorted(rot[ARMS[1]])))
            torch.cuda.empty_cache()

print("\n判读提示: ① 三种激活那一行 = 分布的代价(权重/形状全同, 只有激活变了);"
      "\n        ② 段0 的每专家字节 < 玩具 => 真形状下 CPU 工作集可能重新进 L3, 字节墙没那么高;"
      "\n        ③ ③纯GPU 若仍反超 => '放错就全热' 这条结论在真分布下也成立;"
      "\n        ④ 四笔里的 +h 那一笔 = swiglu 独有的量化代价, 真权重下它可能和玩具不是一个数量级。")
