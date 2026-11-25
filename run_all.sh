#!/bin/bash
# 统一入口。真正的修复(OMP_PLACES/OMP_PROC_BIND)已经写进 hetero_moe_int8.py 顶部,
# 直接 python 单跑也生效; 这个脚本只是把测量条件打印出来 + 保证用对解释器。
#
#   ./run_all.sh                      # 主表(含轮转总账) —— 最慢, 几分钟
#   ./run_all.sh red                  # 红线校验, ~15s。改完任何异步/图代码【先跑这个】
#   ./run_all.sh scope                # 判决表: 同进程轮转 A/B(off/hot/full), 分辨真假收益
#   ./run_all.sh budget full          # 分账 + 固定成本明细(graph_scope=full)
#   ./run_all.sh micro                # 纯 CPU kernel: 路由分布 x tok_block x 线程
#   ./run_all.sh d2h                  # 主机入队成本探针
#   ./run_all.sh aff 6 "标签"          # 绑核 A/B
#   ./run_all.sh place                # 判决表: GPU 容量 gc vs CPU 侧【被摸到】的权重字节 (T9 放置规则)
#   ./run_all.sh d2hwait              # 判决表: "等 D2H"放主线程/worker —— 两种卡尺+逐轮配对 (T8 终审)
#   ./run_all.sh sweep                # 判决表: 放置规则在 bs=1..32 上还能不能选中实测最优
#   ./run_all.sh pre                  # 融合前段 kernel 自身的逐位闸门 + 延迟(bs 扫描)
#   ./run_all.sh t11                  # 判决表: 前段 torch(12张aten) vs fused(2张triton), 轮转配对
#   ./run_all.sh halves              # 分账: 每个 gc 的 GPU半边/CPU半边 各自多少 (T13 放置规则重标定)
#   ./run_all.sh t12                  # 判决表: GPU半边 二流+Event / 单流 / 单流+尾图, 轮转配对
#   ./run_all.sh t14                  # 判决表: CPU kernel 的 OMP 行调度 static,32 vs guided(抗掉队) + kernel卡尺/外挂抢核列 (T14)
#   ./run_all.sh decode              # 判决表: bs=1/4/32/128 扫描 —— 真 decode 下前 14 刀还剩多少 + 最优线程数随 bs 漂 (乙-1)
#   ./run_all.sh swiglu             # 判决表: 真形状(w1/w3/w2 三个矩阵=6MB) vs 演示形状(2MB) —— 延迟+精度四笔+两堵墙重标 (乙-2)
#   ./run_all.sh pairtax            # 归因探针: 同格【单档驻留 vs 两档在场/还跑另一档】差多少 —— 乙-1b 那条水位差到底是谁造成的 (乙-1c)
#   ./run_all.sh solopair           # 判决表: 同进程内【solo 口径 vs 轮转口径】各测一遍 swiglu/gemm 比值 —— 已发布比值要不要按 solo 口径更正 (乙-1e)
#   ./run_all.sh bimodal            # 归因探针: 双峰臂的高簇那几发, 同发的 /proc 环境量一不一样 —— 双峰是桌面抢核还是驻留稳态 (乙-1f)
#   ./run_all.sh dynpair            # 判决表: 【放置完全相同】的三台引擎同进程轮转+正反两序 —— dyn 那列跨策略 p50 还能不能读 (P1-②附2)
#   ./run_all.sh swapcost           # 判决表: 迁移价签 sync vs AsyncSwapper(侧 stream+pinned+预存表) —— 「换一次 ~15ms=3~4 帧」这张价签该不该重贴 (P1-②附3)
#   ./run_all.sh drifttrack         # 判决表: 段内最优成员真的移动时 dyn 滞回开关追不追得上 —— 追踪率 T + oracle-per-block 上限锚 + 反应延迟 (P1-②附6)
#   ./run_all.sh drifttrackN      # §5.2i附3 B1 换装置轮: NP 个【独立进程】各跑一遍 drifttrack(建臂序各自随机)
#                                 #   + 跨进程组合器印 K‴ 落点 —— 消「臂身份固定效应」, 🚨 不是第五次加发数
#   ./run_all.sh dtype              # 判决表: GPU 半边热专家权重 fp32->bf16/fp16 —— 每帧少读一半字节值多少 + 放置规则重定价 (乙-2c)
#   ./run_all.sh dtrate             # 归因探针: 裸 mm(不套图) 的 fp32/bf16/fp16 等效带宽 —— 判断降精该兑现多少、差距卡在哪张算子 (乙-2c)
#   ./run_all.sh base               # 判决表: 四个【外部基线】同进程轮转 —— 异质(重叠) / 不重叠 / 纯GPU(gc=E) / 纯CPU(gc=0), 附峰值显存与 token/s (P0-②)
#   ./run_all.sh metrics             # 判决表: 质量指标 —— 覆盖真模型第8层MLP出口后的 top-1一致率/KL/TV/ΔNLL (P0-③)
#   ./run_all.sh swap              # 判决表: 显存↔延迟↔质量 三方交换(真权重, 逐点独占 + dtype 轮转配对 + 跨bench对账闸门) (P0-④a)
#   ./run_all.sh dyn               # 判决表: 运行时【动态专家放置】值不值 —— 三方(延迟/质量/迁移成本)同格 + 决策日志 + 迁移不改数值闸门 (P1-②)
#   ./run_all.sh exact               # 判决表: CPU kernel 有没有算错 —— 真kernel vs 理想int32 vs 饱和仿真 (T10)
#   ./run_all.sh hetero_moe_int8.py   # 也接受真文件名, 原样透传
cd "$(dirname "$0")"
# 🚨 上面那段"用法"是注释, 每行都必须以 # 开头 —— 2025-11-20 我插新的一行时漏了 #,
#    那行就成了可执行语句, run_all.sh 递归调自己, 18 分钟炸出 38000 个 bash 把机器打死。
#    下面这道闸是防复发的保险丝(递归时立刻退出而不是继续 fork)。
[ -n "$RUN_ALL_GUARD" ] && { echo "🚨 run_all.sh 被递归调用了(用法注释少了 # ?)" >&2; exit 99; }
export RUN_ALL_GUARD=1   # 🚨 必须 export, 子进程才看得见
export PYTHONIOENCODING=utf-8
PY=${PY:-$HOME/miniconda3/envs/dl_lab/bin/python}   # 🚨 别用 base 的 python3.14(profiler 是坏的)
echo "== 机器状态 =="
echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" \
     "MHz=$(grep -m1 'cpu MHz' /proc/cpuinfo | awk '{print $4}')" \
     "loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "可用核数=$(nproc)  物理核=$(grep -m1 'cpu cores' /proc/cpuinfo | awk -F: '{print $2}')"
echo "🚨 loadavg > 1.5 时绝对值整体偏高(桌面在抢核), 只看同一次运行内的相对值"
[ $# -eq 0 ] && set -- hetero_moe_int8.py
case "$1" in                    # 短名 -> 脚本
  red)    set -- check_redlines.py "${@:2}";;
  scope)  set -- bench_scope_ab.py "${@:2}";;
  budget) set -- budget_check.py "${@:2}";;
  micro)  set -- bench_kernel_micro.py "${@:2}";;
  d2h)    set -- bench_d2h_enqueue.py "${@:2}";;
  aff)    set -- bench_affinity.py "${@:2}";;
  dist)    set -- bench_dist_ab.py "${@:2}";;
  place)   set -- bench_placement.py "${@:2}";;
  halves)  set -- bench_halves.py "${@:2}";;
  d2hwait) set -- bench_d2hwait_ab.py "${@:2}";;
  sweep)   set -- bench_rule_sweep.py "${@:2}";;
  exact)   set -- bench_int8_exactness.py "${@:2}";;
  pre)     set -- bench_pre_fuse.py "${@:2}";;
  t11)     set -- bench_t11_ab.py "${@:2}";;
  t12)     set -- bench_t12_ab.py "${@:2}";;
  t14)     set -- bench_t14_ab.py "${@:2}";;
  decode)  set -- bench_decode.py "${@:2}";;
  swiglu)  set -- bench_swiglu.py "${@:2}";;
  pairtax) set -- bench_pairtax.py "${@:2}";;
  solopair) set -- bench_solopair.py "${@:2}";;
  bimodal)  set -- bench_bimodal.py "${@:2}";;
  dynpair)  set -- bench_dynpair.py "${@:2}";;
  swapcost)  set -- bench_swapcost.py "${@:2}";;
  drifttrack)  set -- bench_drifttrack.py "${@:2}";;
  drifttrackN)
    NPv=${NP:-5}; OUTD=${OUTD:-/tmp/dt_r5}
    mkdir -p "$OUTD"; find "$OUTD" -maxdepth 1 -name 'p*.json' -delete
    echo "== B1 换装置轮(§5.2i附3): NP=$NPv 个独立进程 -> $OUTD | 每进程 ROUNDS=${ROUNDS:-48} 逐字不动(🚨 不是加发数) =="
    for i in $(seq 1 "$NPv"); do
      # 🚨 等空载闸门: 上一进程的 12 线程会把 1-min loadavg 顶上去(冒烟实测 p2 头载 1.89 > 入库门槛 0.8)。
      #    每 10s 查一次, <0.8 才起下一个进程; 上限 300s, 超时打印 🚨 但照跑(头载列会记下这笔账)。
      waited=0
      while :; do
        la=$(cut -d' ' -f1 /proc/loadavg)
        awk -v x="$la" 'BEGIN{exit !(x+0<0.8)}' && break
        if [ "$waited" -ge 300 ]; then echo "🚨 p$i 等空载超时 ${waited}s loadavg=$la —— 照跑, 头载列记账"; break; fi
        sleep 10; waited=$((waited+10))
      done
      echo "---- 进程 p$i $(date '+%F %T') loadavg=$(cut -d' ' -f1 /proc/loadavg) 等空载=${waited}s ----"
      BUILD_SEED=$i PROCID=p$i DUMP="$OUTD/p$i.json" "$PY" bench_drifttrack.py || exit 1
    done
    echo "== $PY combine_drifttrack.py $OUTD =="
    "$PY" combine_drifttrack.py "$OUTD" || exit 1
    exit 0;;
  dtype)   set -- bench_dtype.py "${@:2}";;
  dtrate)  set -- bench_dtype_rate.py "${@:2}";;
  base)    set -- bench_baselines.py "${@:2}";;
  real)    set -- bench_real.py "${@:2}";;
  metrics) set -- bench_metrics.py "${@:2}";;
  swap)    set -- bench_swap.py "${@:2}";;
  dyn)     set -- bench_dyn.py "${@:2}";;
  err)     set -- bench_err.py "${@:2}";;
esac
echo "== $PY $* =="
exec "$PY" "$@"
