#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1-②附6 · 真漂移追踪轮: 段内最优成员真的随时间移动时, dyn 的滞回开关到底追不追得上。

【这一刀只答那一句】
  「动态放置省 20%~27% 延迟」这句主张到 §5.2g 为止被拆过两次:
    ① §5.2f 轮转配对 + 空对照 => 落 D 挂账【底_pair 8.9% = 7 发不够】;
    ② §5.2g 发数标度 => 落 E1、甲【引擎身份】否证 => 尺子磨到 n≈30 能读 2.5%。
  但 §5.2g 顺手挖出更硬的一条: 【dyn 计表轮 468 发 H1/H2 决策 0 次】=> 那三段的 p50 差全部
  长在【开局一次性定成员】上(H1 末态 [0,6,5] ≈ 段3 上帝集、H2 末态 [0,2,5] ≈ 段2 上帝集),
  也就是「自适应」这三个字到今天为止【一次都没有被考核过】。
  🚨 0 次不是 bug, 是测量构造: 每段喂的是同一批 64 帧同一个 x => 冷发追上一次之后窗内统计
  就恒定 => 判据再也等不到「收益 > 价签」那一步。所以真问题不是「dyn 省多少」, 而是:
     **当最优成员在段内真的移动时, 开关开不开、开几次、能追上上限几成?**

【块构造 = 本刀唯一的新东西】
  原来一段 = 一个 x 连跑 FRAMES 帧。现在一段 = NBLK 个【块】, 每块 BLK 帧、各自一份 x,
  块与块之间 GPU 成员的最优集合【被人为挪动】:
    blk0  自然行(与 r2/r3 的 readme 段同一批挑行)   => 上帝成员 = A序前 GC = 开局成员【不错】
    blk1  从 readme 前 POOL 行里挑「命中 order_A[GC] 最多」的 BS 行 => 把一个 CPU 侧专家捧热
    blk2  同上但捧 order_A[GC+1]                    => 最优成员再挪一次【 dyn 必须二次开火】
  挑行只用 route 分数, 与 bench_dynpair.drift_ids 同一把尺; 🚨「块间上帝成员两两不同」是
  K0 前置硬条件, 不满足 = 刺激没造出来 => 直接挂账, 不许把「没开火」读成「自适应没用」。

【臂(全部常驻、全程不 del; 与 bench_dynpair 同一套建臂路)】
  A1/A2/A3  三台逐字同配置的 static_A = 空对照组 => 本轮自己的 底_pair
  C1        static_C = 正对照【判读提示⑥ 同源】: 它不过星标 = 这把尺子坏了, 本刀负读全废
  H1/H2     hyst_1x / hyst_2x = 被测的自适应两档(gain=1/2, cooldown=16, max_swaps=1)
  O1        【oracle-per-block 上限】: 每个块边界在计时窗外强制 flip 到该块的实测上帝成员
            —— 它付的是【同一笔 flip】(生产 sync 路, 同一个 sw.total_us), 只是零反应延迟
            => 它是「自适应做到极致」的上界, 不是「白送迁移」的空头锚点

【口径(六条, 别看错)】
  1) 一发 = 一台引擎把这个多块段跑一遍; 一轮 = 各臂各一发, 🚨奇数轮反序、偶数轮正序(位次两序对称)。
  2) 逐帧计时 = 前后各 sync 一次(与 dyn/dynpair 同一把帧长尺); 块边界与 flip 都落在【帧与帧
     之间】=> 不进任何一帧的 µs。🚨 这条与 §5.2h 的 F 项同源, 所以表 4 单独把 flip 自身列出来。
  3) 配对 = 【同一轮同一块】内 臂/A1, 跨轮取几何均值; 正序轮/反序轮各算一次 => 两序底。
  4) 追踪率 T = (1 - GAIN_dyn)/(1 - GAIN_O1), GAIN = 各块配对比值再跨块取几何。
     T=1 = 吃到 oracle 的全部上限; T=0 = 白开火; T<0 = 开了火比不放开还慢。
  5) 反应延迟 = 块边界到「GPU 成员集合 == 该块上帝成员」的帧数(逐帧采样 slot2eid, 免费)。
     🚨 它和 T 是两条独立证据: 延迟大但 T 高 = 块太长; 延迟小但 T 低 = 判据换错了人。
  6) 价签不在本刀重测: 表 4 直接引 §5.2h附2 归档 `22ff0967` 的 once(sync)=4.89 帧=19.9ms
     与稳态惩罚 P≈4.6ms/帧, 只用来算「回本需要几帧」并与本轮块长对照 = 对账列, 不是判据。
  7) 每发【起跑线对齐】= 成员回到起跑序 + dyn 臂滑窗/冷却/禁令复位(保留 n_obs => 不重付冷启动),
     复位 flip 在计时窗外单独记账 = 【装置钱】(同一场漂移要重复 N 次才有统计功效, 而滑窗有记忆:
     不复位 => 第 1 发就学会、之后每发 0 决策 => K 树读不到东西)。它不进配对、不进 FLIP 列。

【闸门(与 dynpair 同一批, 全绿才有资格读表)】
  ① 计数器 == route(x) 逐位; ③ 图不重录(id(_g_pre)/id(_g_hot)/_gkey/_wver 全程不变);
  ⑥ slot2eid == 权重实际内容; ⑦ 静态档【A1/A2/A3/C1】全程零迁移;
  ② 末态 slot2eid 重建引擎 => 输出逐位同(只核真迁移过的臂);
  route 放置无关性: A1/C1/H1 同一块的 eid 直方图逐位同 = 用 A1 挑块行的合法性凭据;
  🚨O1 的 flip 由探针强制 => 它不进 ⑦, 但必须过「逐块末态成员 == 该块上帝成员」这一条硬检查。

【预设判决树 K(跑前写死在 docstring 与 OPTLOG §5.2i, 跑完不许改) 求值序 K0→K1→K2→K3】
  K0 前置(全过才有资格读后三条):
     a 闸门全绿(上面六项 + O1 逐块放对);
     b 块间上帝成员两两不同 = 刺激真的存在;
     c 空对照 底_pair <= TARGET=5.0%;
     d 正对照 C1 在【每一块】都过星标 = 尺子没坏;
     e oracle O1 至少一块过星标且配对 < 1 = 这块漂移真的有可吃的上限。
     任一不过 => 挂账(印出是哪一条), 后三条不许读。
  K1 存在 dyn 臂: 决策数/发 >= 1 且 追踪率 T >= T_LO=0.60
     => 自适应真能追块内漂移、吃到上限 >=6 成 => dyn 的「自适应」第一次可记账。
  K2 有 dyn 臂开火, 但没有一臂 T >= 0.60
     => 会开火但追不上(反应延迟/冷却/max_swaps=1 把块收益吃掉了) => 判据参数要重标定,
        那是另一刀【要新的预设树】; 本轮旧表一张不动、thr_rel 不许改。
  K3 所有 dyn 臂决策数 = 0
     => 块内真漂移下从未开火 => 「自适应」主张作废, 全部旧证据重新登记为「开局一次性放置」
        => 判据必须重造(不是调参)。
  其余 => 挂账。
【冒烟记账(2026-11-23 17:32, 模型自跑 BLK=8/ROUNDS=3/POOL=2048 => 不进记录、不算预测兑现)】
  修① timed() 返 list, 帧时误当标量 append => 改 extend(与 dynpair 的 ss += s 同)。
  修② thr = max([..], TAX_MIN) 写成对 list 取 max =>  TypeError。
  修③(口径级, 不是笔误) 首烟 4 发跑完 = H1 只在【进程冷发】开火 1 次(f15 5->1), 之后计表轮
       决策全 0、成员冻在 [0,2,5] => 「重复同一场漂移 + 滑窗有记忆」会让被测策略第 1 发就学完。
       这同时回头解释了 §5.2g 那条「dyn 计表轮 468 发 0 决策」的部分成因(那条落点只用空对照组,
       E1 不受影响)。=> 新增口径 7 起跑线复位。首烟同时证明刺激造得出来: 三块上帝成员
       [0,2,1]/[0,2,5]/[0,3,2] 两两不同(K0b 可过), 捧起来的 eid 占比 23.6%/16.3% >> promote_min。
  判据只做了收紧(加 OCK 硬检查与闸门⑦ 物理零迁移), 没有放松任何一条。

【事前预测(可证伪, 跑前写死)】押 **K2**, 机制假设 = 【反应延迟吃掉块收益】:
  理由(假设不是结论, 全部来自读代码): 块边界一次要挪【2 个槽位】, 而 HYSTERESIS 是
  max_swaps=1 + cooldown=16 => 最快也要跨【两个检查点】= 16 帧才挪完; 窗口 window=16 帧
  在边界处必然混着上一块的计数 => 首个检查点(块后第 7 帧)窗口里一半以上是旧块。
  块长 BLK=24 帧 => 光挪人就付掉块的 2/3, 预期 T ≈ 0.2~0.4 = 落 K2 而不是 K1。
  竞争假设 = K3【promote_min=5% 这一条在新块上真满足(捧起来的 eid 占比预期 >20%), 所以
  押 K3 的概率低; 若真落 K3, 说明判据的瓶颈根本不在阈值而在窗口/占比的算法本身】;
  竞争假设 K1 需要 T>=0.60, 按上面的算术它要块长 >=2 倍延迟, 本轮 BLK=24 给不出来。
  🚨 赌注写明: 落 K2 否定的是【当前参数 + BLK=24 这个漂移节奏】这一对,
  不否定「自适应本身没用」—— 那要 BLK 扫描轮才读得出来(交付边界里挂的下一刀)。

【§5.2i附 磨尺子轮(开工单; r1 已入库 c299b2b9 = 落 K0 挂账, 事前预测押 K2 => 打脸)】
  修⑩(事后发现的口径缺陷, 只更正写法, **不改 r1 的 K0 落点**): 表 3 的「净收益」列印的是
     Σ_块(med A1 - med 臂), 单位是 ms/帧, 却被标成 ms/发, 又被减掉一笔【每发】flip 总 ms
     => r1 那一整列「真净收益」作废, 它当场造出「连 oracle 都亏 21.53」这种物理上不成立的账。
     正解 = 乘块长 BLK。按 r1 印出的数重算: O1 +97.8 / H1 +58.6 / H2 +11.5 ms/发,
     而空对照 A2/A3 的噪声就有 +18.5/+24.2 ms/发 => O1 的真信号约是噪声底的 5 倍。
     K0 五条前置不读这一列 => r1 落点不动。
  磨尺子两条(都在收紧方向, r1 的 K 树一个字不动):
     ① 前置 c′ = 底_pair <= TARGET=5.0% **且** 每格发数 n >= N_MIN=45 —— r1 的 6.91% 按纯抽样
        底 ∝ n^-0.5 折算, n=48 期望 ~4.8%; 先例 = §5.2g 发数标度轮 d9d54f5b(加发数把底压下去)。
        表 2 另印「折算到 n=48」一列当旁证, 防「拿小 n 蒙过前置」, 它不是判据。
     ② 新星标 ★2 = 双峰格不再一票否决, 改成【低/高两簇各自 |Δ| > 门槛 且同号】才计分;
        非双峰格与旧 ★ 逐字同。r1 有 13/21 格吃双峰(连 O1 三块全中) => 旧 ★ 下前置 e 必然过不了,
        那是尺子的问题, 不能读成「可吃上限不存在」。
  ③ ⑥' 口径 2026-11-23 r2 前补(第一棵 K 树 r1 已归档 c299b2b9, 本条只加严不放宽): 探针在 6 发
     7332 次快查里抓到 2 次【单比特瞬态】不符 —— 各差 1/884736 字节且都翻在 bit5=0x20 上,
     立刻复跑完整 check_content 即为绿, 该行不等于【任何】专家的期望行, 同行 fp32 伙伴与 scale 都对
     => 不是记账/布局缺陷(那类必然持续), 是这台机器的单比特事件。
     故 ⑥ 由「每块末核一次」改为「每块末核一次 + 红则立刻复跑一次」: 复跑仍红 = 硬红(逻辑不符),
     复跑即绿 = 瞬态, 单独计数; 【追加判据】瞬态 > 2 次/轮 亦判 ⑥ 红 = 本机判决不可信 => 挂账。
  修⑰ 落地更正 2026-11-23 r2 后(不改上面口径一个字): r2 的 ⑥ 红是【装置伪红】= 闸门谓词写的是
     「首查即绿」, 与本条口径矛盾 => 0 硬红 + 2 瞬态 被印成 1027/1029 红 => 两棵 K 树前置 a 红。
     已把谓词改成只看硬红, 瞬态仍走追加判据; r2 的 M0 属装置作废, K' 树未被真正消耗(见 OPTLOG 5.2i附)。
  【预设判决树 K′(跑前写死, 跑完不许改) 求值序 M0 -> K1′ -> K2′ -> K3′】
     M0 前置: a 闸门全绿; b 块间上帝成员两两不同; c′ 见上; d′ 正对照 C1 在每块都过 ★2;
              e′ oracle O1 至少一块过 ★2 且配对 < 1。任一不过 => 挂账, 后三条不许读。
     K1′/K2′/K3′ 三条问句与 K 树**共用同一份代码**(决策/发 >=1 且 T >= T_LO=0.60 / 开火但无一臂
     T >= 0.60 / 零开火) => 本刀换的只有尺子, 问句不许跟着尺子改。
     落 M0 挂账第二次 => 「24 帧块 + TARGET=5%」这套尺子判不可用, 下一刀换口径而不是再加发数。
  【事前预测(可证伪, 跑前写死)】押 **K1′**, 机制假设 = r1 那台 H1 的 T=0.92 主要是【真信号】:
     反应延迟中位 16 帧 < 块长 24 帧 => 每块还有约 1/3 帧数跑在正确成员上, 收益应能吃到上限 6 成以上;
     竞争假设 = K2′(r1 的空对照 A2/A3 不开火也「赚」T=0.08/0.15 => 噪声底确实垫高过分子, 尺子磨干净
     之后 H1 的 T 可能掉到 0.6 以下), 再次 = M0 挂账(48 发仍收不进 5% 底, 或 C1/O1 的 ★2 仍不过)。
     🚨 落 K1′ 也只允许写「在 BLK=24 这个漂移节奏下 dyn 追得上上限的 >=6 成」; 块长扫描与阈值扫描
     是另外两刀。thr_rel=0.05、全部旧表、swapper 默认 "sync" 一律不动(升默认 = §5.2h附3 要点头)。

  §5.2i附2 换口径轮 r4(2026-11-23 跑前写死): r3【96ecadf1】落 M0 挂账【第二次】, a/b/c′/d′ 全绿而
  唯一红项换到 e′(O1 ★2块 []/3), 两条结构死因:
     甲 = 底_pair 4.18% 本身达标, 但 底_序 8.95%→16.00% 把配对门槛 max(2x max(底_pair,底_序), TAX_MIN)
          从 17.9% 抬到 32.0% = 六倍 TARGET => 门槛不是被臂撑大的, 是被位次项撑大的;
     乙 = ★2 要求两簇【各自】过门槛, 而少数簇只有 2 发 => O1/blk1 低簇 46 发 −43.0% 已过 /
          高簇 2 发 −30.1% 差 1.9pp => 整格不计分(O1/blk2、O1/blk0 同因)。
  按 r1 跑前提示④「挂账第二次 => 这套尺判不可用, 换口径而不是第三次加发数」, 本刀【唯一自变量 =
  估计量】: 每块 p50 -> 块和(BLK 帧相加, 一格单位 ms/块)。主尺 RULER(默认 sum)决定落点; 旁证尺 =
  另一把(p50), 只用来看两把底 + 复读已消耗的两棵树 => 🚨跑完不许在两把尺里挑一颗树认。
  前置 a/b/c″/d″/e″ 与 ★2 = r3 口径逐字不动(只是底由块和算), 问句 K1″/K2″/K3″ 与 K 树共用同一份代码。
  【预设判决树 K″(跑前写死, 跑完不许改) 求值序 M0″ -> K1″ -> K2″ -> K3″】
     M0″ 前置: a 闸门全绿; b 块间上帝成员两两不同; c″ 底_pair<=TARGET=5.0% 且 n>=N_MIN=45;
               d″ C1 每块过 ★2; e″ O1 至少一块过 ★2 且配对<1。任一不过 => 挂账, 后三条不许读。
     K1″/K2″/K3″ 与 K/K′ 树**共用同一份代码**(决策/发 >=1 且 T >= T_LO=0.60 / 开火但无一臂 T>=0.60 /
     零开火) => 本刀换的只有估计量, 问句与判据数值一字未动。
  【修⑪ 单位换算(随尺子换)】表 3 净收益: 主尺=块和时一格已是 ms/块 => 乘子 1; 主尺=p50 时一格是
     ms/帧 => 仍乘 BLK(修⑩)。表 5 上限: 块和口径要 /BLK 才回到 ms/帧。乘子错一位就是 r1 那种废列。
  【事前预测(可证伪, 跑前写死)】押 **K1″**, 机制假设 = 块和把 BLK=24 帧折成一格 => 纯抽样底按
     √24≈4.9x 缩, 对位次项/瞬态打折取 3x => 底_pair/底_序 ~1.4%/~5% => 门槛 ≈10%(r3 是 32.0%)
     => e″/d″ 第一次有资格被读; 问句侧 = r2/r3 两次独立读到的 H1 T=0.86→0.84 稳 => T>=T_LO。
     竞争假设 = **K2″**(门槛过了但迁移仍追不上 BLK 帧的漂移) 与 **M0″ 第三次**(底_序 是位次系统项,
     不按 √n 缩 => 那时才谈换装置 = 真漂移语料/换机器/memtest, 🚨不许第三次加发数, 不许下调
     TARGET/T_LO/thr_rel)。
     🚨 落 K1″ 也只允许写「在 BLK=24 漂移节奏 + 块和口径下 dyn 能吃到上限 >= T_LO」。

【§5.2i附3 B1 换装置轮 r5(2026-11-24 跑前写死): 复制单位从【发】换成【进程】+ 建臂顺序随机化】
  r4【48499173】落 M0″ 挂账【第三次】, 唯一红项 c″ 底_pair 8.41% > TARGET=5.0%, 四条证据把真因指向
  【臂身份固定效应】(甲 A2/A1 三块全负且两序同号; 乙 A3/A1 同向; 丙 两台从不迁移的臂「白赚」净收益
  物理不成立; 丁 位次曲线除掉自身跨块几何均值后只剩 ±2~3.6%)。本刀【唯一自变量 = 复制单位 +
  建臂顺序】, 两件事都在装置层, 不碰任何判据数值:
     · 复制单位 发 -> 进程: 跑 NP 个【独立进程】(NP_MIN=4, 默认 NP=5), 每进程 ROUNDS=48 逐字不动。
       🚨 这不是「第四次加发数」—— 每格仍然 n=48, 新加的是【进程】这个独立重复层(r4 的丁条明说
       臂身份项「不吃发数」, 所以加发数必然仍死, 只有换重复层才有资格谈把底压下去);
     · 建臂顺序: 每进程用不同 BUILD_SEED 随机化【7 臂的建臂/分配顺序】—— 只动建臂顺序,
       不动 ARMS 语义、不动轮内正反序(位次项已由 K″ 的块和尺处理)、不动 mk() 里的 torch.manual_seed
       (=> 权重逐字同, 变的只有分配落点); BUILD_SEED 为空 = 逐字退回 r1~r4 固定建臂序 =>
       已入库四张表的复现性一字不动;
     · 装置(7 臂 x 3 块 x 24 帧)/TARGET=5.0/T_LO=0.60/N_MIN=45/BIM_THR=8%/TAX_MIN/thr_rel=0.05/
       主尺 RULER=sum 全部一字不动。新增 NP_MIN=4: 进程数不足 4 无权判中位数。
  【预设判决树 K‴(跑前写死在 docstring 与 OPTLOG §5.2i附3, 跑完不许改) 求值序 M0‴ -> K1‴ -> K2‴ -> K3‴】
     a‴ 【每个】进程闸门全绿(装置清白是逐进程的事, 不打折);
     b‴ 【每个】进程块间上帝成员两两不同 = 刺激真的存在;
     c‴ 【本刀被考核项】跨进程 底_pair = 各进程 底_pair 的【中位数】<= TARGET=5.0%,
        且【每个】进程 n >= N_MIN(=48);
     d‴ 正对照 C1: 跨进程配对(同格各进程配对的几何均值)在【每一块】都过 门槛‴
        = max(2 x max(中位底_pair, 中位底_序), TAX_MIN), 且至少 ceil(2NP/3) 个进程在自己进程内
        C1 三块全过 ★2;
     e‴ 上限锚 O1: 跨进程配对至少一块 <1 且过 门槛‴, 且至少 ceil(2NP/3) 个进程在自己进程内该块
        同样过 ★2 且配对<1。
     任一不过 => M0‴ 挂账【第四次】, 后三条问句不许读。
     K1‴/K2‴/K3‴ 与 K/K′/K″ 树**同一份问句**(决策/发 >=1 且 T >= T_LO=0.60 / 开火但无一臂 T >= T_LO /
     零开火), 只是 T 取【跨进程中位数】、决策/发 取跨进程均值。
  【事前预测(可证伪, 跑前写死)】押 **M0‴**, 机制假设 = 实例偏移属于【第 k 次分配出来的那块内存】
     而不是【A2 这个身份】: 同一条 r4 证据自己就写着 —— 建臂序在 r2/r3/r4 三轮【逐字相同】,
     而 底_pair = 0.97% / 4.18% / 8.41% 三轮三个数 => 洗牌建臂序只是把偏移在身份间重新分配, 而
     底_pair = max(|A2/A1-1|, |A3/A1-1|) 读的是【三个 iid 实例偏移的极差】, 洗牌不改变它的分布
     => 跨进程 底_pair 中位仍在 ~5~8% => c‴ 红 = M0‴。
     竞争假设 = **K1‴/K2‴**(若偏移其实【与建臂位置相关】, 例如越晚建臂越靠分配器高地址/更容易
     命中大页与对齐 => 洗牌后同一身份的偏移被跨进程平均掉 => c‴ 绿 => 三条问句今天第一次有资格
     被读; 那时 r2/r3/r4 三次独立读到的 H1 T=0.86→0.84→0.64 跨轮稳 => 落 K1‴ 的概率略高于 K2‴)。
  【K‴ 判读提示(跑前写死)】① c‴ 绿 = 本机这套尺【可用】定案 => K1‴/K2‴/K3‴ 落点第一次可以进正文;
        c‴ 红 = 偏移不吃建臂序 => 【本机「7 臂同进程轮转 + BLK=24 + TARGET=5%」装置定案不可用】
        (挂账第四次) => 下一刀只能换物理装置/语料(真漂移语料 / 换机器 / memtest)或走 C 兜底收窄主张,
        🚨 不许第五次加发数、不许下调 TARGET/T_LO/N_MIN/NP_MIN/thr_rel、不许改 ★/★2 口径、旧表一张不动;
     ② 臂身份诊断列(身份 x 进程 偏移矩阵、按建臂位置分组的空对照 Δ%、r(建臂位置,Δ))是本刀真正产出,
        但它【不进判据】: 判据只有 a‴~e‴ 与三条问句;
     ③ 每进程自己打印的 K/K′/K″ 三棵树【全部已消耗】=> 单进程落点只是旁证, 唯一新落点 = 末尾
        combine_drifttrack.py 印出的 K‴ 树; 三棵已消耗树在跨进程层不重算也不复读;
     ④ 净收益/真净收益仍只作旁证(修⑩/修⑪ 的单位口径不变: 主尺=sum => 一格 ms/块, 乘子 1);
     ⑤ 三种落点都不许动 thr_rel=0.05、不许动旧表、swapper 默认仍是 "sync"(升默认 = §5.2h附3 要点头)。
【B1 冒烟记账(2026-11-24 08:08, 模型自跑 BLK=8/ROUNDS=3/POOL=2048/NP=2 => 不进记录、不算预测兑现)】
  管路走通: 两进程建臂序真的不同[C1,O1,H2,A3,A1,H1,A2 / A3,C1,A2,H1,H2,A1,O1]、闸门各自全绿、
  p*.json 落盘、组合器不崩且 c‴ 按 NP=2 < NP_MIN=4 与 n=3 < N_MIN=45 正确判红 => 只证明【管路】;
  那一轮的 底_pair=10.96% 一类数字全部作废(块长 8 帧 / 每格 3 发, 与 r5 装置不同)。BUILD_SEED 不设时
  建臂行退回 "[建臂 A1 ] ..." 逐字旧格式 => r1~r4 四张已入库表复现性未动。

【交付边界】产出只有五种: ①块画像(每块真值直方图 + 上帝成员 + 捧了谁);
  ②逐发配对表(每块主尺值(p50 或 块和, 随 RULER)/配对/末态成员/迁移次/ms/决策); ③汇总(含两序底与双峰列);
  ④追踪率与账(T、反应延迟、净收益、flip 自身、引 22ff0967 的回本帧数对账);
  ⑤按 K 树自动印出的落点 + 事前预测兑现/打脸。
  🚨 本刀不改任何旧表、不改生产代码、不改 thr_rel、不碰裁判、`swapper` 默认仍是 "sync"。
"""
import os, sys, time, math, random, statistics, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H
import real_smollm as R
# 🚨 分簇/统计的尺子复用 乙-1c 那一份 => 本刀切出来的簇与已入库表是同一刀
from bench_pairtax import stat, bimodal, TAX_MIN, BIM_THR
from dyn_placement import (DynPlacer, counts_from_engine, slot_counts, policy_from_name,
                           POLICY_PRESETS)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, K, MODE = 42, 2, "L2_零CUDA+pinned"
LAYERS = tuple(int(v) for v in os.environ.get("LAYERS", "0,4,8,12,16,20,24,28").split(","))
GATE = os.environ.get("GATE", "embed")
CAP_LAYER = int(os.environ.get("CAP_LAYER", 8))
N_CHARS = int(os.environ.get("N_CHARS", 24000))
BS = int(os.environ.get("BS", 512))
GC = int(os.environ.get("GC", 3))
DTYPE = os.environ.get("DTYPE", "fp32").strip()
POOL = int(os.environ.get("POOL", 4096))          # 挑块行的池子行数(<= readme token 数)
BLK = int(os.environ.get("BLK", 24))              # 每块帧数 = 漂移节奏(事前预测算的是 24)
WARM_ENG = int(os.environ.get("WARM_ENG", 12))    # 段前暖机帧(在 blk0 的 x 上)
WINDOW = int(os.environ.get("WINDOW", 16))      # dyn 滑窗帧数(与 dynpair/dyn 同)
CHECK = int(os.environ.get("CHECK", 8))          # 每 CHECK 帧一个决策检查点
WARM = int(os.environ.get("WARM", 16))           # 满窗之前不许开火
E = len(LAYERS)
FULL = dict(gpu_impl="cat", graph_scope="full", n_threads=6, tok_block=8,
            pre_impl="fused", l2_stream="single", cpu_sched=1, d2h_wait="main")
ARMS = [a.strip() for a in os.environ.get("ARMS", "A1,A2,A3,C1,H1,H2,O1").split(",") if a.strip()]
ROUNDS = int(os.environ.get("ROUNDS", 25))        # 计入表的发数
WARM_ROUNDS = int(os.environ.get("WARM_ROUNDS", 1))
TARGET = float(os.environ.get("TARGET", 5.0))     # K0c: 空对照 底_pair 上限
T_LO = float(os.environ.get("T_LO", "0.60"))      # K1: 追踪率下限
PREDICT = os.environ.get("PREDICT", "K2")         # K 树事前预测(跑前写死, 见 docstring)
PREDICT2 = os.environ.get("PREDICT2", "K1")       # K′ 树(§5.2i附)事前预测; 比较时去掉 prime 号
N_MIN = int(os.environ.get("N_MIN", 45))          # K′c′: 每格最少发数, 少于它就无权判 底_pair
ONCE_MS = float(os.environ.get("ONCE_MS", "19.9"))    # 22ff0967 表5 once_main(sync) = 4.89 帧
P_MS = float(os.environ.get("P_MS", "4.6"))            # 22ff0967 表5/表6 稳态惩罚 ms/帧
# §5.2i附2 换口径轮 r4 = 本刀唯一自变量(估计量) + K″ 树事前预测; 判据数值一字未动
RULER = os.environ.get("RULER", "sum")              # 主尺估计量: sum=每块块和(BLK 帧相加) / p50=每块中位(r1~r3 旧尺)
assert RULER in ("sum", "p50"), "RULER 只能是 sum 或 p50"
SUM_RULER = RULER == "sum"
ALT_RULER = "p50" if SUM_RULER else "sum"           # 旁证尺 = 另一把(只印两把底 + 复读已消耗的 K/K′ 两棵树)
PREDICT3 = os.environ.get("PREDICT3", "K1")         # K″ 树(§5.2i附2)事前预测(跑前写死); 比较时去掉 prime 号
# §5.2i附3 B1 换装置轮 r5 = 唯一自变量(复制单位 发->进程 + 建臂顺序随机化) + K‴ 树事前预测
BUILD_SEED = os.environ.get("BUILD_SEED", "").strip()   # 非空 = 随机化建臂/分配顺序(空 = r1~r4 固定序)
PROCID = os.environ.get("PROCID", "").strip()           # 进程编号, 只用于打印与 dump
DUMP = os.environ.get("DUMP", "").strip()               # 非空 = 进程末把标量写成 JSON 给组合器
NP_MIN = int(os.environ.get("NP_MIN", 4))               # K‴c‴: 少于 4 个进程无权判中位数
PREDICT4 = os.environ.get("PREDICT4", "M0")             # K‴ 树(§5.2i附3)事前预测(跑前写死)
# 臂 -> (策略名, 起跑序, 身份)  🚨 A 组三台必须逐字同配置: 同策略名 + 同起跑序 + 都不开火
ARMSPEC = {"A1": ("static_A", "A", "null"), "A2": ("static_A", "A", "null"),
           "A3": ("static_A", "A", "null"), "C1": ("static_C", "C", "pos"),
           "H1": ("hyst_1x", "A", "dyn"), "H2": ("hyst_2x", "A", "dyn"),
           "O1": ("static_A", "A", "oracle")}
DYN = [a for a in ARMS if ARMSPEC[a][2] == "dyn"]
NULL = [a for a in ARMS if ARMSPEC[a][2] == "null"]
assert set(ARMS) <= set(ARMSPEC), f"未知臂, 可选 {sorted(ARMSPEC)}"
assert "A1" in ARMS, "A1 是配对分母, 不能摘"
assert "O1" in ARMS and DYN, "O1 是上限锚点、H* 是被测档, 都不能摘"


# ---------------------------------------------------------------- 真模型(只 load 一次; 与 bench_dynpair 逐字同)
def _load():
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"          # 🚨 这台机器 DNS->127.0.0.1, 强制离线
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        snap = R.snapshot()
        tok = AutoTokenizer.from_pretrained(snap)
        mdl = AutoModelForCausalLM.from_pretrained(snap)
        mdl.eval().to(torch.float32)
        for p in mdl.parameters():
            p.requires_grad_(False)
        return mdl, tok
    finally:
        if prev is not None:
            os.environ["HF_HUB_OFFLINE"] = prev
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)


def rss_mb():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmRSS:"):
            return int(ln.split()[1]) / 1024.0
    return float("nan")


t_load = time.perf_counter()
_MSTATE = H.machine_state()
print(f"[机器状态] {_MSTATE}")
_PROC0 = (time.strftime("%Y-%m-%d %H:%M:%S"), open("/proc/loadavg").read().split()[0])
_SD = R.state_dict()
R.state_dict = lambda: _SD
dims = R.dims(_SD)
D, N, NL = dims["D"], dims["N"], dims["layers"]
MODEL, TOK = _load()
torch.manual_seed(SEED)


def capture_from(ids):
    box = {}

    def _h(_m, inp, _o):
        box["x"] = inp[0].detach()

    hk = MODEL.model.layers[CAP_LAYER].mlp.register_forward_hook(_h)
    try:
        with torch.no_grad():
            MODEL(ids)
    finally:
        hk.remove()
    return box["x"][0].float().contiguous()


def perm_src(order):
    base = R.real_src(LAYERS, gate_mode=GATE)
    return dict(experts=[base["experts"][i] for i in order],
                gate_w=base["gate_w"][list(order)].contiguous(),
                gate_b=base["gate_b"].clone(),
                note=base["note"] + f" perm={list(order)}")


def hits_of(gate_w, rows):
    W = gate_w.float()
    cnt = torch.zeros(E)
    for i in range(0, rows.shape[0], 64):
        sc = rows[i:i + 64] @ W.t()
        cnt += torch.bincount(sc.topk(K, dim=1).indices.flatten(), minlength=E).cpu()
    return cnt


def pct(c):
    return [round(float(v) / float(sum(c)) * 100, 1) for v in (c.tolist() if hasattr(c, "tolist") else c)]


BASE = R.real_src(LAYERS, gate_mode=GATE)


def mk(gc, order, dt=DTYPE):
    """建引擎 + 上 GPU + 暖一次缓存(与 bench_dynpair.mk 逐字同 => 闸门②/⑥ 才与归档同源)。"""
    torch.manual_seed(SEED)
    m = H.HeteroMoE(D, E, N, gpu_expert_count=gc, top_k=K, expert_kind="swiglu",
                    real_src=perm_src(order), cpu_delay_ms=0.0, gpu_dtype=dt, **FULL).cuda()
    m(torch.zeros(1, D, device="cuda"), async_mode=True, mode=MODE)
    return m


def fwd(m, x):
    with torch.no_grad():
        return m(x, async_mode=True, mode=MODE)


def timed(m, x, n=1):
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = fwd(m, x)
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out, y


def phase_ids(name, kth, rows=None):
    rows = BS if rows is None else int(rows)
    txt = (R.real_text(N_CHARS) if name == "readme"
           else open(os.path.join(HERE, name), encoding="utf-8", errors="ignore").read()[:N_CHARS])
    ids = TOK(txt, return_tensors="pt").input_ids
    lo = kth * BS
    assert ids.shape[1] >= lo + rows, f"语料 {name} 只有 {ids.shape[1]} token, 取不出第 {kth} 段 {rows} 行"
    return ids[:, lo:lo + rows], len(txt), ids.shape[1]


def route_eid(m, x, order):
    ts, ti = m.route(x)
    return ts, ti, torch.tensor(list(order))[ti.detach().cpu().long()]


# ---------------------------------------------------------------- 命中画像 -> A序/C序(与 dynpair 同)
XREAL, _d = R.capture(layer=CAP_LAYER, bs=BS, n_chars=N_CHARS)
cnt_e = hits_of(BASE["gate_w"], XREAL)
order_A = torch.argsort(cnt_e, descending=True).tolist()
order_C = torch.argsort(cnt_e).tolist()
FAVRAW = os.environ.get("FAV", "").strip()
FAV = ([None] + [int(v) for v in FAVRAW.split(",") if v.strip()]) if FAVRAW else \
      [None, int(order_A[GC]), int(order_A[GC + 1])]
NBLK = len(FAV)
NF = NBLK * BLK


# ---------------------------------------------------------------- 7 臂: 一次建齐, 全程常驻, 一次都不 del
ENG = {}
BUILD_ORDER = list(ARMS)
BUILD_SEED_I = int(BUILD_SEED) if BUILD_SEED else None
if BUILD_SEED_I is not None:
    random.Random(BUILD_SEED_I).shuffle(BUILD_ORDER)
    _pos = lambda q: f"{BUILD_ORDER.index(q) + 1}/{len(BUILD_ORDER)} "
else:
    _pos = lambda q: ""
print(f"[建臂序] {'未随机(r1~r4 固定序 ' + ','.join(BUILD_ORDER) + ')' if BUILD_SEED_I is None else ','.join(BUILD_ORDER)}"
      f" | 进程={PROCID or '-'} BUILD_SEED={BUILD_SEED or '-'} | 🚨 只随机建臂/分配顺序:"
      f" ARMS 轮内正反序、mk() 里 torch.manual_seed({SEED}) 逐字未动 => 权重与 r1~r4 全同")
print(f"      🚨 本进程末尾印的 K/K′/K″ 三棵树【全部已消耗】=> 单进程落点只作旁证;"
      f" 唯一新落点 = combine_drifttrack.py 的 K‴ 树")
for a in BUILD_ORDER:
    pol_name, st_tag, kind = ARMSPEC[a]
    start = order_A if st_tag == "A" else order_C
    t0 = time.perf_counter()
    m = mk(GC, start, DTYPE)
    dyn = DynPlacer(m, BASE["experts"], GC, policy=policy_from_name(pol_name, GC),
                    window=WINDOW, check_every=CHECK, warm=WARM, order=start)
    dyn.sw.src_gate = BASE["gate_w"]      # 🚨 挂不上 = 闸门⑥ 不核门控行
    ENG[a] = dict(m=m, dyn=dyn, pol=pol_name, kind=kind, start=list(start), gk0=None, orc=[],
                  rst=[], rst_us=0.0, rst_last=(0, 0.0),
                  enabled=bool(POLICY_PRESETS.get(pol_name, {}).get("enabled", True)),
                  build_s=time.perf_counter() - t0)
    print(f"  [建臂 {_pos(a)}{a:<3}] {pol_name:<9} {kind:<6} 起跑={'A序' if st_tag == 'A' else 'C序'} "
          f"enabled={ENG[a]['enabled']} {ENG[a]['build_s']:.1f}s | RSS={rss_mb():.0f}MB "
          f"显存={torch.cuda.memory_allocated() / 1024 ** 2:.0f}MB")


# ---------------------------------------------------------------- 块: 让 GPU 最优成员在段内真的移动
def blk_pick(router, order, favor, rows):
    """从池子里挑 rows 行: favor=None => 池子最前 rows 行(= r2/r3 的 readme 段同一批挑行);
    否则挑「命中 favor 最多」的 rows 行并把行序还原(与 bench_dynpair.drift_ids 同一把尺:
    只用 route 分数、stable argsort、sel.sort() 还原行序)。"""
    if favor is None:
        return torch.arange(rows)
    _ts, _ti, eid = route_eid(router, XP, order)
    eid = eid.reshape(-1, K)
    score = (eid == int(favor)).sum(1).float()
    try:
        sel = torch.argsort(-score, stable=True)[:rows]
    except TypeError:
        sel = torch.argsort(-score)[:rows]
    return sel.sort().values


ids_pool, nch_pool, ntok_pool = phase_ids("readme", 0, POOL)
XP = capture_from(ids_pool).cuda()
BLKS = []
for b, fav in enumerate(FAV):
    sel = blk_pick(ENG["A1"]["m"], order_A, fav, BS)
    ids = ids_pool[:, sel]
    x = capture_from(ids).cuda()
    c = hits_of(BASE["gate_w"], x.cpu().float())          # 🚨 真值直方图: 只吃 gate_w, 与放置无关
    top = torch.argsort(c, descending=True).tolist()[:GC]
    BLKS.append(dict(tag=f"blk{b}", favor=(None if fav is None else int(fav)), ids=ids, x=x,
                     cnt=c, top=top, nsel=int(ids.shape[1]),
                     fav_share=(float(c[int(fav)]) / float(c.sum()) if fav is not None else float("nan"))))
del XP
torch.cuda.empty_cache()

# 闸门【route 放置无关性】: 用 A1 挑行, 就必须现场证明 route 的 eid 直方图与放置无关
GR = []
for b, B in enumerate(BLKS):
    seen = {}
    for a in [q for q in ("A1", "C1", "H1") if q in ENG]:
        _ts, _ti, eid = route_eid(ENG[a]["m"], B["x"], ENG[a]["start"])
        seen[a] = torch.bincount(eid.reshape(-1), minlength=E)
    ks = list(seen)
    for i, a in enumerate(ks):
        for c in ks[i + 1:]:
            GR.append((b, f"{a}/{c}", int(torch.equal(seen[a], seen[c]))))

print(f"\n[真源] D={D} N={N} 层数={NL} | 取 {list(LAYERS)} 层 MLP 当 {E} 个专家 top_k={K}"
      f" | 路由={GATE} | GC={GC} BS={BS} dtype={DTYPE} | 建模+建臂 {time.perf_counter() - t_load:.0f}s")
print(f"[序] A序={order_A}(GPU 成员={order_A[:GC]}) C序={order_C}(GPU 成员={order_C[:GC]})")
print(f"[段] {NBLK} 块 x {BLK} 帧 = {NF} 帧/发, 段前暖机 {WARM_ENG} 帧 | dyn: 窗口{WINDOW}"
      f" 暖机{WARM} 每{CHECK}帧一决策 cooldown=16 max_swaps=1")
print(f"[臂] {' '.join(a + '=' + ENG[a]['pol'] for a in ARMS)} | 空对照={NULL}"
      f" 被测自适应={DYN} 上限锚=O1 正对照={[a for a in ARMS if ENG[a]['kind'] == 'pos']}")
print(f"[池] readme 前 {POOL} 行(全文 {ntok_pool} token) 当挑行池; 捧的人 FAV={FAV}")
for b, B in enumerate(BLKS):
    cov = 100.0 * sum(float(B["cnt"][e]) for e in B["top"]) / float(B["cnt"].sum())
    wrong = [e for e in order_A[:GC] if e not in B["top"]]
    print(f"   · {B['tag']} 捧={str(B['favor']):<5} 占比="
          + (f"{100 * B['fav_share']:.1f}%" if B["favor"] is not None else "  - ")
          + f" 命中%={pct(B['cnt'])} 上帝成员={B['top']}(覆盖{cov:.1f}%)"
          + f" A1开局成员错在={wrong}")
print(f"[常驻] RSS={rss_mb():.0f}MB 显存={torch.cuda.memory_allocated() / 1024 ** 2:.0f}MB"
      f"({len(ARMS)} 台常驻 = 本刀故意付的代价; 🚨 不是被测系统的显存)")


# ---------------------------------------------------------------- 一发 = 一台引擎把多块段跑一遍
LATP = {a: [[] for _ in range(NBLK)] for a in ARMS}     # 旁证尺: 每块 p50, 逐轮(r1/r2/r3 用的旧尺)
LATS = {a: [[] for _ in range(NBLK)] for a in ARMS}     # 主尺: 每块块和 = 该块 BLK 帧之和(ms/块), 逐轮
LAT = LATS if SUM_RULER else LATP                       # 表 2~5 与【新】K″ 树读主尺
LAT2 = LATP if SUM_RULER else LATS                      # 🚨 已消耗的 K/K′ 树读旁证尺 = 忠实复读 r1/r3
NSW = {a: [[] for _ in range(NBLK)] for a in ARMS}      # 该块 dyn 决策次数
SWM = {a: [[] for _ in range(NBLK)] for a in ARMS}      # 该块 flip ms(含探针强制的)
MEM = {a: [[] for _ in range(NBLK)] for a in ARMS}      # 该块末态 GPU 成员
DLY = {a: [[] for _ in range(NBLK)] for a in ARMS}      # 该块反应延迟(帧; BLK+1 = 未到位)
FLIP = {a: [] for a in ARMS}                            # 每发 flip 总 ms(含强制)
G1, G3, G6, G2, OCK = [], [], [], [], []                # 闸门明细
TRACE, TAGS, POS = {}, [], {}
SHOT_NO = {a: 0 for a in ARMS}
fm = lambda m: "[" + ",".join(str(int(e)) for e in m) + "]"


def force_to(st, top):
    """O1 探针强制: 把 GPU 成员换成 top(集合)。付的是与 dyn 同一笔 flip(sync 生产路),
    同一个 sw.total_us, 也在计时窗外 —— 它买到的是【零反应延迟】, 不是白送迁移。"""
    dyn = st["dyn"]
    cur = list(dyn.sw.slot2eid)
    miss = [s for s in range(GC) if cur[s] not in top]
    want = [e for e in top if e not in cur[:GC]]
    done = []
    for s, eid in zip(miss, want):
        j = next(q for q in range(len(cur)) if cur[q] == eid)
        rec = dyn.sw.swap(s, j)
        cur = list(dyn.sw.slot2eid)
        done.append(rec)
    st["orc"] += done
    return done


def reset_state(st):
    """每发【起跑线对齐】= 本刀的测量装置钱, 不是被测策略的钱:
      ① GPU 成员回到起跑序(slot2eid 逐位对齐 => 每发起点逐字同 = 空对照组才成立的前提);
      ② dyn 臂复位滑窗 + 冷却 + 禁令 => 每发都是「最优成员刚移动的那一场漂移」的第一次。
    🚨 为什么必须复位: 同一场漂移要重复 N 次才有统计功效, 而滑窗/成员态是有记忆的 =>
       不复位 = 第 1 发就学会, 之后每发 0 决策。§5.2g 那条「dyn 计表轮 468 发 0 决策」
       有一部分正是这个成因(它不影响 E1 落点, 那条只用空对照组)。
    🚨 保留 n_obs: 生产里引擎早就跑热了, 不该在每发重付冷启动暖机帧。
    复位 flip 落在计时窗外、单独记账(不进 FLIP 也不进配对)。"""
    dyn = st["dyn"]
    n, us0 = 0, dyn.sw.total_us
    tgt, cur = list(st["start"])[:GC], list(dyn.sw.slot2eid)   # 🚨 只对齐 GPU 那 GC 个槽
    for _ in range(2 * len(tgt)):                      # 有界: 每轮至少摆正一个槽
        bad = next((q for q in range(GC) if cur[q] != tgt[q]), None)
        if bad is None:
            break
        want = tgt[bad]
        q = next((k for k in range(GC, len(cur)) if cur[k] == want), None)
        if q is None:                                  # want 还在别的 GPU 槽 => 先顶到 CPU
            r = next(k for k in range(GC) if cur[k] == want)
            junk = next(k for k in range(GC, len(cur)) if cur[k] not in tgt)
            dyn.sw.swap(r, junk)
            cur = list(dyn.sw.slot2eid)
            q = next(k for k in range(GC, len(cur)) if cur[k] == want)
        dyn.sw.swap(bad, q)
        cur = list(dyn.sw.slot2eid)
        n += 1
    assert list(dyn.sw.slot2eid[:GC]) == list(tgt), f"复位没把成员摆回起跑序: {cur[:GC]} != {tgt}"
    if st["kind"] == "dyn":
        dyn.win.reset()
        dyn.pol.last_frame = -10 ** 9
        dyn.pol.banned = {}
        dyn.pol.hist.clear()
    st["rst"] += [n]
    st["rst_us"] += dyn.sw.total_us - us0
    st["rst_last"] = (n, dyn.sw.total_us - us0)


def shot(a, tag, keep):
    st = ENG[a]
    m, dyn = st["m"], st["dyn"]
    dyn.pol.enabled = st["enabled"]              # 🚨 每发重挂: 静态档一旦被点亮就是对照组污染
    out = []
    for _ in range(WARM_ENG):
        fwd(m, BLKS[0]["x"])
    torch.cuda.synchronize()
    gk = (id(m._g_pre), id(m._g_hot), m._gkey, tuple(m._wver or ()))
    if st["gk0"] is None:
        st["gk0"] = gk
    G3.append((a, tag, gk == st["gk0"]))
    reset_state(st)
    f_us0 = dyn.sw.total_us
    orc0 = len(st["orc"])
    for b, B in enumerate(BLKS):
        s_n, s_us, s_rs = dyn.pol.n_swap, dyn.sw.total_us, len(dyn.pol.reasons)
        if st["kind"] == "oracle":
            force_to(st, B["top"])
        states = [list(dyn.sw.slot2eid[:GC])]     # 块边界上的成员态(计时窗外, 免费)
        ss = []
        for f in range(BLK):
            s, _y = timed(m, B["x"], 1)
            ss.extend(s)
            if f == 0:                             # 块首帧兼计数器闸门(与 dynpair 同一发法)
                c = counts_from_engine(m, E)
                _ts, ti = m.route(B["x"])
                G1.append((a, b, tag, int(torch.equal(c, slot_counts(ti.detach().cpu(), E))),
                           int(c.sum())))
                dyn.step(b * BLK + f, b, counts=c)
            else:
                dyn.step(b * BLK + f, b)
            states.append(list(dyn.sw.slot2eid[:GC]))   # 🚨 迁移在计时窗口【外】
        tgt = set(int(e) for e in B["top"])
        hit = next((i for i, s_ in enumerate(states) if set(int(e) for e in s_) == tgt), BLK + 1)
        b1 = dyn.sw.check_content()
        b2 = dyn.sw.check_content() if b1 else []   # ⑥' 红则立刻复跑: 分「逻辑硬红」与「单比特瞬态」
        G6.append((a, b, tag, b1, not b2, len(b2)))
        p50 = statistics.median(ss)
        bsum = sum(ss)                                   # §5.2i附2 主尺 = 块和(单位 ms/块, 一格 = BLK 帧)
        memb = list(dyn.sw.hot_members())
        nsw = dyn.pol.n_swap - s_n
        sms = (dyn.sw.total_us - s_us) / 1e3
        dec = [" ".join(r.split()[:2]) for r in dyn.pol.reasons[s_rs:]]
        out.append((bsum if SUM_RULER else p50, memb, nsw, sms, dec, hit))
        if keep:
            LATP[a][b].append(p50); LATS[a][b].append(bsum)
            NSW[a][b].append(nsw); SWM[a][b].append(sms)
            MEM[a][b].append(memb); DLY[a][b].append(hit)
        if b >= 1:
            TRACE.setdefault((a, b), []).append((tag, hit, dec, memb))
    FLIP[a].append((dyn.sw.total_us - f_us0) / 1e3)
    if st["kind"] == "oracle":
        OCK.append((a, tag, len(st["orc"]) - orc0))
    SHOT_NO[a] += 1
    return out


def run_round(r, keep):
    seq = list(ARMS) if r % 2 == 0 else list(reversed(ARMS))
    order_tag = "fwd" if r % 2 == 0 else "rev"
    tag = f"#{r}{'正' if r % 2 == 0 else '反'}{'表' if keep else '暖'}"
    if keep:
        TAGS.append(order_tag)
    t0, got = time.perf_counter(), {}
    for pos, a in enumerate(seq):
        got[a] = shot(a, tag, keep)
        if keep:
            for b in range(NBLK):
                POS.setdefault((pos, b), []).append((a, got[a][b][0], order_tag))
    for pos, a in enumerate(seq):
        rat = [got[a][b][0] / got["A1"][b][0] for b in range(NBLK)]
        decs = [d for b in range(NBLK) for d in got[a][b][4]]
        print(f"  [{tag} {a:<3} pos{pos + 1}] {'块和' if SUM_RULER else 'p50'} " + "/".join(f"{got[a][b][0]:7.3f}" for b in range(NBLK))
              + " | 配A1 " + "/".join(f"{100 * (v - 1):+6.1f}%" for v in rat)
              + " | mem " + "/".join(fm(got[a][b][1]) for b in range(NBLK))
              + " | 迁 " + "/".join(str(got[a][b][2]) for b in range(NBLK))
              + "次 " + "/".join(f"{got[a][b][3]:.1f}" for b in range(NBLK))
              + "ms | 反 " + "/".join(f"{got[a][b][5]:>{2 if BLK < 10 else 3}}" for b in range(NBLK))
              + (f" | 复位 {ENG[a]['rst_last'][0]}次/{ENG[a]['rst_last'][1] / 1e3:.1f}ms"
                 if ENG[a]["rst_last"][0] else "")
              + (" | " + ";".join(decs) if decs else ""))
    print(f"  [{tag}] {(time.perf_counter() - t0):.0f}s | RSS={rss_mb():.0f}MB "
          f"loadavg={open('/proc/loadavg').read().split()[0]}")


for r in range(WARM_ROUNDS):
    print(f"\n---- 暖轮 {r}(不进表; 只为了让 CUDA 图/分配器/滑窗进入稳态) ----")
    run_round(r, False)
for r in range(WARM_ROUNDS, WARM_ROUNDS + ROUNDS):
    print(f"\n---- 第 {r} 轮({'正序' if r % 2 == 0 else '反序'}, 进表) ----")
    run_round(r, True)

# ---------------------------------------------------------------- 闸门② 末态 slot2eid 重建引擎
for a in ARMS:
    dyn, m = ENG[a]["dyn"], ENG[a]["m"]
    if not dyn.sw.n_swap:
        continue                                   # 没迁移过 = 没资格(不是跳过)
    last_order = list(dyn.sw.slot2eid)
    m2 = mk(GC, last_order, DTYPE)
    for _ in range(WARM_ENG):
        fwd(m2, BLKS[-1]["x"])
    yb = fwd(m2, BLKS[-1]["x"]).float().cpu().clone()
    ya = fwd(m, BLKS[-1]["x"]).float().cpu().clone()
    G2.append((a, last_order, int(torch.equal(ya, yb)), (ya - yb).abs().max().item()))
    del m2, yb, ya
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- 配对数学(与 bench_dynpair 同一把尺)
# 🚨 §5.2i附2 换口径轮: 配对数学包成纯函数 analyze(SRC), 同一份跑轮数据同时喂【两把尺】——
#    主尺 = RULER(默认 块和) 决定落点, 旁证尺 = 另一把, 只印两把底 + 复读已消耗的 K/K′ 两棵树。
#    🚨 落点跑前就定死只消费主尺 => 跑完不许在两把尺里挑一颗树认。
def geo(v):
    return math.exp(sum(math.log(float(x)) for x in v) / len(v))


def _geo(v):
    v = [q for q in v if q == q]
    return geo(v) if v else float("nan")


def split2(v):
    """双峰格的【成员划分】: 与 bench_pairtax.bimodal 同一把尺(最大相邻间隔, 两侧各 >=2 发,
    间隔 > BIM_THR), 但返回的是两簇的下标而不是簇心 => ★2 才能对每簇单独算配对。"""
    sv = [round(float(t), 3) for t in v]
    order = sorted(range(len(sv)), key=lambda k: sv[k])
    sv = [sv[k] for k in order]
    if len(sv) < 6 or sv[0] <= 0:
        return None
    cand = [k for k in range(len(sv) - 1) if 2 <= k + 1 <= len(sv) - 2]
    if not cand:
        return None
    k = max(cand, key=lambda q: sv[q + 1] / sv[q])
    if (sv[k + 1] / sv[k] - 1) * 100 <= BIM_THR:
        return None
    return [order[:k + 1], order[k + 1:]]


def _g(xs):
    xs = list(xs)
    return geo(xs) if xs else float("nan")


nP = NBLK
FWD = [n for n, t in enumerate(TAGS) if t == "fwd"]
REV = [n for n, t in enumerate(TAGS) if t == "rev"]
ALL_IDX = list(range(len(TAGS)))
NULL2 = [a for a in NULL if a != "A1"]
REAL = [a for a in ARMS if ENG[a]["kind"] != "null"]
# 修⑪(§5.2i附2): 单位随尺子换 —— p50 尺一格 = 1 帧的 ms(→ ms/发 要乘 BLK),
# 块和尺一格 = BLK 帧的 ms(本身就是【一块】的 ms, 乘子 1; 表 5 的 ms/帧 要 ÷ BLK)。
MS_PER_UNIT = "ms/块" if SUM_RULER else "ms/帧"
TO_FRAME = 1 if SUM_RULER else BLK
PEN_DIV = BLK if SUM_RULER else 1


def analyze(SRC):
    """一把尺的逐发数据 -> 全套配对/星标/追踪率。纯函数: 两把尺各算一份, 互不污染。"""
    def pairs(idx):
        if not idx:
            return {a: [float("nan")] * nP for a in ARMS}
        return {a: [geo([SRC[a][k][q] / SRC["A1"][k][q] for q in idx]) for k in range(nP)] for a in ARMS}

    PAL, PFW, PRV = pairs(ALL_IDX), pairs(FWD), pairs(REV)
    GS = {a: [_geo(SRC[a][k]) for k in range(nP)] for a in ARMS}
    底_pair = max([abs(PAL[a][k] - 1) * 100 for a in NULL2 for k in range(nP)] or [float("nan")])
    底_序 = (max(abs(PFW[a][k] / PRV[a][k] - 1) * 100 for a in NULL for k in range(nP))
             if FWD and REV else float("nan"))
    BAD = {(a, k): bimodal(SRC[a][k]) for a in ARMS for k in range(nP) if len(SRC[a][k]) >= 6}
    BAD = {kk: v for kk, v in BAD.items() if v}
    _fs = [v for v in (底_pair, 底_序) if v == v]
    thr = max((2.0 * max(_fs)) if _fs else 0.0, TAX_MIN)
    STAR = {(a, k): (a in REAL and (a, k) not in BAD and abs(PAL[a][k] - 1) * 100 > thr)
            for a in ARMS for k in range(nP)}
    CLU = {}                                             # 双峰格 -> [(簇名, 发数, 该簇配对)]
    for _a in ARMS:
        for _k in range(nP):
            _sp = split2(SRC[_a][_k]) if (_a, _k) in BAD else None
            if _sp:
                CLU[(_a, _k)] = [("低", len(_sp[0]), _g(SRC[_a][_k][q] / SRC["A1"][_k][q] for q in _sp[0])),
                                 ("高", len(_sp[1]), _g(SRC[_a][_k][q] / SRC["A1"][_k][q] for q in _sp[1]))]
    # ★2(§5.2i附 磨尺子轮写死): 非双峰格与旧 ★ 逐字同; 双峰格不再一票否决, 改成
    # 【低/高两簇各自 |Δ| > 门槛 且同号】才计分 —— 撤了否决但加了「两簇同向」这道更硬的闸。
    STAR2 = {}
    for _a in ARMS:
        for _k in range(nP):
            _v = SRC[_a][_k]
            if _a not in REAL or len(_v) < 6:
                STAR2[(_a, _k)] = False
            elif (_a, _k) not in BAD:
                STAR2[(_a, _k)] = abs(PAL[_a][_k] - 1) * 100 > thr
            else:
                _cl = CLU.get((_a, _k), [])
                STAR2[(_a, _k)] = (len(_cl) == 2 and all(c == c and abs(c - 1.0) * 100 > thr for _, _, c in _cl)
                                   and len({1 if c > 1 else -1 for _, _, c in _cl}) == 1)
    NETP = {a: (sum(stat(SRC["A1"][k])[0] - stat(SRC[a][k])[0] for k in range(nP))
                if (SRC[a][0] and SRC["A1"][0]) else float("nan"))
            for a in ARMS}                               # Σ_块 每格省下的 ms(单位见 MS_PER_UNIT)
    NETG = {a: (NETP[a] * TO_FRAME if NETP[a] == NETP[a] else float("nan")) for a in ARMS}   # 净收益 ms/发
    GAIN = {a: _geo([PAL[a][k] for k in range(nP)]) for a in ARMS}
    GO = GAIN.get("O1", float("nan"))
    T_R = {a: ((1.0 - GAIN[a]) / (1.0 - GO) if GO == GO and abs(1.0 - GO) > 1e-9 else float("nan"))
           for a in ARMS}
    T_K = {a: [((1.0 - PAL[a][k]) / (1.0 - PAL["O1"][k]) if abs(1.0 - PAL["O1"][k]) > 1e-9 else float("nan"))
               for k in range(nP)] for a in ARMS}
    nn = min([len(SRC[a][0]) for a in ARMS if SRC[a][0]] or [0])
    DZ = 底_pair * math.sqrt(nn / 48.0) if 底_pair == 底_pair else float("nan")
    return dict(PAL=PAL, PFW=PFW, PRV=PRV, GS=GS, 底_pair=底_pair, 底_序=底_序, BAD=BAD, thr=thr,
                STAR=STAR, CLU=CLU, STAR2=STAR2, NETP=NETP, NETG=NETG, GAIN=GAIN, GO=GO,
                T_R=T_R, T_K=T_K, DZ=DZ, nn=nn)


MAIN = analyze(LAT)      # 主尺(RULER) => 唯一能决定落点的一把
ALT = analyze(LAT2)      # 旁证尺(ALT_RULER) => 只印两把底 + 复读已消耗的两棵树
_TREE_KEYS = ["PAL", "PFW", "PRV", "GS", "底_pair", "底_序", "BAD", "thr", "STAR", "CLU",
              "STAR2", "NETP", "NETG", "GAIN", "GO", "T_R", "T_K", "DZ", "nn"]
globals().update(MAIN)   # 表 2~5 与【新】K″ 树读主尺; 两棵已消耗的树在打印前临时换回旁证尺
nR = {a: len(LAT[a][0]) for a in ARMS}
DTOT = {a: sum(sum(x) for x in NSW[a]) for a in ARMS}                  # 计表轮决策总次数
DEC = {a: (DTOT[a] / nR[a] if nR[a] else 0.0) for a in ARMS}           # 决策次数/发
DFS = {a: (sum(sum(x) for x in SWM[a]) / nR[a] if nR[a] else 0.0) for a in ARMS}   # 块内自记 flip ms/发
FLPM = {a: (statistics.median(FLIP[a]) if FLIP[a] else 0.0) for a in ARMS}          # 每发 flip 总 ms
DLB = {a: [d for k in range(1, nP) for d in DLY[a][k]] for a in ARMS}   # 反应延迟只看漂移块 b>=1
UNH = {a: sum(1 for d in DLB[a] if d > BLK) for a in ARMS}
ORCF = {a: (statistics.median([q[2] for q in OCK if q[0] == a]) if any(q[0] == a for q in OCK) else 0.0)
        for a in ARMS}                                                  # O1 每发被强制 flip 几次

# ----------------------------------------------- O1 硬检查: 每块【末态成员】必须 == 该块上帝成员
OCK2 = []
for _a in [q for q in ARMS if ENG[q]["kind"] == "oracle"]:
    for _k in range(nP):
        _t = set(int(e) for e in BLKS[_k]["top"])
        for _j in range(len(MEM[_a][_k])):
            OCK2.append((_a, _k, set(int(e) for e in MEM[_a][_k][_j]) == _t))


def _gate(name, rows, ok, note=""):
    bad = [r for r in rows if not ok(r)]
    mark = "OK" if not bad and rows else ("🚨" if bad else "没资格")
    print(f"  {name:<30} {len(rows) - len(bad)}/{len(rows)} {mark}"
          + (f" —— 不符 {bad[:5]}" if bad else "") + (f"  [{note}]" if note else ""))
    return not bad and bool(rows)


GT = {}
GT["①"] = _gate("闸门① 计数器==route 逐位", G1, lambda r: r[3] == 1 and r[4] > 0,
                 "块首帧兼核, counts_from_engine vs route(x), nsum>0")
GT["③"] = _gate("闸门③ 图不重录", G3, lambda r: r[2], "id(_g_pre)/id(_g_hot)/_gkey/_wver 全程不变")
TR6 = [r for r in G6 if r[3]]
TR_SOFT = [r for r in TR6 if r[4]]
TR_HARD = [r for r in TR6 if not r[4]]
# 修⑰ 2026-11-23 r2 后: ⑥' 的口径 r2 前就写死在 docstring 第③条 = 「复跑仍红=硬红判红,
# 复跑即绿=瞬态只单独计数, 追加判据 瞬态>2 格/轮 才判 ⑥ 红」—— 但下面这行谓词漏改:
# `not r[3]` 仍要求【首查即绿】=> 复跑即绿的瞬态格被记成不符 = 装置伪红。
# r2 实证: 0 硬红 + 2 瞬态 却印成 1027/1029 🚨 => 旧 K 树与 K' 树的前置 a 双双红 => 落 M0。
# 本修只是把实现拉回跑前口径(不是放宽判据): 谓词只看 r[4] = 连查后仍红 = 硬红;
# 瞬态仍受下一行【追加判据】约束, 一个字没动。
GT["⑥"] = _gate("闸门⑥ slot2eid==内容[⑥' 连查]", G6, lambda r: r[4],
                 "每块末核一次 + 红则立刻复跑; 谓词只查硬红, 瞬态走追加判据 账在下一行")
print(f"  ⑥' 复跑账[修⑰]: 共查 {len(G6)} 格, 首查红 {len(TR6)} 格 => 硬红 {len(TR_HARD)} {TR_HARD[:2]} "
      f"| 瞬态(复跑即绿, 只当机器健康量看) {len(TR_SOFT)} {TR_SOFT[:2]} "
      f"| 追加判据 瞬态 <= 2 格/轮 | 瞬态密度 {len(TR_SOFT)}/{len(G6)}")
if len(TR_SOFT) > 2:
    GT["⑥"] = False
    print("  🚨 闸门⑥ 追加判据红: 单比特瞬态 > 2 格 => 本机判决不可信, ⑥ 按红处理(挂账方向)")
GT["GR"] = _gate("闸门 route 放置无关性", GR, lambda r: r[2] == 1, "A1/C1/H1 同块 eid 直方图逐位同 => 用 A1 挑块行合法")
ST0 = [(a, ENG[a]["dyn"].sw.n_swap) for a in ARMS if not ENG[a]["enabled"] and ENG[a]["kind"] == "null"]
GT["⑦"] = _gate("闸门⑦ 静态档零迁移", ST0, lambda r: r[1] == 0,
                "空对照组物理零迁移(复位也不碰它们); C1/O1 单独记账, 不进这条")
GT["OCK"] = _gate("闸门 O1 逐块放对", OCK2, lambda r: r[2],
                  "O1 每块末态成员 == 该块上帝成员 = 上限锚点真的锚住了, 否则 K0e 读不到东西")
if G2:
    GT["②"] = _gate("闸门② 末态重建逐位", G2, lambda r: r[2] == 1, "按运行时 slot2eid 重造引擎再对输出")
else:
    GT["②"] = False
    print("  闸门② 末态重建逐位            0/0 没资格 —— 本轮【没有任何臂迁移过】, 不是跳过")
GATES_OK = all(GT.values())

# ---------------------------------------------------------------- 表 2 汇总
print("\n" + "=" * 158)
print(f"表 2 汇总: 配对 = 【同一轮同一块】内 臂/A1 再跨轮取几何 | 正序/反序各算一次 = 位次底"
      f" | 门槛={thr:.1f}% (=max(2xmax(底_pair,底_序), TAX_MIN={TAX_MIN}%)) | ★=真实档过门槛 ✗=双峰不可判[旧 ★ 口径; K′ 树改读 ★2 两簇同向]"
      f" | 🚨主尺估计量 RULER={RULER} => 该表所有延迟/配对/门槛都是【{MS_PER_UNIT}】口径")
print("=" * 158)
print(f"{'臂':<4}{'块':>5}{'发':>4}{('块和中位' if SUM_RULER else 'p50中位'):>10}{'σ%':>6}{'min':>9}{'配对(全)':>10}{'配对(正)':>10}"
      f"{'配对(反)':>10}{'Δ%':>8}{'决策/发':>9}{'flip ms/发':>11}{'反延迟中位':>10}{'末态成员':>13}  判读")
for a in ARMS:
    for k in range(nP):
        v = LAT[a][k]
        if not v:
            continue
        md, sd, mn = stat(v)
        b = BAD.get((a, k))
        tag = ("✗双峰" + f"(低{b[1]}≈{b[0]:.3f}/高{b[3]}≈{b[2]:.3f} 隔{b[4]:.0f}%)") if b else \
              ("没查(发<6)" if len(v) < 6 else ("★过门槛" if STAR[(a, k)] else "带内"))
        if (a, k) in CLU:                                            # ★2: 双峰格改读两簇, 不再一票否决
            tag += (" |★2两簇Δ=" + "/".join(f"{cn}{cc}发{100 * (cr - 1):+.1f}%"
                                            for cn, cc, cr in CLU[(a, k)])
                    + ("计分" if STAR2[(a, k)] else "不计分"))
        dly = DLY[a][k]
        memb = sorted(set(tuple(x) for x in MEM[a][k]))
        print(f"{a + '/' + BLKS[k]['tag']:<9}{k:>3}{len(v):>4}{md:>10.3f}{sd:>6.1f}{mn:>9.3f}"
              f"{PAL[a][k]:>10.4f}{PFW[a][k]:>10.4f}{PRV[a][k]:>10.4f}{100 * (PAL[a][k] - 1):>+8.1f}"
              f"{DTOT[a] / len(v):>9.2f}{DFS[a]:>11.2f}{statistics.median(dly):>10.0f}"
              f"{str(memb):>13}  {tag}")
_nn = min([nR[a] for a in ARMS if nR[a]] or [0])
_dz = 底_pair * math.sqrt(_nn / 48.0) if 底_pair == 底_pair else float("nan")
print(f"\n  底_pair(空对照组 A2/A3 对 A1, 各块取最大) = {底_pair:.2f}%   门槛 TARGET={TARGET}%"
      f"   每格发数 n={_nn}   折算到 n=48 = {_dz:.2f}%(旁证: 纯抽样 底 ∝ n^-0.5, 不是判据)")
print(f"  底_序  (空对照组+A1, 正序轮/反序轮 配对之差取最大) = {底_序:.2f}%   正序轮{len(FWD)}发 反序轮{len(REV)}发")
print(f"  两把尺对照[§5.2i附2 换口径轮唯一自变量] = 主尺 {RULER}({MS_PER_UNIT}) 底_pair={MAIN['底_pair']:.2f}%"
      f" 底_序={MAIN['底_序']:.2f}% => 门槛 {MAIN['thr']:.1f}% | 旁证尺 {ALT_RULER}"
      f"{'(ms/帧)' if ALT_RULER == 'p50' else '(ms/块)'} 底_pair={ALT['底_pair']:.2f}%"
      f" 底_序={ALT['底_序']:.2f}% => 门槛 {ALT['thr']:.1f}% | 🚨落点只消费主尺, 跑完不许两把尺挑一颗树认")
print("  逐臂迁移总账: 计表轮(只数 dyn 决策) " + " | ".join(f"{a} {DTOT[a]}次/{DFS[a] * nR[a]:.0f}ms" for a in ARMS))
print("            全程物理迁移(sw.total_us, 含暖轮与进程冷发) "
      + " | ".join(f"{a} {ENG[a]['dyn'].sw.n_swap}次/{ENG[a]['dyn'].sw.total_us / 1e3:.0f}ms" for a in ARMS))
print("            装置钱 = 每发起跑线复位 flip(计时窗外, 不进配对也不进 FLIP): "
      + " | ".join(f"{a} 复位{len(ENG[a]['rst'])}次/{sum(ENG[a]['rst'])}个/{ENG[a]['rst_us'] / 1e3:.0f}ms"
                   f"/O1强制{sum(q[2] for q in OCK if q[0] == a)}次" for a in ARMS))
print("\n位次曲线(每台先除掉【自己】的跨块几何均值 => 身份被扣干净, 剩下的只可能是位次; 单位=偏离%)")
for k in range(nP):
    for ot, lab in (("fwd", "正序"), ("rev", "反序")):
        cells = []
        for pos in range(len(ARMS)):
            vals = [v / GS[a][k] for (a, v, t) in POS.get((pos, k), []) if t == ot]
            who = ARMS[pos] if ot == "fwd" else ARMS[len(ARMS) - 1 - pos]
            cells.append(f"pos{pos + 1}{who}={100 * (geo(vals) - 1):+5.1f}" if vals else f"pos{pos + 1}{who}=  n/a")
        print(f"   块{k} {BLKS[k]['tag']:<6} {lab} " + " ".join(cells))

# ---------------------------------------------------------------- 表 3 追踪率与账
print("\n" + "=" * 158)
print(f"表 3 追踪率与账: GAIN = 各块配对比值再跨块取几何 | 收益 = 1-GAIN | T = 收益(臂)/收益(O1)"
      f" | T=1 吃满上限 0 白开火 <0 比不放还慢 | 上限锚 O1 自己那一行是 T 的分母")
print("=" * 158)
print(f"{'臂':<10}{'身份':<8}{'决策/发':>9}{'flip ms/发':>11}{'强制flip/发':>12}{'反延迟中位':>11}"
      f"{'反延迟最差':>11}{'未到位次':>10}{'GAIN':>9}{'Δ%':>8}{'T':>8}{'净收益ms/发':>12}{'真净收益':>10}")
for a in ARMS:
    dl = DLB[a]
    print(f"{a + '=' + ENG[a]['pol']:<18}{DEC[a]:>9.2f}{DFS[a]:>11.2f}{ORCF[a]:>12.1f}"
          + (f"{statistics.median(dl):>11.0f}{max(dl):>11.0f}{UNH[a]:>6}/{len(dl):<3}" if dl else f"{'-':>11}{'-':>11}{'n/a':>10}")
          + f"{GAIN[a]:>9.4f}{100 * (GAIN[a] - 1):>+8.1f}"
          + (f"{T_R[a]:>8.2f}" if T_R[a] == T_R[a] else f"{'-':>8}")
          + f"{NETG[a]:>12.2f}{NETG[a] - FLPM[a]:>10.2f}")
print(f"  🚨 净收益 ms/发 = Σ_块(med A1 - med 臂) x {'1' if SUM_RULER else 'BLK=' + str(BLK)}"
      f"[主尺={RULER} 一格单位 {MS_PER_UNIT}; 修⑩: r1 那列没乘 BLK 整列作废 / 修⑪: 换口径后乘子随尺子换"
      f" 别再乘错] | flip 落在帧与帧之间 => 该尺几乎不沾它, 所以要【从外面】减一笔 => "
      f"真净收益 = 净收益 - 每发 flip 总 ms 中位(含 O1 强制的; 起跑线复位那笔装置钱不在其中)。")
print(f"  反应延迟 = 块边界到「GPU 成员集合 == 该块上帝成员」的帧数(逐帧采样 slot2eid, 免费);"
      f" {BLK + 1} = 整块没到位, 只数漂移块(b>=1)。")

print("\n表 4 分块追踪率 T_k = (1-配对(臂,k))/(1-配对(O1,k)) —— 读【哪一块追上了】而不是只读总账")
print(f"{'臂':<10}" + "".join(f"{BLKS[k]['tag'] + ' T':>11}{BLKS[k]['tag'] + ' 反延迟':>11}" for k in range(nP)))
for a in DYN + ["O1"]:
    row = []
    for k in range(nP):
        dk = DLY[a][k]
        row.append((f"{T_K[a][k]:>11.2f}" if T_K[a][k] == T_K[a][k] else f"{'-':>11}")
                   + f"{statistics.median(dk):>11.0f}")
    print(f"{a:<10}" + "".join(row))
print("  🚨 blk0 的 T 不参与判读(那块开局成员本来就对, 分子分母都接近 0 => 比值是噪声放大)。")

print("\n表 5 回本对账(引 §5.2h附2 归档 22ff0967, 本刀不重测价签): 每块的【可吃上限】与【换一次的价签】")
print(f"{'块':<7}{'med A1':>9}{'med O1':>9}{'上限ms/帧':>11}{'回本帧(sync)':>13}{'回本帧(async)':>14}{'块长':>6}{'数学可达?':>11}")
for k in range(1, nP):
    pen = (stat(LAT["A1"][k])[0] - stat(LAT["O1"][k])[0]) / PEN_DIV   # -> ms/帧(块和尺先 ÷ 块长)
    be1 = ONCE_MS / pen if pen > 1e-9 else float("inf")
    be2 = (ONCE_MS * 0.22) / pen if pen > 1e-9 else float("inf")
    print(f"{BLKS[k]['tag']:<7}{stat(LAT['A1'][k])[0]:>9.3f}{stat(LAT['O1'][k])[0]:>9.3f}{pen:>11.3f}"
          f"{be1:>13.1f}{be2:>14.1f}{BLK:>6}"
          + (f"{'是' if be1 <= BLK else '否':>10}"))
print(f"  🚨 上限ms/帧 = (med A1 - med O1)/{PEN_DIV} —— 主尺 RULER={RULER} 的一格是 {MS_PER_UNIT}, "
      f"块和口径必须除以块长才与 22ff0967 的 ms/帧 价签同单位。")
print(f"  价签来自 22ff0967 表 5: once(sync)={ONCE_MS}ms/次 = 4.89 帧, async_staged once≈{ONCE_MS * 0.22:.1f}ms"
      f"(ratio_once=0.22); 该刀稳态惩罚 P≈{P_MS}ms/帧 与本轮上限是两回事, 别混。")

# ---------------------------------------------------------------- K0 前置 + K 树落点
# 🚨 下面两棵树(K 树 r1 c299b2b9 / K′ 树 r3 96ecadf1)都【已经消耗过】=> 只复读不产生新落点,
#    故临时把全局换回【旁证尺】= r1~r3 当时真正用来落点的 p50 尺, 才谈得上忠实复现。
globals().update({k: ALT[k] for k in _TREE_KEYS})
TOPS = [frozenset(int(e) for e in BLKS[k]["top"]) for k in range(nP)]
blocks_diff = len(set(TOPS)) == nP
c1_ok = ("C1" in ENG) and all(STAR.get(("C1", k), False) for k in range(nP))
o1_ok = any(STAR.get(("O1", k), False) and PAL["O1"][k] < 1.0 for k in range(nP)) if "O1" in ENG else False
K0 = [("a 闸门全绿(含 O1 逐块放对)", GATES_OK, f"绿={[q for q in GT if GT[q]]} 红={[q for q in GT if not GT[q]]}"),
      ("b 块间上帝成员两两不同", blocks_diff, f"各块上帝成员={[sorted(t) for t in TOPS]}"),
      ("c 空对照 底_pair <= TARGET", (底_pair == 底_pair) and 底_pair <= TARGET,
       f"底_pair={底_pair:.2f}% TARGET={TARGET}%"),
      ("d 正对照 C1 每块过星标", c1_ok, f"过星标块={[k for k in range(nP) if STAR.get(('C1', k), False)]}/{nP}"),
      ("e O1 至少一块过星标且配对<1", o1_ok,
       f"O1 配对={[round(PAL['O1'][k], 4) for k in range(nP)]} 过星标={[k for k in range(nP) if STAR.get(('O1', k), False)]}")]


def verdict():
    bad = [f"{n} => {note}" for n, ok, note in K0 if not ok]
    if bad:
        return "K0", "前置不过 => 挂账, 后三条不许读: " + " ; ".join(bad)
    return _k123()


def _k123(prf="K"):
    """K1/K2/K3 三条问句 = K 树与 K′ 树共用同一份代码 => 磨尺子只换前置, 不许顺手改问句。"""
    tracked = [a for a in DYN if DEC[a] >= 1.0 and T_R[a] == T_R[a] and T_R[a] >= T_LO]
    if tracked:
        a = max(tracked, key=lambda q: T_R[q])
        return prf + "1", (f"{a} 决策/发={DEC[a]:.2f} >=1 且 T={T_R[a]:.2f} >= T_LO={T_LO}"
                      f"(反应延迟中位={statistics.median(DLB[a]):.0f}帧) => 自适应真能追块内漂移、"
                      f"吃到上限 >= {T_LO:.0%} => dyn 的「自适应」第一次可记账")
    firing = [a for a in DYN if DTOT[a] > 0]
    if firing:
        a = max(firing, key=lambda q: T_R[q] if T_R[q] == T_R[q] else -9.0)
        return prf + "2", (f"开火的 dyn 臂={firing}(决策/发 " + "/".join(f"{DEC[q]:.2f}" for q in firing)
                      + f") 但无一臂 T >= {T_LO}(最好 {a}: T="
                      + (f"{T_R[a]:.2f}" if T_R[a] == T_R[a] else "nan")
                      + f", 反应延迟中位={statistics.median(DLB[a]):.0f}/{BLK}帧, 未到位 {UNH[a]}/{len(DLB[a])})"
                      + f" => 会开火但追不上 => 判据参数[cooldown/max_swaps/window/BLK]要重标定, 那是另一刀")
    return prf + "3", ("块内真漂移下所有 dyn 臂决策数 = 0(检查点 "
                  + f"{max(1, BLK // CHECK)} 个/块 x {nP - 1} 个漂移块 x {sum(1 for _ in TAGS)} 发 = 零开火)"
                  + f" => 「自适应」主张作废, 全部旧证据重新登记为「开局一次性放置」=> 判据要重造不是调参")


V, why = verdict()
print("\n" + "=" * 158)
print("§5.2i 预设判决树 K 落点(判据跑前写死在 docstring 与 OPTLOG 5.2i, 跑完不许改) —— 求值序 K0 -> K1 -> K2 -> K3")
print("      🚨 第一棵 K 树已在 r1 消耗 = 落 K0 挂账 归档 c299b2b9 => 本节【只复读】; r2/r3 又死掉两次,")
print("      🚨 r4 真正落点是最下面的 K″ 树。本节及下一节一律用【旁证尺 p50】复算 = 忠实复现当时口径。")
print("=" * 158)
for n, ok, note in K0:
    print(f"  {'OK ' if ok else '🚨 '} {n:<34} {note}")
print(f"  门槛: T_LO={T_LO} TARGET={TARGET}% 块长 BLK={BLK} 帧 窗口{WINDOW} 每{CHECK}帧一检查点 暖机{WARM} 帧")
print(f"  旁证(不参与落点): C1 配对={[round(PAL['C1'][k], 4) for k in range(nP)]}"
      f" | O1 配对={[round(PAL['O1'][k], 4) for k in range(nP)]} 强制flip/发={ORCF.get('O1', 0):.1f}"
      f" | 双峰格={sorted(BAD)}")
print(f"  => 落点: {V}    理由: {why}")
print(f"     事前预测(跑前写死) = {PREDICT}(机制假设 = 反应延迟吃掉块收益: 块边界要挪 2 槽而 cooldown=16"
      f" max_swaps=1 => 最快跨两个检查点; 窗口 {WINDOW} 帧在边界处混旧块) => "
      + ("兑现" if V == PREDICT else "打脸(如实记账)"))
print("""
判读提示: ① T 的分母是 O1 的收益 = 「零反应延迟但要付同一笔 flip」=> T 低只说明【追得慢】, 不说明【上限不存在】;
        ② O1 不白送迁移: 它每块付的是生产 sync 路那笔 flip(同一个 sw.total_us), 只是探针知道答案 => 它是上界不是天堂;
        ③ 反应延迟与 T 是两条独立证据: 延迟大而 T 高 = 块太长(节奏问题); 延迟小而 T 低 = 判据换错了人(阈值问题);
        ④ K0 任一不过 => 后三条不许读。尤其是 b(上帝成员两两不同)不过 = 刺激根本没造出来,
           那时「没开火」绝不能读成「自适应没用」;
        ⑤ 表 2 的双峰格一律不可判(同一把 BIM_THR=8%, 要 >=6 发), 但 K3 读的是【决策计数器】不是延迟列 => 双峰不影响 K3;
        ⑥ C1 是正对照(已知约 2.6x): 它不过星标 = 这把尺子坏了, 本刀一切负读都作废(K0d 已把它写进前置);
        ⑦ 落 K1 才允许把「自适应」写进主张; 落 K2 只允许说「会开火但当前参数追不上 BLK 帧的漂移」;
           落 K3 则要回头把所有 dyn 旧表重登记为「开局一次性放置」—— 三种落点都不许动 thr_rel 与旧表。""")

# ==================== §5.2i附 K′ 树 = 已在 r3 消耗(落 M0) => 只复读; r4 起用旁证尺 p50 复算
print("\n" + "=" * 158)
print("§5.2i附 磨尺子轮 K′ 树【只复读】(判据跑前写死在 docstring 与 OPTLOG §5.2i附, 跑完不许改)"
      " —— 求值序 M0 -> K1′ -> K2′ -> K3′; 🚨已在 r3 96ecadf1 消耗 = 落 M0 挂账【第二次】")
print("=" * 158)
print("      🚨 换的只有【尺子】= 前置 c′/d′/e′ + 新星标 ★2; K1/K2/K3 三条问句与 K 树共用同一份代码,")
print("      🚨 问句不许跟着尺子改(上一棵 K 树死于前置, 问句从未被读过 => 重读它是本刀唯一目的);")
print("      🚨 净收益列 修⑩ 之后才是 ms/发 —— r1 那一列没乘块长 BLK, 它印的「真净收益」整列作废。")
print("      🚨 a 条的 ⑥ 已在 r2 前改成 ⑥'(连查两次 + 瞬态 > 2 格判红) —— 单比特事件档案见 OPTLOG §5.2i附")


def _st2(a):
    return [k for k in range(nP) if STAR2.get((a, k), False)]


c_ok2 = (底_pair == 底_pair) and _nn >= N_MIN and 底_pair <= TARGET
c1_ok2 = ("C1" in ENG) and all(STAR2.get(("C1", k), False) for k in range(nP))
o1_ok2 = (any(STAR2.get(("O1", k), False) and PAL["O1"][k] < 1.0 for k in range(nP))
          if "O1" in ENG else False)
M0 = [("a 闸门全绿(含 O1 逐块放对)", GATES_OK,
       f"绿={[q for q in GT if GT[q]]} 红={[q for q in GT if not GT[q]]}"),
      ("b 块间上帝成员两两不同", blocks_diff, f"各块上帝成员={[sorted(t) for t in TOPS]}"),
      ("c′ 底_pair<=TARGET 且 n>=N_MIN", c_ok2,
       f"底_pair={底_pair:.2f}% TARGET={TARGET}% n={_nn} N_MIN={N_MIN} 折算n48={DZ:.2f}%"),
      ("d′ 正对照 C1 每块过 ★2", c1_ok2,
       f"C1 ★2块={_st2('C1')}/{nP} 配对={[round(PAL['C1'][k], 4) for k in range(nP)]}"),
      ("e′ O1 至少一块过 ★2 且配对<1", o1_ok2,
       f"O1 ★2块={_st2('O1')}/{nP} 配对={[round(PAL['O1'][k], 4) for k in range(nP)]}")]
print("  ★2 明细(只印双峰格; 两簇【各自】过门槛且同号才计分):")
for _a in ARMS:
    for _k in range(nP):
        if (_a, _k) in CLU:
            print("    " + f"{_a}/blk{_k}: " + " ".join(
                f"{cn}簇{cc}发配对{cr:.4f}({100 * (cr - 1):+.1f}%)" for cn, cc, cr in CLU[(_a, _k)])
                + (" => ★2计分" if STAR2[(_a, _k)] else " => ★2不计分"))
for n, ok, note in M0:
    print(f"  {'OK ' if ok else '🚨 '} {n:<32} {note}")
print(f"  门槛: T_LO={T_LO} TARGET={TARGET}% N_MIN={N_MIN} 块长 BLK={BLK} 帧 窗口{WINDOW}"
      f" 每{CHECK}帧一检查点 暖机{WARM} 帧")
print("  旁证(不参与落点): 净收益 ms/发 " + " ".join(f"{a}={NETG[a]:.1f}" for a in ARMS)
      + " | 真净收益 ms/发 " + " ".join(f"{a}={NETG[a] - FLPM[a]:.1f}" for a in ARMS))
V2, why2 = _k123("K′")
_bad2 = [f"{n} => {note}" for n, ok, note in M0 if not ok]
if _bad2:
    V2, why2 = "M0", "尺子前置不过 => 挂账, 三条问句不许读: " + " ; ".join(_bad2)
print(f"  => 落点: {V2}    理由: {why2}")
print(f"     事前预测(跑前写死) = {PREDICT2}(机制假设 = r1 的 H1 T=0.92 主要是真信号: 反应延迟中位 16 帧"
      f" < 设计块长 24 帧, 块内还有约 1/3 帧数跑在正确成员上 => T 应 >= 0.6; 竞争假设 = K2′【r1 空对照"
      f" 不开火也 T=0.08/0.15 => 噪声底垫高过分子, 尺子磨干净后 H1 可能掉到 T_LO={T_LO} 以下】"
      f" 与 M0 挂账; 本轮 BLK={BLK} n={_nn}) => "
      + ("兑现" if V2.replace("′", "") == PREDICT2 else "打脸(如实记账)"))
print("""
K′ 判读提示: ① ★2 只解决「双峰格一票否决」这一件事: 两簇同向且各自过门槛 => 该格的【方向】可信,
              配对绝对值仍带簇心偏差 => T 读它, 净收益列只作旁证;
           ② c′ 的 n>=N_MIN 是给「加发数」这条修法上的锁: 发数不够时 底_pair 偏高 = 尺子没磨完,
              不许读成「本机噪声就是这么大」, 更不许反过来下调 TARGET;
           ③ 落 K1′ 只允许写「在 BLK=24 这个漂移节奏下 dyn 能吃到上限的 >=6 成」; 块长扫描(节奏)
              与阈值扫描(换人)是两刀, 都要各自的预设树;
           ④ 落 M0 挂账第二次 => 本机「24 帧块 + TARGET=5%」这套尺子判【不可用】, 要换口径
              (例如把每块 p50 换成整段墙钟差), 而不是第三次加发数;
           ⑤ 三种落点都不许动 thr_rel=0.05、不许动旧表、swapper 默认仍是 "sync"
              (升默认 = §5.2h附3, 要用户点头)。""")

# ============================ §5.2i附2 K″ 树 = 本轮真正的落点(判据跑前写死, 跑完不许改)
globals().update({k: MAIN[k] for k in _TREE_KEYS})   # 换回【主尺】= RULER(默认 块和)
print("\n" + "=" * 158)
print("§5.2i附2 换口径轮 K″ 树落点(判据跑前写死在 docstring 与 OPTLOG §5.2i附2, 跑完不许改)"
      " —— 求值序 M0″ -> K1″ -> K2″ -> K3″")
print("=" * 158)
print(f"      🚨 本刀唯一自变量 = 【估计量】: 每块 p50 -> 块和({BLK} 帧相加), 主尺 RULER={RULER}"
      f" 一格单位 {MS_PER_UNIT};")
print(f"      🚨 判据数值一字不改(TARGET={TARGET}% / N_MIN={N_MIN} / T_LO={T_LO} / TAX_MIN={TAX_MIN}% /"
      f" BIM_THR={BIM_THR}% / ★2 两簇各自过门槛且同号), 装置(7 臂 x {NBLK} 块 x {BLK} 帧)也一字不改;")
print("      🚨 上面两棵已消耗的树(K r1 c299b2b9 / K′ r3 96ecadf1)用【旁证尺 p50】复读, 不产生新落点;")
print("      🚨 落点只消费主尺 => 跑完不许在两把尺里挑一颗树认(两把底已在表 2 末行同时打印)。")
print("  ★2 明细(主尺; 只印双峰格, 两簇【各自】过门槛且同号才计分):")
for _a in ARMS:
    for _k in range(nP):
        if (_a, _k) in CLU:
            print("    " + f"{_a}/blk{_k}: " + " ".join(
                f"{cn}簇{cc}发配对{cr:.4f}({100 * (cr - 1):+.1f}%)" for cn, cc, cr in CLU[(_a, _k)])
                + (" => ★2计分" if STAR2[(_a, _k)] else " => ★2不计分"))
c_ok3 = (底_pair == 底_pair) and _nn >= N_MIN and 底_pair <= TARGET
c1_ok3 = ("C1" in ENG) and all(STAR2.get(("C1", k), False) for k in range(nP))
o1_ok3 = (any(STAR2.get(("O1", k), False) and PAL["O1"][k] < 1.0 for k in range(nP))
          if "O1" in ENG else False)
M03 = [("a 闸门全绿(含 O1 逐块放对)", GATES_OK,
        f"绿={[q for q in GT if GT[q]]} 红={[q for q in GT if not GT[q]]}"),
       ("b 块间上帝成员两两不同", blocks_diff, f"各块上帝成员={[sorted(t) for t in TOPS]}"),
       ("c″ 底_pair<=TARGET 且 n>=N_MIN", c_ok3,
        f"底_pair={底_pair:.2f}% TARGET={TARGET}% n={_nn} N_MIN={N_MIN} 折算n48={DZ:.2f}%"),
       ("d″ 正对照 C1 每块过 ★2", c1_ok3,
        f"C1 ★2块={_st2('C1')}/{nP} 配对={[round(PAL['C1'][k], 4) for k in range(nP)]}"),
       ("e″ O1 至少一块过 ★2 且配对<1", o1_ok3,
        f"O1 ★2块={_st2('O1')}/{nP} 配对={[round(PAL['O1'][k], 4) for k in range(nP)]}")]
for n, ok, note in M03:
    print(f"  {'OK ' if ok else '🚨 '} {n:<32} {note}")
print(f"  门槛: T_LO={T_LO} TARGET={TARGET}% N_MIN={N_MIN} 块长 BLK={BLK} 帧 窗口{WINDOW}"
      f" 每{CHECK}帧一检查点 暖机{WARM} 帧 | 主尺={RULER}")
print("  旁证(不参与落点): 净收益 ms/发 " + " ".join(f"{a}={NETG[a]:.1f}" for a in ARMS)
      + " | 真净收益 ms/发 " + " ".join(f"{a}={NETG[a] - FLPM[a]:.1f}" for a in ARMS)
      + " | 旁证尺 p50 底_pair/底_序="
      + f"{ALT['底_pair']:.2f}%/{ALT['底_序']:.2f}%")
V3, why3 = _k123("K″")
_bad3 = [f"{n} => {note}" for n, ok, note in M03 if not ok]
if _bad3:
    V3, why3 = "M0″", "前置不过 => 挂账, 三条问句不许读: " + " ; ".join(_bad3)
print(f"  => 落点: {V3}    理由: {why3}")
print(f"     事前预测(跑前写死) = {PREDICT3}(机制假设 = 块和把 {BLK} 帧折成一格 => 纯抽样底按 √{BLK}≈"
      f"{math.sqrt(BLK):.1f}x 缩、对位次/瞬态打折取 3x => 底_pair/底_序 ~1.4%/~5% => 门槛 ≈10%"
      f"(r3 是 32.0%) => e″/d″ 第一次有资格被读; 问句侧 r2/r3 两次读到的 H1 T=0.86→0.84 稳 => T>=T_LO;"
      f" 竞争假设 = K2″【门槛过了但迁移仍追不上】与 M0″ 第三次【底_序 是位次系统项不按 √n 缩】;"
      f" 本轮 BLK={BLK} n={_nn} RULER={RULER}) => "
      + ("兑现" if V3.replace("″", "").replace("′", "") == PREDICT3 else "打脸(如实记账)"))
print("""
K″ 判读提示: ① 块和把【整块 BLK 帧的墙钟】折成一格 => 迁移帧的尖峰被摊进整块和里, 双峰大概率变少,
              但它【不消除】位次项: 若 ★2 仍不过, 那是位次系统项不是抽样, 别再用「再跑一轮」赌;
           ② 表 2 末行的【两把底对照】= 本刀唯一要读的前置旁证: 主尺底不降 = 换口径无效 = 落 M0″;
           ③ 落 K1″ 只允许写「在 BLK=%d 这个漂移节奏 + 块和口径下 dyn 能吃到上限 >= T_LO」;
              配对比值与净收益的绝对值仍带簇心偏差 => 只作旁证, 不进正文主张;
           ④ 落 M0″ = 挂账【第三次】=> 本机这套尺判【不可用】定案, 下一刀换装置(真漂移语料 / 换机器 /
              memtest), 🚨不许第三次加发数、不许下调 TARGET/T_LO/thr_rel;
           ⑤ 三种落点都不许动 thr_rel=0.05、不许动旧表、swapper 默认仍是 "sync"
              (升默认 = §5.2h附3, 要用户点头)。""" % BLK)

print("[耗时] 建模+建臂+跑轮共 {:.0f}s | RSS={:.0f}MB 显存={:.0f}MB".format(
    time.perf_counter() - t_load, rss_mb(), torch.cuda.memory_allocated() / 1024 ** 2))

# ------------------------------------ §5.2i附3 B1 换装置轮: 进程级 dump(只写标量, 判据仍在本文件里算)
if DUMP:
    import json
    payload = dict(
        ver=1, proc=PROCID or "p?", seed=BUILD_SEED, build_order=list(BUILD_ORDER),
        head=dict(mstate=_MSTATE, date=_PROC0[0], loadavg=_PROC0[1]),
        cfg=dict(TARGET=TARGET, T_LO=T_LO, N_MIN=N_MIN, NP_MIN=NP_MIN, TAX_MIN=TAX_MIN,
                 BIM_THR=BIM_THR, RULER=RULER, BLK=BLK, BS=BS, GC=GC, NBLK=NBLK,
                 ARMS=list(ARMS), MS_PER_UNIT=MS_PER_UNIT, PREDICT4=PREDICT4,
                 KIND={a: ARMSPEC[a][2] for a in ARMS},
                 POL={a: ARMSPEC[a][0] for a in ARMS}),
        gates=dict(ok=bool(GATES_OK), gt={k: bool(v) for k, v in GT.items()},
                   blocks_diff=bool(blocks_diff), tops=[sorted(t) for t in TOPS],
                   ock_all_ok=(all(q[2] for q in OCK2) if OCK2 else False)),
        main=dict(bp=MAIN["底_pair"], bs=MAIN["底_序"], thr=MAIN["thr"], nn=MAIN["nn"],
                  PAL={a: [float(q) for q in MAIN["PAL"][a]] for a in ARMS},
                  STAR2={a: [bool(MAIN["STAR2"].get((a, k), False)) for k in range(nP)] for a in ARMS},
                  STAR={a: [bool(MAIN["STAR"].get((a, k), False)) for k in range(nP)] for a in ARMS},
                  GAIN={a: MAIN["GAIN"][a] for a in ARMS}, T_R={a: MAIN["T_R"][a] for a in ARMS},
                  T_K={a: [float(q) for q in MAIN["T_K"][a]] for a in ARMS},
                  DEC={a: DEC[a] for a in ARMS}, DECT={a: DTOT[a] for a in ARMS},
                  FLPM={a: FLPM[a] for a in ARMS}, NETG={a: NETG[a] for a in ARMS},
                  DLBmed={a: (float(statistics.median(DLB[a])) if DLB[a] else float("nan"))
                          for a in ARMS},
                  UNH={a: UNH[a] for a in ARMS}, nDLY={a: len(DLB[a]) for a in ARMS}),
        alt=dict(bp=ALT["底_pair"], bs=ALT["底_序"], thr=ALT["thr"]),
        wall_s=time.perf_counter() - t_load, rss_mb=rss_mb(),
        loadavg_end=open("/proc/loadavg").read().split()[0])
    with open(DUMP, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"[dump] {DUMP} | 主尺 {RULER} 底_pair={MAIN['底_pair']:.2f}% 底_序={MAIN['底_序']:.2f}%"
          f" 门槛={MAIN['thr']:.1f}% n={MAIN['nn']} 闸门={'全绿' if GATES_OK else '有红'}"
          f" 建臂序={','.join(BUILD_ORDER)}")
