#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1-②附3 · 迁移价签探针: 同一副字节走 sync 与 AsyncSwapper 两条路, 「换一次 ~15ms = 3~4 帧」这张价签该不该重贴。

【靶子不是双峰, 是价签】
  dyn_placement.py:71 与 :264 两处白纸黑字写着「换一次迁移真花 ~15ms = 3~4 帧」, 而 PRICE 里那个
  thr_rel=0.05 的开关点就是按它标定的 => 价签错 = 滞回开关点错 = P1-②b/②c 两张 price_* 表的输入错。
  🚨 这张价签的来源是 sync 版 swap() 的【墙钟】, 而 sync 那 15ms 由三样混成, 该不该付是三个问题:
    (1) 前后各一个 torch.cuda.synchronize() = device-wide 等上一帧排空;
    (2) 约 6 次【pageable 主机】H2D copy_ = CUDA 内部先 sync 再边拷边发;
    (3) CPU 腿 .t().contiguous() + 3 次 requant = 纯 CPU 计算(不是设备同步)。
  AsyncSwapper(= 外部建议的"选项 B": 后台搬字节 + 版本号一次性切换)把 (1) 去掉、(2) 换成
  pinned + 侧 stream、(3) 换成按专家预存的表。本刀只问两句: 换完之后的【价签】是多少,
  省下来的钱是不是原来记在账上的那两样。
  🚨 顺带把 §5.2i 的牙装上: F2(切换成本吃光自适应收益)成不成立, 直接由这张价签决定。

【臂(全部常驻、全程不 del; 三台同一起跑序 order_A、同一批 x、同一串翻转脚本)】
  sync            生产路: DynPlacer 默认 swapper="sync" => ExpertSwapper.swap() 逐字不变(本轮只加了
                  5 个 perf_counter 时间戳, 换权重语句一字未动) —— 它就是价签的分母
  async_direct    AsyncSwapper, 不预告: flip 里当场 plan(H2D) 再提交 D2D => H2D 藏不进上一帧
  async_staged    AsyncSwapper, 提前一整帧 plan: 侧 stream 的 H2D 与那一帧的计算重叠, flip 只提交 D2D

【口径(五条, 别看错)】
  1) 🚨 flip 调用自己的 µs 【不是】价签。sync 的账落在 flip 的 host stall 上, async 的账落在
     【下一帧的设备时间线】上(host 早返回了) —— 只读一类会系统性偏袒 async, 所以价签读墙钟差:
        价签(帧/次) = (swap 段墙钟 - base 段墙钟) / p50 帧长 / 翻转次数
  2) 两种口径都算, 判据取【两者里较大的那个】tag_main = max(tag_lat, tag_cont)(防挑列):
        lat  = 逐帧前后各 sync 一次(与 dyn/dynpair 同一把帧长尺)
        cont = 整段只在末尾 sync 一次 = 真 decode 流水线(前台只提交, 不等设备)
     🚨 lat 口径下 flip 前已经 sync 过 => sync 臂的 front_sync≈0, 所以 G4 只可能从 cont 口径出现。
  3) 帧长尺子只有一把 = base 段 lat 口径的 p50, lat/cont 两种口径共用 => 三臂的价签可横比。
  4) 翻转脚本 = 【同一对槽位 (0, GC) 打一个来回】(换出再换回), 所以计时窗结束时 GPU 成员回到基线。
     这是本轮相对上一版草稿的更正: 原来用互不相交的槽位对 (k, GC+k), 成员在窗内永久改变,
     于是"帧长差"里混进"换了成员之后稳态帧长本来就变" —— 那笔钱不该记在 flip 机制头上。
  5) 每轮 base/swap 的先后按轮次奇偶互换(防进程内时间爬升单侧污染, 与 dynpair 奇偶正反序同理);
     每臂每次计时段前先 reset_prof(), 把建臂/暖段/红线趟的分相全部排除。

【闸门 = 四条红线(全部在【非计时的专用趟】里查)】
  🚨 为什么不能塞进计时段: check_content() 自己就是 D2H 同步, 塞进 cont 段 = 把被测的异步性测没。
  R1 每次 flip 后 check_content() 必须返回空表(记账 == 五处布局的实际内容, 逐位);
  R2 _versions()/_ptrs() 从建臂到收工全程不变(= 没重录图、没换缓冲地址);
  R3 sync 引擎与两台 async 引擎跑【同一串翻转】=> 逐帧输出 md5 必须逐位相同(跨实现等价性);
  R4 torn: 发一帧不 sync -> 立刻 flip -> 再 sync => 那一帧的输出必须等于 flip 前的参照
     (前提 = FULL 里 d2h_wait="main" 使 CPU 腿在 fwd() 返回前已跑完; 换 "worker" 必须重开这条。
      🚨 它的牙主要咬在"侧 stream 是否越界写 live 缓冲": 一旦有人把 D2D 挪到侧 stream 就红。)

【预设判决树(跑前写死, 跑完不许改)】求值序 = G3 -> G1 -> G2 -> G4 -> 其余挂账
  ratio = tag_main(两台 async 里较优的那台) / tag_main(sync)
  G3) 任一红线不绿 => 实现坏掉了 => 🚨【不得用"快了"记账】, 先修实现再来;
  G1) ratio <= 1/3 且四红线全绿 => 价签按本表实测重贴, thr_rel 的开关点按新价签重标定;
  G2) 1/3 < ratio <= 1/2 => 瓶颈不在这两处(同步/pageable/量化), 回分相表重找, 本刀不落价签;
  G4) ratio > 1/2 且 sync 分相表里 front_sync + tail_sync >= 50% total
      => 钱本来就主要花在"等 GPU 排空"上而不是搬字节 => 异步化治不了它, 价签口径要改成
         【尾巴 + flip】(设备时间线), §5.2i 的事前预测 F2 随之复活;
  其余 => 挂账(如实记"这一刀没能把价签重贴出来", 不改任何旧表、不改 thr_rel)。
【事前预测(可证伪, 跑前写死)】押 G1。理由(假设不是结论):
  ① flip 的量是 3 张 fp32(576x1536) = 10.6MB 级 + 门控行 2.3KB, pinned H2D 约 5~6GB/s => ~2ms 量级,
     而一整帧是 4~6ms => 提前一整帧 plan 理论上能把 H2D 整段藏进计算里;
  ② D2D(侧 stream 到 live 的那几张)在卡内, 10.6MB 级 -> 0.1ms 量级, 比 pageable H2D 小一个数量级;
  ③ sync 那 15ms 里 pageable copy_ 与 requant 是【可消除同步源/纯 CPU 计算】, 本实现两条都消了。
  竞争假设 = G4(钱主要在前后的 device-wide 排空上): 它若成立, 押 G1 就地打脸, 按 G4 落点写。
【§5.2h附 预设树(判据跑前写死, 跑完不许改) —— 问「表 3 那笔钱落在【哪几帧】」】
  r1(判决轮 6eabc1de, 2026-11-23)实测: sync 的 flip 自身 host stall 15.9ms, 但墙钟价签 tag_main=19.30 帧/次
  = 75ms/次 => 🚨约 4/5 的钱不在 flip 调用里; 两台 async 把 flip 自己压到 2.9ms 却只把价签压到 0.74 倍。
  口径 4 只保证【窗末】成员回到基线, 没保证【窗内】成员不变 => 两次 flip 之间那 21 帧是在「最热 eid 被换成
  第 4 热」的稳态里跑的 => 那笔【稳态放置惩罚】本不该记在 flip 机制头上, 但表 3 把它整笔记成了切换成本。
  工具 = lat 口径逐帧墙钟(r1 已在收, 本附不新增测量): excess[f] = med(swap 段第 f 帧) - med(base 段第 f 帧)。
  四个窗(落点只读 sync 臂 = 生产路/价签分母, 另两臂只作旁证):
    W0  = 第一次 flip 之前        -> C  【段内负对照: 成员与 base 逐字同, 期望 ~0】
    W1h/W2h = 每次 flip 后头 HEAD_F=6 帧 -> H【恢复窗: 重建派生缓存之类的一次性代价】
    W1t = 翻转态去掉头之后的剩余  -> M  【稳态: 成员变了 => 每帧都贵, 与 flip 次数无关】
    W2t = 第二次 flip(成员已换回基线)去掉头之后的剩余 -> P【残留: 换回来也不消散的那部分】
  I0) 前置: 四红线全绿 且 T>0 且 C/T <= 0.15 且 逐帧口径 (T + flip 自身)/次 与表 3 tag_lat 折成的 ms 差 <= 25%
      => 否则表 4 整表不可读(口径没对上 或 段内对照脏了);
      🚨冒烟记账(2026-11-23, 判决轮未跑 => 不算改判据): 原写作 T/次, 冒烟实测 sync 差 39% 而两台 async 只差几个点 ——
         查因 = lat 口径的逐帧计时夹在 flip 【外面】, flip 那 15.8ms(sync)落在帧与帧之间、一个 per 帧都不进,
         所以 T 恒比表 3 少一个 flip 自身(sync ~15.8ms、async ~2.9ms)。这是本探针的口径漏项, 不是系统性质,
         故对账前先补上这一笔; 份额分母仍只用 T(四个帧窗), 理由 = 本树问的是 flip【之外】的钱在哪几帧。
  I1) H/T >= 0.60 且 P/T <= 0.15 => 钱 = 一次性恢复成本 => 价签【可以】重贴成「N 帧/次」(N = T/次/帧长),
      thr_rel 按它重标定, 「async 只省 flip 自身那 13ms」也就解释清楚了;
  I2) M/T >= 0.40 => 钱 = 【翻转态的稳态放置惩罚】被记成了切换成本 => 🚨表 3 的价签口径本身作废:
      它喂给 dyn 的是「每次 flip 摊 N 帧」, 真实形状是「成员换了之后每帧都贵」=> thr_rel 不许按本表重标定;
  I3) P/T >= 0.30 => 换回基线后仍然慢 => 代价与【驻留时长】成正比、不随 flip 次数摊薄 => 口径同样要改;
  其余 => 挂账。求值序 = I0 -> I1 -> I2 -> I3 -> 其余挂账。
【§5.2h附 事前预测(可证伪, 跑前写死)】押 I2。理由(假设不是结论):
  ① 表 1/表 2 已把 flip 自身钉死在 sync 15.8ms / async 2.9ms, 而价签是它的 4~5 倍 => 大头必然在 flip 之外;
  ② 槽位对是 (0, GC) = 最热 eid 换第 4 热 => W1 那 21 帧的流量结构变了, 按主表口径「热专家下到 CPU = 每帧
     多付一整条腿」, 21 帧 x 毫秒级 = 与那笔差额同量级;
  ③ 探针【故意】选最贵那对槽位, 而生产里 dyn 只在收益 > thr_rel 时才换 => 稳态项在生产里应当是【负】的
     => 若 I2 兑现, 结论是「探针口径要改」而不是「生产切换这么贵」。
  竞争假设 = I1(flip 后重建派生缓存的一次性恢复成本) => 它若成立, 押 I2 就地打脸, 价签反而可以直接重贴。

【§5.2h附2 预设树(判据跑前写死, 跑完不许改) —— 问「剥掉稳态项之后, 一次性价签到底是几帧」】
  r2(判决轮 0e305a89, 2026-11-23)落 I2: sync 那笔墙钟价签的 81% 是【翻转态稳态惩罚】(M/T=0.81) => 表 3 作废。
  但作废的是【口径】, 不是价签本身: dyn 每次真换成员, 除了「之后每帧都贵」还要付一次 flip 自身 + flip 后那
  6 帧的恢复窗。生产里 dyn 只在【收益 > thr_rel】时才换 => 稳态项在真决策里符号是反的 => 喂给滞回开关点的
  必须是【一次性】那一笔, 而它恰恰被 r2 的墙钟差淹没了。本附把稳态项从【设计】上抵消, 只量这一笔。
  工具 = 同成员对照(每轮加发两段, 段长/帧序/翻转脚本全不变, 只把开场成员换成翻转态并暖 J_WARM=12 帧):
    base  = S0 稳态, 窗内不翻转            swap  = S0 开场, 窗内翻 2 次 => 中途进 S1 再回 S0
    basef = S1 稳态, 窗内不翻转            swapf = S1 开场, 窗内翻 2 次 => 中途回 S0 再进 S1
  于是 tag = 一次性 + 整窗稳态惩罚, tagf = 一次性 - 整窗稳态惩罚 => (tag+tagf)/2 = 一次性,
  (tag-tagf)/2 = 稳态项(拿它和表 4 的 M/次 互校 = J0 的自洽检查)。
  🚨 s1 开场那次成员设定、以及 s1 段内的 flip, 都【不进】表 1/表 2/表 4 与设备 Event(方向相反, 混进去
     就是脏分母, 由 do_flip(book=) 挡), 只进「迁移总账」那一行。
  J0) 前置: 四红线全绿 且 once(sync) > 0 且 抵消自洽 —— 表 5 稳态项 (tag-tagf)/2 x 帧长(ms/次) 与 表 4 的
      M/次 之差 <= J0_TOL=0.50(表 5 数整个 21 帧窗, 表 4 的 M 去掉了头 6 帧 => 只要求同量级)
      => 否则稳态项不是常数(爬坡)或口径又漂了, 一次性价签读不出;
  J1) once_main(sync) <= 1.0 帧/次 => 旧价签「~15ms = 3~4 帧」高估 >= 3 倍 => 允许下调 thr_rel=0.05;
  J2) 1.0 < once_main(sync) <= 6.0 帧/次 => 与旧价签同量级 => 旧表一张不动, 但必须写明「一次性」与「稳态」
      是两笔账、滞回开关点该读前者;
  J3) once_main(sync) > 6.0 帧/次 => 换一次成员 >= 6 帧 => 生产里几乎永远不该换 => 滞回阈值要大幅提高;
  其余 => 挂账。求值序 = J0 -> J1 -> J2 -> J3 -> 其余挂账。
  追加判据(只追加结论, 不改落点): ratio_once = once_main(最优 async)/once_main(sync) <= 0.60 才允许写
      「异步化在【一次性价签】上省 >= 40%」; 表 3 那个 ratio(0.74/0.78)含稳态项, 不算这一笔。
      🚨冒烟记账(2026-11-23, 判决轮未跑 => 不算改判据): 原写作「两台 async 里较优的那台」= 直接取 min,
      冒烟里 async_direct 的 once_main = -0.80 帧 <= 0, 被 min 挑中后 ratio_once = -0.16 => "省 >= 40%"
      会自动成立。负 once = 【过度抵消】(表 5 的稳态项在 s1 段被放大, 自洽差同时冲到 90%), 不是省。
      故分子只在 once_main > 0 的 async 臂里取优; 一台都不正 => 只报数、不下「省了多少」的结论。
      同时冒烟给 s1 开场那次成员设定加了一发 check_content() 逐位核(它在计时窗外, 不脏任何表)。
      🚨冒烟记账2(2026-11-23, 判决轮未跑 => 只【收紧】不改判据): 修⑥ —— 上面 J0 那句「(tag-tagf)/2 x 帧长
      与 表 4 的 M/次 之差 <= 0.50」在实现上是【归一化错配的假差】: 表 5 那笔稳态 = 每次翻转摊到半窗
      NF_T5/2 = 21/2 = 10.5 帧, 表 4 的 M = 整窗去掉头 NF_T5-HEAD_F = 15 帧 => 同一个每帧惩罚 P 除以两个
      不同分母, 直接相减恒差 10.5/7.5 = 1.4 倍 = 40% 假差, 那条自洽检查形同虚设(冒烟 sync 56.9 vs 37.4
      = 52% 被无脑判 J0, 而那不是数据坏了)。修后两列都归一到 ms/帧 再比 |P_T5 - P_M|/P_M, 门槛
      J0_TOL=0.50 一字不动 => 检查变严(冒烟归一后 sync 自洽差落到个位~十几百分点); 任一把读不出 = inf
      => J0 前置不过(以前 nan 比较恒 False = 静默跳过, 也算堵掉的漏洞)。
      🚨冒烟记账3(同上): 修⑤ —— R3 冒烟 4 次里偶发 1 次红(三台逐帧 md5 出现分歧帧), 而两个探针都复现不出:
      探针① 单机 6 pass x 60 帧不 flip = 0 分歧 => 引擎逐位可复现, R3 的逐位口径成立;
      探针② 高保真重建三臂、20 轮「s1 段停在翻转态 => 前置补 flip => 64 帧 trace」= 20/20 轮 0 分歧 ≈3840 帧。
      归因未定 => 不改判据、不让红线变松: 把 trace 主体抽成 r3_pass(), 只在【首轮红】时对全部臂原样复跑
      一遍并印出复跑的分歧帧集合 —— 与首轮逐帧相同 = 确定性【跨实现不等价】= 真红; 消失或变了 = 瞬时
      【状态相关】。判据仍只读首轮(相同臂数 != 全部就是红), 复跑只多花约 0.8s、只给那行 note 加归因。
      🚨冒烟记账4(同上): 修⑦ —— 那条【追加判据】(ratio_once <= 0.60 才允许写「异步省 >= 40%」)原先被末尾
      一对 if/else 无条件覆写成「追加判据未评」, 判决轮会永远印不出它 => 改成一条链: 表 5 读不出/红线不绿
      才「未评」, 否则一台 once_main>0 的 async 都没有 => 报「过度抵消」, 有 => 真评 ratio_once。纯显示口径。
      修⑧ —— [R3 前置] 那句 print 原先把原因写死成「SWAP_N 为奇数」, 但 §5.2h附2 的 s1 对照段开场就在 S1,
      SWAP_N=2(偶数)也会停在翻转态 => 原因按奇偶/末段分开印, 免得判决轮读出一句假话。
      修⑨ —— 那条【追加判据】原先不吃 J0 前置: 抵消不自洽时 once 本身就不可读, ratio_once 是拿不可读的
      数算出来的 => 现在落点为 J0 时它一并「未评」(只收紧不放宽; 判决轮若落 J1/J2/J3 它照常评)。
【§5.2h附2 事前预测(可证伪, 跑前写死)】押 J2, 且 ratio_once ≈ 0.6。理由(拿 r2 实测推, 假设不是结论):
  ① 一次性 = flip 自身 + 恢复窗的一半(恢复窗在 tag/tagf 里各计一次): sync = 16.1 + (0.30+21.76-6.92)/2
     = 23.7ms = 5.1 帧@4.646ms; async_staged = 2.8 + (3.64+26.44-6.96)/2 = 14.3ms = 3.1 帧 => 比 0.61;
  ② 落 J2 带(1~6 帧)而不是 J1: sync 的 flip 自身就有 16.1ms = 3.5 帧, 一次性不可能 <1 帧;
  ③ 两臂恢复窗 H 几乎同价(21.76 vs 26.44 ms/段)=> async 省的只有 flip 自身那 13ms => ratio_once 上限 ~0.6。
  竞争假设 = J1: 它若成立, 说明 H 那 22ms 也是 sync 独有的一次性, 一次性会掉到 1 帧以下 => thr_rel 该下调。
【§5.2h附2 跑前事实(读代码 + r1/r2 实测得出的算术上限, 用来裁决外部建议的两个新选项)】
  a) 设备端真拷贝 = 0.279ms/次(记账 22.78MB => 81GB/s), 而 async 的 flip 2.77ms 里 cpu_leg 占 2.42ms = 87%
     => 「INT4 传权重 / 增量迁移」能动的上限 <= 设备拷贝 0.28 + 提交 0.21 ≈ 0.5ms = 0.11 帧 => 不立项;
  b) cpu_leg 是【host->host】memcpy(3 份 fp32 转置 ≈3.5MB + 3 份 int8 ≈0.88MB), 不是 H2D; 它的前提 = C++
     入口每次收【整块 stacked 张量】=> CPU 腿只能写行、不能换指针(换指针 = _ptrs 变 = 重录图 = 红 R2)
     => 「预迁移做到 0 开销」有地板 = cpu_leg 2.4ms + 恢复窗 H, 不等于 0;
  c) hetero_moe_int8.py:501/:524 那份 GPU 侧 fp32 stacked = 裁判 _cpu_w_dev_cache 的副本, 不在生产前向路上
     => 它压根不在价签里, 谁也别拿它算收益。
【交付边界】① 三臂分相表(µs/次, 同一套键名, sync 没有侧 stream 所以 h2d_enq/d2d_enq 印 "-");
  ② flip 调用自身 host stall 表(probe us vs 类内 us, 差 = 前沿 drain); ③ 两种口径的价签 + ratio;
  ④ 四条红线 + 按 G 树自动印出的落点; ⑤ 表 4 事后归因(四窗份额 + 两把尺子的交叉对账)
  + 按 §5.2h附 I 树自动印出的落点; ⑥ 表 5 一次性价签(同成员对照, 稳态项抵消)+ 表 6 稳态惩罚曲线
  + 按 §5.2h附2 J 树自动印出的落点与那条追加判据。不改 thr_rel(除非落 I1 或 J1)、不改旧表、不碰裁判、
  不改生产语义(swapper 默认仍是 "sync")。
"""
import os, sys, time, hashlib, statistics, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H
import real_smollm as R
from dyn_placement import DynPlacer, policy_from_name

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, K, MODE = 42, 2, "L2_零CUDA+pinned"
LAYERS = tuple(int(v) for v in os.environ.get("LAYERS", "0,4,8,12,16,20,24,28").split(","))
GATE = os.environ.get("GATE", "embed")
CAP_LAYER = int(os.environ.get("CAP_LAYER", 8))
N_CHARS = int(os.environ.get("N_CHARS", 24000))
BS = int(os.environ.get("BS", 512))
GC = int(os.environ.get("GC", 3))
DTYPE = os.environ.get("DTYPE", "fp32").strip()
E = len(LAYERS)
# 与 bench_dynpair.py:110 逐字同源(条件不许漂): 换权重路径的全部前提都在这几个开关里
FULL = dict(gpu_impl="cat", graph_scope="full", n_threads=6, tok_block=8,
            pre_impl="fused", l2_stream="single", cpu_sched=1, d2h_wait="main")
TXT = os.environ.get("TXT", "readme")
SWAP_M = int(os.environ.get("SWAP_M", 64))            # 段长(帧)
SWAP_N = int(os.environ.get("SWAP_N", 2))             # 段内翻转次数(同一对槽位打一个来回)
REPS = int(os.environ.get("REPS", 15))                # 计时重复次数(价签取中位)
WARM_SEG = int(os.environ.get("WARM_SEG", 24))        # 每臂开场暖段(不进表)
WINDOW, WARM, CHECK = 24, 16, 8
ARMS = [a.strip() for a in os.environ.get("ARMS", "sync,async_direct,async_staged").split(",")
        if a.strip()]
PREDICT = os.environ.get("PREDICT", "G1")
G1_R = float(os.environ.get("G1_R", "0.3333"))        # G1: ratio <= 1/3
G2_R = float(os.environ.get("G2_R", "0.5"))           # G2: <= 1/2, 再大进 G4 判据
G4_SHARE = float(os.environ.get("G4_SHARE", "0.50"))  # G4: sync 的 front+tail 占总时长比
HEAD_F = int(os.environ.get("HEAD_F", "6"))           # §5.2h附: flip 后头几帧算「恢复窗」
I1_HEAD = float(os.environ.get("I1_HEAD", "0.60"))    # I1: 恢复窗吃掉 >= 60% 的钱
I1_POST = float(os.environ.get("I1_POST", "0.15"))    # I1: 换回基线后的残留 <= 15%
I2_MID = float(os.environ.get("I2_MID", "0.40"))      # I2: 翻转态稳态(去掉头)吃掉 >= 40%
I3_POST = float(os.environ.get("I3_POST", "0.30"))    # I3: 换回基线后仍残留 >= 30%
H0_CTRL = float(os.environ.get("H0_CTRL", "0.15"))    # I0 前置: 段内负对照 C 占 T <= 15%
H0_XCHK = float(os.environ.get("H0_XCHK", "0.25"))    # I0 前置: 逐帧口径与表 3 折成的 ms 差 <= 25%
PREDICT_I = os.environ.get("PREDICT_I", "I2")         # §5.2h附 事前预测(跑前写死)
# §5.2h附2: 同成员对照 = 把【稳态项】从设计上抵消, 只留一次性价签
J_ON = os.environ.get("J_ON", "1") == "1"              # 0 = 退回 r2 那三臂口径(省一半时间)
J_WARM = int(os.environ.get("J_WARM", "12"))           # 翻转态开场暖帧(不计时, 让 cache/页进稳态)
J0_TOL = float(os.environ.get("J0_TOL", "0.50"))       # J0 前置: 两把尺子的稳态项自洽容差
J1_FR = float(os.environ.get("J1_FR", "1.0"))          # J1: 一次性价签 <= 1 帧/次
J3_FR = float(os.environ.get("J3_FR", "6.0"))          # J3: 一次性价签 > 6 帧/次
J_RATIO = float(os.environ.get("J_RATIO", "0.60"))     # 追加判据: 异步化在一次性口径上省 >= 40%
PREDICT_J = os.environ.get("PREDICT_J", "J2")          # §5.2h附2 事前预测(跑前写死)
ARMSPEC = {"sync": dict(swapper="sync"),
           "async_direct": dict(swapper="async_direct", stage_ahead=False, prebuild=True, prof=True),
           "async_staged": dict(swapper="async_staged", stage_ahead=True, prebuild=True, prof=True)}
assert set(ARMS) <= set(ARMSPEC), f"未知臂, 可选 {sorted(ARMSPEC)}"
assert "sync" in ARMS, "sync 是价签分母, 不能摘"
assert SWAP_M >= 4 * SWAP_N, "翻转要落在段中间, 前后都得留出帧"
SLOT_I, SLOT_J = 0, GC                                   # 单对槽位往返(口径 4)
F_AT = [int(SWAP_M * (k + 1) / (SWAP_N + 1)) for k in range(SWAP_N)]
NF_T5 = F_AT[1] - F_AT[0] if SWAP_N == 2 else 0      # 修⑥: 翻转窗长[21,42)=21 帧, 只在 SWAP_N==2 有定义
ASYNC = [a for a in ARMS if a != "sync"]
PHASE_COLS = ["front_sync", "h2d_enq", "d2d_enq", "gpu_live", "gate", "cpu_leg",
              "tail_sync", "total"]


def rss_mb():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmRSS:"):
            return int(ln.split()[1]) / 1024.0
    return float("nan")


def med(v):
    return statistics.median(v) if len(v) else float("nan")


def _shr(v, tot):
    """份额专用: 分母 <= 0 一律 nan, 不许炸表格。"""
    return v / tot if tot > 0 else float("nan")


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


t_load = time.perf_counter()
print(f"[机器状态] {H.machine_state()}")
_SD = R.state_dict()
R.state_dict = lambda: _SD            # 建 3 台常驻引擎, 别每次重读 safetensors
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


BASE = R.real_src(LAYERS, gate_mode=GATE)      # 未置换基表: 换入换出的内容按 eid 从这里取
XREAL, _d = R.capture(layer=CAP_LAYER, bs=BS, n_chars=N_CHARS)
order_A = torch.argsort(hits_of(BASE["gate_w"], XREAL), descending=True).tolist()
# 两种成员态: 基线 S0 = 前 GC 热; 翻转态 S1 = 槽位对 (SLOT_I, SLOT_J) 换过一次之后
BASE_MEM = [int(v) for v in order_A[:GC]]
MEM_ODD = [int(order_A[SLOT_J])] + [int(v) for v in order_A[SLOT_I + 1:GC]]


def mk(gc, order, dt=DTYPE):
    torch.manual_seed(SEED)
    m = H.HeteroMoE(D, E, N, gpu_expert_count=gc, top_k=K, expert_kind="swiglu",
                    real_src=perm_src(order), cpu_delay_ms=0.0, gpu_dtype=dt, **FULL).cuda()
    m(torch.zeros(1, D, device="cuda"), async_mode=True, mode=MODE)
    return m


def fwd(m, x):
    with torch.no_grad():
        return m(x, async_mode=True, mode=MODE)


def ph(x):
    return hashlib.md5(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:12]


# ---------------------------------------------------------------- 一批发激活(只一段语料: 价签不比语料)
ids = TOK(R.real_text(N_CHARS), return_tensors="pt").input_ids[:, :BS]
X = capture_from(ids).cuda()
print(f"\n[真源] D={D} N={N} 层数={NL} | 取 {list(LAYERS)} 层 MLP 当 {E} 个专家 top_k={K}"
      f" | 路由={GATE} GC={GC} BS={BS} dtype={DTYPE} | 建模 {time.perf_counter() - t_load:.0f}s")
print(f"[序] A序={order_A}(GPU 成员={order_A[:GC]}) | 价签靶子 = dyn_placement.py:71/:264 那句"
      f"「换一次 ~15ms = 3~4 帧」(thr_rel=0.05 按它标定)")
print(f"[段] {len(ARMS)} 臂 x {REPS} 轮 x (lat|cont) x (base|swap){'|basef|swapf' if J_ON else ''}"
      f" | 段长 {SWAP_M} 帧"
      f" | 翻转 {SWAP_N} 次@帧 {F_AT} 槽位对=({SLOT_I},{SLOT_J}) 往返 | 暖段 {WARM_SEG} 帧")

ENG = {}
for a in ARMS:
    t0 = time.perf_counter()
    m = mk(GC, order_A, DTYPE)
    # 🚨 async 的 gate_abs 在 ctor 里定 => src_gate 必须走 **swap_kw(不能像 dynpair 那样事后挂)
    dyn = DynPlacer(m, BASE["experts"], GC, policy=policy_from_name("static_A", GC),
                    window=WINDOW, check_every=CHECK, warm=WARM, order=order_A,
                    src_gate=BASE["gate_w"], **ARMSPEC[a])
    sw = dyn.sw
    ENG[a] = dict(m=m, dyn=dyn, sw=sw, staged=(a == "async_staged"),
                  v0=sw._versions(), p0=sw._ptrs(), build_s=time.perf_counter() - t0,
                  pinned_mb=getattr(sw, "pinned_mb", 0.0),
                  tag={mo: [] for mo in ("lat", "cont")},
                  ph={mo: [] for mo in ("lat", "cont")},
                  probe={mo: [] for mo in ("lat", "cont")},
                  cls_us={mo: [] for mo in ("lat", "cont")},
                  dev=[] , mb=[], ruler=[], per={"base": [], "swap": []},
                  tag2={mo: [] for mo in ("lat", "cont")},
                  perf={"basef": [], "swapf": []})
    print(f"  [建臂 {a:<13}] {ENG[a]['build_s']:.1f}s | pinned={ENG[a]['pinned_mb']:.1f}MB"
          f" | 类={type(sw).__name__} RSS={rss_mb():.0f}MB 显存={torch.cuda.memory_allocated()/1024**2:.0f}MB")


# ---------------------------------------------------------------- 翻转与计时
def do_flip(arm, mode, recs, book=True):
    """一次翻转 = swap 语义(换出再换回靠调用两次同一对槽位实现)。计时段内【不查内容】(见红线口径)。
    book=False = §5.2h附2 的 s1 对照段: 它的 flip 方向相反(S1->S0), 混进表 2/表 4 的 F 就是脏分母。"""
    sw = arm["sw"]
    eid_in = int(sw.slot2eid[SLOT_J])
    t0 = time.perf_counter()
    if arm["staged"]:
        # 🚨 必须在 fwd() 入队【之后】调: cont 口径下侧 stream 的 H2D 才与那一帧计算重叠
        sw.plan(eid_in)
    rec = sw.swap(SLOT_I, SLOT_J)
    dt = (time.perf_counter() - t0) * 1e6
    if book:
        arm["probe"][mode].append(dt)
        arm["cls_us"][mode].append(rec["us"])
        arm["mb"].append(rec["mb"])
    recs.append(rec)
    return rec


def segment(arm, mode, flips, start="s0"):
    """跑一段 SWAP_M 帧。返回 (墙钟 ms, 每帧 ms 列表或 None, 本次 flip 记录)。
    start="s1" = §5.2h附2 的同成员对照: 开场先在【翻转态】(计时窗外换成员 + J_WARM 暖帧),
    于是这一整段的稳态就是翻转态 => 它与 swapf 段的稳态差相互抵消, 剩下才是一次性代价。"""
    m, sw = arm["m"], arm["sw"]
    want = BASE_MEM if start == "s0" else MEM_ODD
    if [int(v) for v in sw.slot2eid[:GC]] != want:
        torch.cuda.synchronize()
        sw.swap(SLOT_I, SLOT_J)          # 计时窗外的成员设定, 只进「迁移总账」不进任何表
        torch.cuda.synchronize()
        # 🚨 s1 段的整条结论都建立在「开场成员真的是 S1」上 => 当场逐位核内容(计时窗外, 不脏表)
        bad = sw.check_content()
        assert not bad, f"🚨 s1 开场成员设定后 slot2eid 与内容不符: {bad}"
    if start == "s1":
        for _ in range(J_WARM):
            fwd(m, X)
        torch.cuda.synchronize()
    sw.reset_prof()
    per, recs = ([] if mode == "lat" else None), []
    torch.cuda.synchronize()
    tw0 = time.perf_counter()
    for f in range(SWAP_M):
        if mode == "lat":
            t0 = time.perf_counter()
            fwd(m, X)
            torch.cuda.synchronize()
            per.append((time.perf_counter() - t0) * 1e3)
        else:
            fwd(m, X)
        if flips and f in F_AT:
            do_flip(arm, mode, recs, book=(start == "s0"))
    torch.cuda.synchronize()
    wall = (time.perf_counter() - tw0) * 1e3
    phz = sw.phase_us()
    phz["n"] = sw.phase_n
    if hasattr(sw, "dev_ms") and start == "s0":
        arm["dev"] += sw.dev_ms()     # reset_prof 每段清 flips => 不收 s1 段不会攒出脏数
    return wall, per, recs, phz


# ---------------------------------------------------------------- 开场暖段(不进表)
for a in ARMS:
    segment(ENG[a], "lat", False)
    segment(ENG[a], "lat", True)          # 把 flip 的代码路径也暖过(含 pinned 表首触页)
    ENG[a]["probe"] = {mo: [] for mo in ("lat", "cont")}     # 暖段那几次不算数, 清空
    ENG[a]["cls_us"] = {mo: [] for mo in ("lat", "cont")}
    ENG[a]["dev"], ENG[a]["mb"] = [], []
print(f"[暖段] 完成 RSS={rss_mb():.0f}MB 显存={torch.cuda.memory_allocated()/1024**2:.0f}MB")

# ---------------------------------------------------------------- 计时主循环
t_run = time.perf_counter()
def run_seg(tag, start, fl, mode):
    """发一段并把墙钟/逐帧放进对应容器。tag 只当容器名用, 不再靠 bool 区分。"""
    for a in ARMS:
        wall, per, recs, phz = segment(ENG[a], mode, fl, start=start)
        if tag == "swap":
            ENG[a]["ph"][mode].append(phz)
        if tag in ("base", "swap"):
            ENG[a]["tag"][mode].append((wall, fl))
        else:
            ENG[a]["tag2"][mode].append((wall, fl))
        if per is None:
            continue
        if tag == "base":
            ENG[a]["ruler"] += per
            ENG[a]["per"]["base"].append(per)
        elif tag == "swap":
            ENG[a]["per"]["swap"].append(per)                  # §5.2h附 表 4 吃这个
        else:
            ENG[a]["perf"][tag].append(per)                    # §5.2h附2 表 5/表 6 吃这个


for rep in range(REPS):
    for mode in ("lat", "cont"):
        first = "base" if rep % 2 == 0 else "swap"             # 口径 5: 奇偶轮互换先后
        for tag, fl in ((first, first == "swap"),
                        ("base" if first == "swap" else "swap", first != "swap")):
            run_seg(tag, "s0", fl, mode)
        if J_ON:                                               # §5.2h附2 同成员对照段
            for tag, fl in ((first + "f", first == "swap"),
                            (("base" if first == "swap" else "swap") + "f", first != "swap")):
                run_seg(tag, "s1", fl, mode)
    if (rep + 1) % 5 == 0 or rep == 0:
        print(f"  [轮 {rep + 1}/{REPS}] RSS={rss_mb():.0f}MB"
              f" loadavg={'/'.join(str(round(v, 2)) for v in os.getloadavg())}"
              f" 已用 {time.perf_counter() - t_run:.0f}s")

# ---------------------------------------------------------------- 表 1 分相
print("\n" + "=" * 118)
print("表 1 分相(µs/次, 每次 = 一次翻转; 同一套键名, sync 没有侧 stream => h2d_enq/d2d_enq 无键印 \"-\")")
print("      front_sync=flip 前那发 device-wide sync | gpu_live=写 live 存储+热权重缓冲 | "
      "tail_sync=清派生缓存+末尾 sync")
print("=" * 118)
print(f"{'臂':<14}{'口径':<6}" + "".join(f"{k:>12}" for k in PHASE_COLS) + f"{'次':>5}")
PH_SAVE = {}
for a in ARMS:
    for mo in ("lat", "cont"):
        rows = ENG[a]["ph"][mo]
        if not rows:
            continue
        acc = {k: med([r.get(k, float("nan")) for r in rows]) for k in PHASE_COLS}
        PH_SAVE[(a, mo)] = acc
        print(f"{a:<14}{mo:<6}" + "".join(
            (f"{acc[k]:>12.0f}" if k in acc and acc[k] == acc[k] else f"{'-':>12}")
            for k in PHASE_COLS) + f"{rows[-1]['n']:>5d}")

# ---------------------------------------------------------------- 表 2 flip 自身 host stall
print("\n" + "=" * 118)
print("表 2 flip 调用自身的 host stall(µs/次)。probe = 探针在 flip 调用前后各取一次表;"
      " 类内 = swapper 自己报的 us; 差 = 前沿 drain(sync 才有)")
print("      🚨 这一列不是价签: async 的账落在下一帧的设备时间线上, 这里读不出来")
print("=" * 118)
print(f"{'臂':<14}{'口径':<6}{'probe p50':>11}{'probe min':>11}{'类内 p50':>11}{'差(前沿)':>11}"
      f"{'设备ms/次':>11}{'记账MB/次':>11}")
for a in ARMS:
    for mo in ("lat", "cont"):
        pr, cu = ENG[a]["probe"][mo], ENG[a]["cls_us"][mo]
        if not pr:
            continue
        dv = f"{med(ENG[a]['dev']):.3f}" if ENG[a]["dev"] else "   -"
        print(f"{a:<14}{mo:<6}{med(pr):>11.0f}{min(pr):>11.0f}{med(cu):>11.0f}"
              f"{med(pr) - med(cu):>11.0f}{dv:>11}{med(ENG[a]['mb']):>11.2f}")

# ---------------------------------------------------------------- 表 3 价签
print("\n" + "=" * 118)
print("表 3 价签(帧/次) = (swap 段墙钟 - base 段墙钟)/帧长尺子/翻转次数; 帧长尺子 = base 段 lat 的 p50(全臂共用)")
print("      末列「旧价签」印的 3.5 = dyn_placement.py:71/:264 那句「换一次 ~15ms = 3~4 帧」的中值, 不是实测")
print("      🚨 判据读 tag_main = max(lat, cont): 两种口径都算, 取较大的那个, 不给挑列留门")
print("=" * 118)
RULER = med([v for a in ARMS for v in ENG[a]["ruler"]])
TAG = {}
print(f"{'臂':<14}{'帧长尺子ms':>11}{'base lat':>10}{'swap lat':>10}{'tag_lat':>9}"
      f"{'base cont':>10}{'swap cont':>10}{'tag_cont':>10}{'tag_main':>10}{'ratio':>8}{'旧价签':>10}")
for a in ARMS:
    row = {}
    for mo in ("lat", "cont"):
        b = [w for w, fl in ENG[a]["tag"][mo] if not fl]
        s = [w for w, fl in ENG[a]["tag"][mo] if fl]
        row[mo] = med(s) - med(b)
    tags = {mo: row[mo] / RULER / SWAP_N for mo in ("lat", "cont")}
    TAG[a] = dict(base=med([w for w, fl in ENG[a]["tag"]["lat"] if not fl]),
                  swap=med([w for w, fl in ENG[a]["tag"]["lat"] if fl]), main=max(tags.values()),
                  **tags)
for a in ARMS:
    t = TAG[a]
    ratio = t["main"] / TAG["sync"]["main"] if a != "sync" and TAG["sync"]["main"] > 0 else float("nan")
    print(f"{a:<14}{RULER:>11.3f}{t['base']:>10.1f}{t['swap']:>10.1f}{t['lat']:>9.2f}"
          f"{med([w for w, fl in ENG[a]['tag']['cont'] if not fl]):>10.1f}"
          f"{med([w for w, fl in ENG[a]['tag']['cont'] if fl]):>10.1f}{t['cont']:>10.2f}"
          f"{t['main']:>10.2f}{ratio:>8.2f}{3.5:>10.1f}")
FRAME_MS = RULER
print(f"\n  旧价签 = 「~15ms = 3~4 帧」(dyn_placement.py:71/:264), 本表帧长尺子 = {FRAME_MS:.3f}ms"
      f" => 15ms = {15.0 / FRAME_MS:.1f} 帧")

# ---------------------------------------------------------------- 表 4 事后归因(§5.2h附)
print("\n" + "=" * 118)
print("表 4 事后归因(§5.2h附): 表 3 那笔钱落在【哪几帧】。ex[f] = med(swap 段第 f 帧) - med(base 段第 f 帧)")
print(f"      口径 = lat 逐帧墙钟, 跨发取中位后再相减; 窗: C=flip 前[0,{F_AT[0] if SWAP_N == 2 else '-'}) = 段内负对照"
      f" | H=每次 flip 后头 {HEAD_F} 帧 = 恢复窗 | M=翻转态去掉头 = 稳态 | P=换回基线去掉头 = 残留")
print("=" * 118)
ATTR = {}
if SWAP_N != 2:
    print(f"  🚨 SWAP_N={SWAP_N} != 2 => C/H/M/P 四个窗只在【同一对槽位打一枪换回来】这套脚本下有定义 => 整表跳过")
else:
    f1, f2 = F_AT

    print(f"{'臂':<14}" + "".join(f"{h:>12}" for h in ("C(负对照)", "H(恢复窗)", "M(稳态)", "P(残留)", "T(合计)"))
          + f"{'C/T':>7}{'H/T':>7}{'M/T':>7}{'P/T':>7}{'T ms/次':>9}{'F ms/次':>9}{'T+F ms/次':>10}"
          f"{'表3 ms/次':>10}{'发':>5}")
    for a in ARMS:
        B, S = ENG[a]["per"]["base"], ENG[a]["per"]["swap"]
        if len(B) < 3 or len(S) < 3:
            print(f"{a:<14}  发数不足(base {len(B)} 发 / swap {len(S)} 发, 需 >=3)=> 不读")
            continue
        cb = [med([r[f] for r in B]) for f in range(SWAP_M)]
        cs = [med([r[f] for r in S]) for f in range(SWAP_M)]
        ex = [c - b for c, b in zip(cs, cb)]
        C = sum(ex[:f1])
        H = sum(ex[f1:f1 + HEAD_F]) + sum(ex[f2:f2 + HEAD_F])
        M = sum(ex[f1 + HEAD_F:f2])
        P = sum(ex[f2 + HEAD_F:])
        T = sum(ex)
        # 🚨 flip 自身落在【帧与帧之间】, 不在任何 per 帧里(lat 口径逐帧计时在 flip 之外) => 对账必须补上它
        fl = med(ENG[a]["cls_us"]["lat"]) / 1e3 * SWAP_N     # ms/段
        book = TAG[a]["lat"] * FRAME_MS                      # 表 3 的墙钟价签折成 ms/次
        ATTR[a] = dict(T=T, book=book, flip=fl, C=C, H=H, M=M, P=P, n=len(S))
        print(f"{a:<14}" + "".join(f"{v:>12.2f}" for v in (C, H, M, P, T))
              + "".join(f"{_shr(v, T):>7.2f}" for v in (C, H, M, P))
              + f"{T / SWAP_N:>9.1f}{fl / SWAP_N:>9.1f}{(T + fl) / SWAP_N:>9.1f}{book:>10.1f}{len(S):>5d}")
    print("      T = 四个帧窗之外的段总超额(ms/段); F = flip 调用自身(lat 口径下它落在帧与帧之间, 不在任何 per 帧里);")
    print("      对账口径 = (T + F)/次 对 表3 ms/次 = tag_lat x 帧长 —— 少了 F 这两把尺子恒差一个 flip 自身。"
          " 🚨 份额分母只用 T: 本树问的是 flip【之外】的钱落在哪几帧, flip 自身由表 2 单独钉死")

# ---------------------------------------------------------------- 表 5 一次性价签(§5.2h附2 同成员对照)
print("\n" + "=" * 118)
print("表 5 一次性价签(§5.2h附2): 每轮加发两段【整段都跑在翻转态】的对照 basef/swapf, 把【稳态项】从设计上抵消")
print("      tag  = med(swap) - med(base)   从基线 S0 开场  = 一次性 + 整窗稳态惩罚")
print("      tagf = med(swapf)- med(basef)  从翻转态 S1 开场 = 同一笔一次性 - 整窗稳态惩罚(符号相反 = 设计如此)")
print("      => once = (tag+tagf)/2 = 一次性价签(帧/次) | 稳态项 = (tag-tagf)/2 x 帧长 = ms/次")
print(f"      修⑥: 两把尺子【分母帧数不同】=> 自洽检查归一到 ms/帧再比: 表 5 稳态项/(翻转窗 {NF_T5} 帧 / 2)"
      f" vs 表 4 M/({NF_T5} - 头 {HEAD_F} = {NF_T5 - HEAD_F} 帧)")
print("      🚨 落点读 once_main = max(lat, cont); 稳态项只对 lat(与表 4 同口径), 它是自洽检查不是判据")
print("=" * 118)
TAG1 = {}
if not J_ON:
    print("  🚨 J_ON=0 => 没发 basef/swapf 对照段 => 表 5/表 6/J 树整块读不出(本轮只复读 r1/r2 那三张表)")
else:
    for a in ARMS:
        pair, nok = {}, []
        for mo in ("lat", "cont"):
            b = [w for w, fl in ENG[a]["tag"][mo] if not fl]
            sv = [w for w, fl in ENG[a]["tag"][mo] if fl]
            bf = [w for w, fl in ENG[a]["tag2"][mo] if not fl]
            sf = [w for w, fl in ENG[a]["tag2"][mo] if fl]
            nok.append(min(len(b), len(sv), len(bf), len(sf)))
            if min(len(b), len(sv), len(bf), len(sf)) < 3:
                pair = None
                break
            pair[mo] = ((med(sv) - med(b)) / RULER / SWAP_N, (med(sf) - med(bf)) / RULER / SWAP_N)
        if pair is None:
            print(f"  🚨 {a} basef/swapf 发数不足(最少那列 {min(nok)} 发, 需 >=3)=> 不读")
            continue
        once = {mo: (pair[mo][0] + pair[mo][1]) / 2 for mo in ("lat", "cont")}
        steady = (pair["lat"][0] - pair["lat"][1]) / 2 * FRAME_MS      # ms/次, 见 docstring: = (NF_T5/2) x P
        # 修⑥: 表 5 那笔稳态 = 每次翻转摊到【半窗 NF_T5/2 帧】, 表 4 的 M = 【整窗去掉头 NF_T5-HEAD_F 帧】。
        #   两边都是同一个每帧惩罚 P, 但分母 10.5 vs 15 => 直接相减恒差 10.5/7.5=1.4 倍 = 假差。
        #   归一到 ms/帧才是真检查: |P_T5 - P_M|/P_M。门槛 J0_TOL 不动 => 这是【收紧】不是放宽。
        p_t5 = steady / (NF_T5 / 2.0) if NF_T5 > 0 else float("nan")                  # ms/帧
        m_abs = ATTR[a]["M"] if a in ATTR else float("nan")                            # ms/段
        p_m = m_abs / (NF_T5 - HEAD_F) if NF_T5 > HEAD_F else float("nan")             # ms/帧
        self_ok = (abs(p_t5 - p_m) / max(abs(p_m), 1e-9)
                   if p_t5 == p_t5 and p_m == p_m else float("inf"))   # 任一把读不出 = inf => J0 前置不过
        TAG1[a] = dict(pair=pair, once=once, main=max(once.values()), steady=steady, p_t5=p_t5,
                       p_m=p_m, self_ok=self_ok, n=min(nok))
    best1 = [a for a in ASYNC if a in TAG1 and TAG1[a]["main"] > 0]
    ro = {}
    for a in TAG1:
        ro[a] = (TAG1[a]["main"] / TAG1["sync"]["main"]
                 if a != "sync" and "sync" in TAG1 and TAG1["sync"]["main"] > 0 else float("nan"))
    print(f"{'臂':<14}{'tag_lat':>9}{'tagf_lat':>9}{'once_lat':>9}{'tag_con':>10}{'tagf_con':>10}"
          f"{'once_con':>10}{'once_main':>10}{'ratio_once':>11}{'稳态ms/帧':>11}{'表4 M/帧':>10}{'自洽差':>9}{'发':>5}")
    for a in ARMS:
        if a not in TAG1:
            continue
        t = TAG1[a]
        print(f"{a:<14}{t['pair']['lat'][0]:>9.2f}{t['pair']['lat'][1]:>9.2f}{t['once']['lat']:>9.2f}"
              f"{t['pair']['cont'][0]:>9.2f}{t['pair']['cont'][1]:>9.2f}{t['once']['cont']:>10.2f}"
              f"{t['main']:>10.2f}{ro[a]:>11.2f}{t['p_t5']:>11.3f}{t['p_m']:>10.3f}"
              f"{100 * t['self_ok']:>8.0f}%{t['n']:>5d}")
    print("      🚨 tagf 是【负】的很正常: 从翻转态开场时, 窗内那段反而换成了更优的基线成员 => 稳态项倒过来记;")
    print("      once 与表 3 的 tag 是两把不同的尺子 —— 表 3 含稳态项(I2 已判它作废), once 剥掉了它。")
    print(f"      修⑥记账: 印的两列都是【ms/帧】(P) —— 表 5 自洽差 <50% 才说明【每帧惩罚是常数】=")
    print("      表 5 那句「稳态项在设计上抵消」成立; r2 之前那列 ms/次 与 表 4 的 M/次 分母不同, 那条检查形同虚设。")

# ---------------------------------------------------------------- 表 6 稳态惩罚曲线(§5.2h附2)
NBIN = 8
print("\n" + "=" * 118)
print("表 6 稳态惩罚曲线(§5.2h附2): ex[f] = med(basef 段第 f 帧) - med(base 段第 f 帧) —— 两段窗内都不翻转,")
print(f"      唯一差别 = 整段开场成员态(S0 vs S1); 每 {max(1, SWAP_M // NBIN)} 帧一箱, 印 {NBIN} 个箱的箱中位(ms/帧)")
print("      🚨 读【形状】不读平不平: 箱间平 = 惩罚是【台阶】(换了成员就永久贵一档) | 单调爬 = 惩罚是【爬坡】")
print("      爬坡 => 表 5 那句「稳态项在设计上抵消」不成立(它不是常数)=> 抵消自洽差会变大, J 树落 J0")
print("=" * 118)
CURVE = {}
if not J_ON:
    print("  🚨 J_ON=0 => 没有 basef 段 => 曲线画不出")
else:
    print(f"{'臂':<14}" + "".join(f"{('箱%d' % k):>9}" for k in range(NBIN))
          + f"{'头2箱':>9}{'末2箱':>9}{'末-头':>9}{'表4 M/帧':>10}{'发':>5}")
    f1_, f2_ = (F_AT[0], F_AT[1]) if SWAP_N == 2 else (0, 0)
    m_frame = (f2_ - f1_ - HEAD_F) if SWAP_N == 2 else 0
    for a in ARMS:
        B, FF = ENG[a]["per"]["base"], ENG[a]["perf"]["basef"]
        if len(B) < 3 or len(FF) < 3:
            print(f"{a:<14}  base/basef 发数不足(base {len(B)} / basef {len(FF)}, 需 >=3)=> 不读")
            continue
        cb = [med([r[f] for r in B]) for f in range(SWAP_M)]
        cf = [med([r[f] for r in FF]) for f in range(SWAP_M)]
        ex = [c - b for c, b in zip(cf, cb)]
        bins = [med(ex[k * SWAP_M // NBIN:(k + 1) * SWAP_M // NBIN]) for k in range(NBIN)]
        CURVE[a] = bins
        h2, t2 = med(bins[:2]), med(bins[-2:])
        mf = (ATTR[a]["M"] / m_frame) if (a in ATTR and m_frame > 0) else float("nan")
        print(f"{a:<14}" + "".join(f"{v:>9.2f}" for v in bins)
              + f"{h2:>9.2f}{t2:>9.2f}{t2 - h2:>9.2f}{mf:>10.2f}{len(FF):>5d}")
    print("      「表4 M/帧」= 表 4 的 M 除以翻转窗去掉头之后的帧数 = 同一件事的另一把尺子(它数的是 flip 之后,")
    print("      本表数的是【从未 flip 过、开场就在 S1】的段 => 两把尺子同高 = 惩罚与 flip 无关, 纯成员态。")

# ---------------------------------------------------------------- 四条红线(非计时专用趟)
print("\n" + "=" * 118)
print("红线(全部在非计时趟里查; check_content() 自己是 D2H 同步, 塞进 cont 段 = 把被测的异步性测没)")
print("=" * 118)
RED = []
N_CHECKED = 0


def flip_checked(arm):
    """一次 flip + 逐位核内容。🚨 只在非计时趟里调(check_content 自己是 D2H 同步)。"""
    global N_CHECKED
    sw = arm["sw"]
    N_CHECKED += 1
    if arm["staged"]:
        sw.plan(int(sw.slot2eid[SLOT_J]))
    sw.swap(SLOT_I, SLOT_J)
    bad = sw.check_content()
    assert not bad, f"🚨R1 flip 后 slot2eid 与内容不符: {bad}"


# R1 + R3: 同一串翻转, 三台逐帧输出 md5 必须逐位同
TRACE = {}
END_MEM = BASE_MEM if SWAP_N % 2 == 0 else MEM_ODD      # SWAP_N 为奇数 => 计时趟本就结束在翻转态


def r3_pass(a):
    """跑【一遍】 R3 trace 并返回逐帧 md5 列表。前置: 该臂成员在基线; 收尾把成员带回基线
    (SWAP_N 为奇数时补 1 次 flip), 所以可以原样复跑第二遍 —— 修⑤ 就靠这个复跑做归因。"""
    arm = ENG[a]
    assert [int(v) for v in arm["sw"].slot2eid[:GC]] == BASE_MEM, f"R3 前置: {a} 成员不在基线"
    out = []
    for f in range(SWAP_M):
        y = fwd(arm["m"], X)
        torch.cuda.synchronize()
        out.append(ph(y))
        if f in F_AT:
            flip_checked(arm)                        # R1: 每次 flip 后逐位核内容
    torch.cuda.synchronize()
    assert [int(v) for v in arm["sw"].slot2eid[:GC]] == END_MEM, f"🚨{SWAP_N} 次翻转后 {a} 成员态应 {END_MEM}"
    if SWAP_N % 2:
        flip_checked(arm)                               # 奇数 => 补回基线, 让 R4/终态从基线出发
    return out


for a in ARMS:
    if [int(v) for v in ENG[a]["sw"].slot2eid[:GC]] != BASE_MEM:
        flip_checked(ENG[a])                            # 奇数趟收尾在翻转态 => 补 1 次 flip 回基线(口径容差, 非判据)
        print(f"  [R3 前置] {a} 计时趟结束在翻转态(SWAP_N={SWAP_N}"
              f"{'为奇数' if SWAP_N % 2 else '末段 = s1 同成员对照段, 开场成员本就是 S1'})=> 已补 1 次 flip 回基线")
    TRACE[a] = r3_pass(a)
ref = TRACE[ARMS[0]]
DIFF = {a: [i for i, (u, v) in enumerate(zip(TRACE[a], ref)) if u != v] for a in ARMS}
same = [a for a in ARMS if not DIFF[a]]
note3 = (f"逐帧相同臂数 {len(same)}/{len(ARMS)} 首帧 {ref[0]} 翻转后首帧 {ref[F_AT[0] + 1]}"
         + "".join(f" | {a}@帧{DIFF[a][:5]} 共{len(DIFF[a])}帧" for a in ARMS if DIFF[a]))
# 修⑤: 首轮红时【原样复跑一遍】给红归因 —— 判据仍读首轮(不让复跑把红线变松), 复跑只多印一句话
if len(same) != len(ARMS):
    TR2 = {a: r3_pass(a) for a in ARMS}
    ref2 = TR2[ARMS[0]]
    D2 = {a: [i for i, (u, v) in enumerate(zip(TR2[a], ref2)) if u != v] for a in ARMS}
    s2 = [a for a in ARMS if not D2[a]]
    if len(s2) == len(ARMS):
        v3 = "复跑全绿 => 瞬时【状态相关】非跨实现不等价(判决仍按首轮读红)"
    elif D2 == DIFF:
        v3 = "复跑分歧帧集合与首轮【逐帧相同】=> 确定性【跨实现不等价】= 真红"
    else:
        v3 = ("复跑仍红但帧集合变了("
              + ";".join(f"{a}@帧{D2[a][:3]}" for a in ARMS if D2[a]) + ") => 瞬时【状态相关】")
    note3 += f" || 修⑤复跑: 相同臂 {len(s2)}/{len(ARMS)} => {v3}"
RED.append(("R1 每次 flip 后 check_content 逐位", True,
            f"已核 {N_CHECKED} 次全部为空表(不符项列表为空)"))
RED.append(("R3 三台同串翻转逐帧输出 md5 逐位同", len(same) == len(ARMS), note3))
# R2: 地址与 _version 全程没动
bad2 = [a for a in ARMS if ENG[a]["sw"]._versions() != ENG[a]["v0"] or ENG[a]["sw"]._ptrs() != ENG[a]["p0"]]
RED.append(("R2 _versions/_ptrs 全程不变(没重录图/没换地址)", not bad2, f"红臂={bad2}"))
# R4: 不 sync 就 flip => 那一帧不许 torn
bad4 = []
for a in ARMS:
    arm = ENG[a]
    y_ref = fwd(arm["m"], X)
    torch.cuda.synchronize()
    h_ref = ph(y_ref)
    y = fwd(arm["m"], X)                             # 入队, 不 sync
    flip_checked(arm)                                # 立刻 flip
    torch.cuda.synchronize()
    if ph(y) != h_ref:
        bad4.append(a)
    ENG[a]["sw"].swap(SLOT_I, SLOT_J)                # 往返复原
    torch.cuda.synchronize()
RED.append(("R4 flip 撞在飞行帧上 => 该帧输出 == flip 前参照(无 torn)", not bad4, f"红臂={bad4}"))
for nm, ok, note in RED:
    print(f"  {'OK ' if ok else '🚨FAIL'} {nm:<52} {note}")
print(f"  成员终态: " + " ".join(f"{a}={ENG[a]['sw'].slot2eid[:GC]}" for a in ARMS)
      + f" | 基线={order_A[:GC]} | 迁移总账 " +
      " ".join(f"{a} {ENG[a]['sw'].n_swap}次/{ENG[a]['sw'].total_us/1e3:.0f}ms" for a in ARMS))

# ---------------------------------------------------------------- 预设 G 树落点
ratio = min([(TAG[a]["main"] / TAG["sync"]["main"]) for a in ASYNC if TAG["sync"]["main"] > 0]
            or [float("nan")])
best_async = min(ASYNC, key=lambda a: TAG[a]["main"]) if ASYNC else None
sp = PH_SAVE.get(("sync", "cont"), PH_SAVE.get(("sync", "lat"), {}))
tot = sp.get("total", 0.0)
share = (sp.get("front_sync", 0.0) + sp.get("tail_sync", 0.0)) / tot if tot else float("nan")
red_ok = all(ok for _, ok, _ in RED)
if not red_ok:
    land, why = "G3", "任一红线不绿 => 实现坏, 不得用「快了」记账"
elif ratio <= G1_R:
    land, why = "G1", f"ratio={ratio:.3f} <= 1/3 且四红线全绿 => 价签按实测重贴 + thr_rel 重标定"
elif ratio <= G2_R:
    land, why = "G2", f"1/3 < ratio={ratio:.3f} <= 1/2 => 瓶颈不在这两处, 回分相表重找, 不落价签"
elif share == share and share >= G4_SHARE:
    land, why = "G4", (f"ratio={ratio:.3f} > 1/2 且 sync 的 front+tail 占 total = {100*share:.0f}%"
                       f" >= {100*G4_SHARE:.0f}% => 钱花在等排空不是搬字节 => 价签口径改成「尾巴+flip」"
                       f" ⇒ §5.2i 的 F2 复活")
else:
    land, why = "挂账", (f"ratio={ratio:.3f} > 1/2 但 sync 的 front+tail 只占 {100*share:.0f}%"
                         f" < {100*G4_SHARE:.0f}% => 四条都没命中, 这一刀没能重贴出价签")
print("\n" + "=" * 118)
print("预设判决树落点(跑前写死, 跑完不许改) —— 求值序 G3 -> G1 -> G2 -> G4 -> 其余挂账")
print("=" * 118)
print(f"  最优 async = {best_async}  tag_main={TAG[best_async]['main'] if best_async else float('nan'):.2f} 帧/次"
      f" | sync tag_main={TAG['sync']['main']:.2f} 帧/次 | ratio = {ratio:.3f}")
print(f"  sync 分相里 front_sync+tail_sync / total = {100*share:.1f}%(读 cont 口径)"
      f" | 门槛 G1<={G1_R:.3f} G2<={G2_R:.3f} G4 份额>={100*G4_SHARE:.0f}%")
print(f"  => 落点: {land}    理由: {why}")
print(f"     事前预测 = {PREDICT}(押 G1; 竞争假设 = G4「钱主要在前后排空」) => "
      + ("兑现" if land == PREDICT else "打脸(如实记账)"))
print("\n判读提示: ① 价签 = 墙钟差/帧长/次数, 读 flip 自己的 µs 会系统性偏袒 async;"
      " ② tag_main 取 lat/cont 较大者, 落点不许挑列;"
      "\n            ③ 红线 R3 不绿 = async 与 sync 不等价, 那时快多少都不许记账;"
      " R4 的牙咬在「侧 stream 越界写 live」;"
      "\n            ④ 落 G1 才允许改 dyn_placement.py:71/:264 那句价签与 thr_rel;"
      " 落 G2/G4/挂账 一张旧表都不动;"
      "\n            ⑤ 落 G4 => §5.2i 的事前预测从 F1 改回 F2(切换成本吃光自适应收益), 且必须写明口径是设备时间线。")
# ---------------------------------------------------------------- §5.2h附 I 树落点
print("\n" + "=" * 118)
print("§5.2h附 事后归因落点(判据跑前写死在 docstring, 跑完不许改) —— 求值序 I0 -> I1 -> I2 -> I3 -> 其余挂账")
print("      🚨 落点只读【sync 臂】= 生产路 = 表 3 那张价签的分母; 另两臂只作旁证, 不参与落点")
print("      🚨 上一棵 G 树(r1 归档 6eabc1de)已消耗 => 复跑只复读它的落点, 本轮真正的落点是这棵 I 树")
print("=" * 118)
st = ATTR.get("sync")
if st is None:
    land_i, why_i = "I0", "表 4 没数据(SWAP_N != 2 或发数 < 3)=> 事后归因读不出"
elif not red_ok:
    land_i, why_i = "I0", "任一红线不绿(与 G3 同一件事)=> 表 3 与表 4 都不可读"
elif st["T"] <= 0:
    land_i, why_i = "I0", f"sync 的 T={st['T']:.2f}ms <= 0 => 份额无意义, 表 4 整表不可读"
else:
    C, H, M, P, T = st["C"], st["H"], st["M"], st["P"], st["T"]
    xchk = abs((T + st["flip"]) / SWAP_N - st["book"]) / max(st["book"], 1e-9)
    print(f"  sync 交叉对账: 逐帧 T/次={T / SWAP_N:.1f} + flip 自身={st['flip'] / SWAP_N:.1f}"
          f" = {(T + st['flip']) / SWAP_N:.1f}ms vs 表 3 价签={st['book']:.1f}ms"
          f" => 差 {100 * xchk:.0f}%(门槛 <= {100 * H0_XCHK:.0f}%) | 段内负对照 C/T={100 * C / T:.0f}%"
          f"(门槛 <= {100 * H0_CTRL:.0f}%)")
    if C / T > H0_CTRL:
        land_i, why_i = "I0", (f"段内负对照 C/T={100 * C / T:.0f}% > {100 * H0_CTRL:.0f}%"
                               f" => flip 之前两段的帧序就已经不等价 => 归因尺子坏了")
    elif xchk > H0_XCHK:
        land_i, why_i = "I0", (f"逐帧 (T + F)/次={(T + st['flip']) / SWAP_N:.1f}ms 与表 3 价签={st['book']:.1f}ms"
                               f" 差 {100 * xchk:.0f}% > {100 * H0_XCHK:.0f}% => 两把尺子没对上 => 表 4 不可读")
    elif H / T >= I1_HEAD and P / T <= I1_POST:
        land_i, why_i = "I1", (f"H/T={100 * H / T:.0f}% >= {100 * I1_HEAD:.0f}% 且 P/T={100 * P / T:.0f}%"
                               f" <= {100 * I1_POST:.0f}% => 钱 = flip 后的一次性恢复成本"
                               f" => 价签【可以】重贴成 {T / SWAP_N / FRAME_MS:.1f} 帧/次, thr_rel 按它重标定")
    elif M / T >= I2_MID:
        land_i, why_i = "I2", (f"M/T={100 * M / T:.0f}% >= {100 * I2_MID:.0f}% => 钱 = 翻转态的【稳态放置惩罚】"
                               f" 被表 3 整笔记成了切换成本 => 🚨表 3 价签口径作废, thr_rel 不许按本表重标定")
    elif P / T >= I3_POST:
        land_i, why_i = "I3", (f"P/T={100 * P / T:.0f}% >= {100 * I3_POST:.0f}% => 换回基线仍然慢"
                               f" => 代价随驻留时长走、不随 flip 次数摊薄 => 口径同样要改")
    else:
        land_i, why_i = "挂账", (f"C/T={100 * C / T:.0f}% H/T={100 * H / T:.0f}% M/T={100 * M / T:.0f}%"
                                 f" P/T={100 * P / T:.0f}% => 三个份额门槛都没过, 钱在四个窗里摊平了")
print("  旁证(不参与落点): " + " | ".join(
    f"{a} H/T={100 * _shr(ATTR[a]['H'], ATTR[a]['T']):.0f}%"
    f" M/T={100 * _shr(ATTR[a]['M'], ATTR[a]['T']):.0f}%"
    f" P/T={100 * _shr(ATTR[a]['P'], ATTR[a]['T']):.0f}% T/次={ATTR[a]['T'] / SWAP_N:.1f}ms"
    for a in ARMS if a in ATTR))
print(f"  => 落点: {land_i}    理由: {why_i}")
print(f"     事前预测(跑前写死) = {PREDICT_I}(押 I2「翻转态稳态」; 竞争假设 = I1「一次性恢复成本」) => "
      + ("兑现" if land_i == PREDICT_I else "打脸(如实记账)"))
print("\n判读提示: ① 落点只读 sync; 另两行是旁证 —— 它们把 flip 自己压到 2.9ms 之后各窗怎么变,"
      " 只用来交叉验证「钱不在 flip 里」;"
      "\n            ② 探针【故意】选最贵的槽位对 (0, GC) = 最热 eid 换第 4 热 => I2 兑现否定的是【探针口径】,"
      " 不是「生产切换这么贵」(生产里 dyn 只在收益 > thr_rel 时才换, 稳态项符号相反);"
      "\n            ③ 只有 I1 兑现才允许把价签重贴成「N 帧/次」并重标定 thr_rel; I2/I3/挂账 一张旧表都不动;"
      "\n            ④ 表 4 吃 lat 逐帧墙钟, 表 3 吃段墙钟差, 是两个口径 => I0 那条交叉对账先过才有后面三条;"
      "\n            ⑤ 落 I2/I3 => §5.2i 的事前预测按「切换成本吃光自适应收益」写(口径必须写明是稳态而非每次)。")
# ---------------------------------------------------------------- §5.2h附2 J 树落点
print("\n" + "=" * 118)
print("§5.2h附2 一次性价签落点(判据跑前写死在 docstring, 跑完不许改) —— 求值序 J0 -> J1 -> J2 -> J3 -> 其余挂账")
print("      🚨 落点只读【sync 臂】= 生产路; 两台 async 只作旁证 + 当 ratio_once 的分子")
print("      🚨 G 树(r1 归档 6eabc1de)与 I 树(r2 归档 0e305a89)都已消耗 => 复跑只复读它们, 本轮真正落点是这棵 J 树")
print("=" * 118)
sj = TAG1.get("sync")
if not J_ON:
    land_j, why_j = "J0", "J_ON=0 => 没发同成员对照段 basef/swapf => 一次性价签读不出"
elif sj is None:
    land_j, why_j = "J0", "表 5 没数据(basef/swapf 发数不足)=> 一次性价签读不出"
elif not red_ok:
    land_j, why_j = "J0", "任一红线不绿(与 G3 同一件事)=> 全部表不可读"
elif sj["main"] <= 0:
    land_j, why_j = "J0", (f"sync 的 once={sj['main']:.2f} 帧/次 <= 0 => 抵消过头: 这个口径里根本没有正的"
                           f" 一次性代价, 稳态项符号或窗长假设需要先自查")
elif sj["self_ok"] > J0_TOL:
    land_j, why_j = "J0", (f"抵消不自洽(修⑥归一化后): 每帧稳态惩罚 表 5={sj['p_t5']:.3f}ms/帧 vs 表 4"
                           f"={sj['p_m']:.3f}ms/帧 差 {100 * sj['self_ok']:.0f}%"
                           f" > {100 * J0_TOL:.0f}% => 稳态项非线性(爬坡)或读不出 => 一次性读不出")
elif sj["main"] <= J1_FR:
    land_j, why_j = "J1", (f"once_main(sync)={sj['main']:.2f} 帧/次 <= {J1_FR:.1f}"
                           f" => 旧价签「~15ms = 3~4 帧」高估 >= 3 倍 => 允许下调 thr_rel=0.05")
elif sj["main"] <= J3_FR:
    land_j, why_j = "J2", (f"once_main(sync)={sj['main']:.2f} 帧/次 落在 ({J1_FR:.1f}, {J3_FR:.1f}]"
                           f" => 与旧价签同量级 => 旧表一张不动, 但必须写明「一次性」与「稳态」是两笔账")
else:
    land_j, why_j = "J3", (f"once_main(sync)={sj['main']:.2f} 帧/次 > {J3_FR:.1f}"
                           f" => 换一次成员 >= 6 帧 => 生产里几乎永远不该换 => 滞回阈值要大幅提高")
if sj:
    b1 = [a for a in ASYNC if a in TAG1 and TAG1[a]["main"] > 0]   # 🚨 once<=0 = 过度抵消 => 不进分子
    bb = min(b1, key=lambda a: TAG1[a]["main"]) if b1 else None
    r1_ = (TAG1[bb]["main"] / sj["main"]) if bb and sj["main"] > 0 else float("nan")
    print(f"  sync once_main = {sj['main']:.2f} 帧/次 = {sj['main'] * FRAME_MS:.1f}ms/次"
          + (f" | 最优 async = {bb} once_main = {TAG1[bb]['main']:.2f} 帧/次"
             f" | ratio_once = {r1_:.2f}" if bb else " | 🚨 本轮没跑 async 臂 => ratio_once 读不出"))
    print(f"  门槛 J1<={J1_FR:.1f} 帧 J3>{J3_FR:.1f} 帧 | J0 自洽容差 {100 * J0_TOL:.0f}%"
          f" | 追加判据 ratio_once<={J_RATIO:.2f} | 帧长尺子 {FRAME_MS:.3f}ms")
    print(f"  旁证(不参与落点): " + " | ".join(
        f"{a} once={TAG1[a]['main']:.2f}帧 P={TAG1[a]['p_t5']:.3f}ms/帧 自洽差={100 * TAG1[a]['self_ok']:.0f}%"
        for a in ARMS if a in TAG1))
print(f"  => 落点: {land_j}    理由: {why_j}")
_pos_async = [a for a in ASYNC if a in TAG1 and TAG1[a]["main"] > 0]   # 🚨 once<=0 = 过度抵消 => 不进分子
_add = (" 追加判据未评(表 5 读不出 / 红线不绿 / J_ON=0)"
        if land_j != "J0" else
        " 追加判据未评: 落点已是 J0 = 抵消不自洽(或读不出)=> once 本身不成立, ratio_once 跟着不成立")
if J_ON and sj and red_ok and land_j != "J0":
    if not _pos_async:
        _add = (" 🚨 没有任何 async 臂的 once_main > 0 => 那是【过度抵消】(稳态项在 s1 段被放大), "
                "不是「异步把一次性抹平了」=> 只报数, 不做「省了多少」的结论")
    else:
        r_ok = bb is not None and r1_ == r1_ and r1_ <= J_RATIO
        _add = (" 追加结论: 异步化在【一次性价签】上省 >= 40%"
                f"(ratio_once={r1_:.2f} <= {J_RATIO:.2f})" if r_ok else
                f" 不追加「异步省 >= 40%」: ratio_once={r1_:.2f} > {J_RATIO:.2f}"
                " => 异步化的钱主要在【稳态并行性】上, 不在一次性切换本身上")
print(f"     事前预测(跑前写死) = {PREDICT_J}(押 J2 且 ratio_once ≈ 0.6; 竞争假设 = J1「H 窗也是 sync 独有一次性」) => "
      + ("兑现" if land_j == PREDICT_J else "打脸(如实记账)"))
print(f"    {_add}")
print("\n判读提示: ① tagf 与 tag 符号相反是【设计如此】(见 表 5 脚注), 它不是数据坏了; 抵消只在稳态项是常数时成立,")
print("            那一条件由 J0 的自洽差和 表 6 的曲线形状一起把关;")
print("            ② 一次性与稳态是两笔账: 表 4/I2 已经把稳态那笔单独挂过, 本树只重贴一次性那一笔 ——")
print("            dyn 的滞回开关点 thr_rel 要的是「换一次付几帧」= 一次性, 稳态项在真决策里符号是反的;")
print("            ③ 探针【故意】选最贵槽位对 (0, GC) = 最热 eid 换第 4 热 => J2 兑现不等于「生产该付这笔」;")
print("            ④ 只有 J1 兑现才允许下调 thr_rel; J2/J3/J0 一律只更正口径写法, 一张旧表都不动;")
print("            ⑤ 落 J1 => §5.2i 的事前预测回到「自适应有净收益」; 落 J3 => 按「切换成本吃光自适应收益」写。")
print(f"[耗时] 建臂+暖段+计时共 {time.perf_counter() - t_run:.0f}s |"
      f" RSS={rss_mb():.0f}MB 显存={torch.cuda.memory_allocated()/1024**2:.0f}MB")
