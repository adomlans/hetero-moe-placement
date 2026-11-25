#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <immintrin.h>
#include <omp.h>   // ✅ T4: OpenMP 多线程 kernel
#include <vector>
#include <cmath>   // ✅ 乙-2: expf/rintf (silu + h 再量化)

namespace py = pybind11;

// 🚨 本机 i5-10400 (Comet Lake) 无 AVX-VNNI => 永远走 #else 的 maddubs
//   证据: objdump -d avx_gemv_int8.so | grep -c vpdpbusd  == 0

inline int32_t hsum256(__m256i acc) {
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

// ============================================================
// ✅ T10: 两种权重打包, 两种数值行为
//
//   biased(S8=false, 历史): 权重存 +128 偏置的 uint8 [0,255], 靠 (d - 128*sum_x) 去偏。
//     🚨 有静默 bug: _mm256_maddubs_epi16 把【相邻两个乘积之和】塞进 int16, 而
//     |x|<=127 且 w_raw<=255 => 单对乘积 <=32385, 一对之和最坏 64770 > 32767 =>
//     【饱和截断】。实测(i5-10400, 配置C)每帧有几百对被截, 凭空多出 0.4818% 的
//     相对误差 —— 理想 int32 精确累加只有 1.3844%, 真 kernel 量到 1.4660%。
//     仿真逐位复现了 kernel(见 bench_int8_exactness.py), 根因就是这个饱和。
//     另外 VNNI 分支 _mm256_dpbusd_epi32(acc, vx, vw) 把【有符号的 x 当无乘数】,
//     负数会被当成 129..255 => 在有 AVX-VNNI 的机器上结果整个是错的。
//
//   signed(S8=true, 现在默认): 权重存真·对称 int8 [-127,127] (量化时就不加偏置),
//     用 |w| 当无乘数、x*sign(w) 当有乘数: maddubs(|w|, x*sign(w)) = Σ w_i*x_i,
//     而 |每对之和| <= 2*127*127 = 32258 < 32767 => 【不可能饱和, 可证明】;
//     同一套操作数角色正好符合 dpbusd 的 (u8,s8) 规定 => 两条路径逐位一致;
//     也不再需要 128*sum_x 这个修正项(少一趟 per-token 的 hidden 求和)。
//     代价: 每个 32B 块多一条 _mm256_sign_epi8(abs 在 tok_block 内是共享的)。
// ============================================================
template <bool S8>
inline void dot_int8_1(const int8_t* w, const int8_t* x, int len, int32_t* res) {
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i <= len - 32; i += 32) {
        __m256i vw = _mm256_loadu_si256((const __m256i*)(w + i));
        __m256i vx = _mm256_loadu_si256((const __m256i*)(x + i));
#if defined(__AVXVNNI__)
        if (S8) {
            acc = _mm256_dpbusd_epi32(acc, _mm256_abs_epi8(vw), _mm256_sign_epi8(vx, vw));
        } else {
            acc = _mm256_dpbusd_epi32(acc, vx, vw);   // 🚨 历史写法, 操作数角色是反的
        }
#else
        if (S8) {
            __m256i vp = _mm256_maddubs_epi16(_mm256_abs_epi8(vw), _mm256_sign_epi8(vx, vw));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(vp, _mm256_set1_epi16(1)));
        } else {
            __m256i vp = _mm256_maddubs_epi16(vw, vx);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(vp, _mm256_set1_epi16(1)));
        }
#endif
    }
    int32_t r = hsum256(acc);
    for (; i < len; ++i) {
        int wv = S8 ? (int)w[i] : (int)(uint8_t)w[i];
        r += (int32_t)x[i] * wv;
    }
    *res = r;
}

// ============================================================
// ✅ T5 核心: 一条权重行只从 L3 搬一次, 同时喂 TB 个 token —— GEMM 化
//    算术语义与 dot_int8_1 逐位相同: 每个 token 自己的累加顺序一个字没动,
//    改的只是"权重行被重复载入的次数"。
// ============================================================
template <int TB, bool S8>
inline void dot_int8_tb(const int8_t* w, const int8_t* const* xs, int len, int32_t* res) {
    __m256i acc[TB];
#pragma GCC unroll 8
    for (int t = 0; t < TB; ++t) acc[t] = _mm256_setzero_si256();
    int i = 0;
#if !defined(__AVXVNNI__)
    const __m256i ones = _mm256_set1_epi16(1);
#endif
    for (; i <= len - 32; i += 32) {
        // 🚨 整个内层唯一一次权重载入, 后面 TB 个 token 全共用它
        __m256i vw = _mm256_loadu_si256((const __m256i*)(w + i));
        if (S8) {
            __m256i wa = _mm256_abs_epi8(vw);          // 每 32B 只算一次, TB 个 token 共享
#pragma GCC unroll 8
            for (int t = 0; t < TB; ++t) {
                __m256i vx = _mm256_loadu_si256((const __m256i*)(xs[t] + i));
                __m256i vx2 = _mm256_sign_epi8(vx, vw);
#if defined(__AVXVNNI__)
                acc[t] = _mm256_dpbusd_epi32(acc[t], wa, vx2);
#else
                acc[t] = _mm256_add_epi32(acc[t],
                    _mm256_madd_epi16(_mm256_maddubs_epi16(wa, vx2), ones));
#endif
            }
        } else {
#pragma GCC unroll 8
            for (int t = 0; t < TB; ++t) {
                __m256i vx = _mm256_loadu_si256((const __m256i*)(xs[t] + i));
#if defined(__AVXVNNI__)
                acc[t] = _mm256_dpbusd_epi32(acc[t], vx, vw);
#else
                __m256i vp = _mm256_maddubs_epi16(vw, vx);
                acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(vp, ones));
#endif
            }
        }
    }
#pragma GCC unroll 8
    for (int t = 0; t < TB; ++t) {
        int32_t r = hsum256(acc[t]);
        for (int j = i; j < len; ++j) {
            int wv = S8 ? (int)w[j] : (int)(uint8_t)w[j];
            r += (int32_t)xs[t][j] * wv;
        }
        res[t] = r;
    }
}

// ============================================================
// 并行块: 一个专家的一小撮 token, 按输出神经元 i 切线程
//   不同 i 写不同 out 元素 => 无数据竞争; 同一 i 的 TB 个 token 由同一线程串行写
//   => 即使同一 token 撞上同一专家两次也安全
// ============================================================
template <int TB, bool S8>
static void run_block(const int8_t* w_mat, int expert_size, int hidden,
                      const int8_t* const* x_rows, const int* bidx,
                      const float* comb, const int32_t* sumx, float* out_ptr,
                      int n, int n_threads, int sched) {
    // 一行的全部工作(该行的 TB 个 token 由同一条线程串行做完 => 累加顺序不变)
    auto run_row = [&](int i) {
        const int8_t* wr = w_mat + (size_t)i * hidden;
        for (int off = 0; off < n; off += TB) {
            const int m = TB < n - off ? TB : (n - off);
            const int8_t* xr[TB];
            int32_t d[TB];
#pragma GCC unroll 8
            for (int t = 0; t < m; ++t) xr[t] = x_rows[off + t];
            if (m == TB) {
                dot_int8_tb<TB, S8>(wr, xr, hidden, d);
            } else {
#pragma GCC unroll 8
                for (int t = 0; t < m; ++t) dot_int8_1<S8>(wr, xr[t], hidden, &d[t]);
            }
            for (int t = 0; t < m; ++t) {
                const int b = bidx[off + t];
                // S8: 权重本来就没偏置, 不需要 128*sum_x 修正
                const int32_t corr = S8 ? 0 : 128 * sumx[b];
                out_ptr[b * expert_size + i] += (d[t] - corr) * comb[off + t];
            }
        }
    };
    // ============================================================
    // ✅ T14: 谁来算哪一行?
    //   sched=0(历史) static,32 —— 开跑前就把 2048 行平摊给 6 条线程。帧内主线程
    //                要同时做 graph replay / D2H 入队, 它和绑在 core0 的 OMP 线程0
    //                同核 => 线程0 半速 => 【整个并行区等它一个】。
    //                实测(配置C 真实 slot 分布 29/26/4, 6 线程, 纯 CPU 卡尺):
    //                  独占 478/559us   外挂 1 个抢核进程 839/905us  <- 帧内量的就是 811us
    //   sched=1(新)  guided —— 每条线程从队尾自取【递减】块, 慢的自动少拿活。
    //                  独占 453/505us   外挂 1 个抢核进程 569/590us(-32%)
    //                  配置A(6 slot) 抢核时 109 -> 79us(-25%)
    //   🚨 逐位等价: 只是换了"哪条线程算哪一行", 每个 (行 i, token t) 的点积、
    //      以及同一 (b,i) 上多个 slot 的累加顺序(都在同一次行迭代内串行)都没变。
    // ============================================================
    if (sched) {
#pragma omp parallel for schedule(guided) num_threads(n_threads) if (n_threads > 1)
        for (int i = 0; i < expert_size; ++i) run_row(i);
    } else {
#pragma omp parallel for schedule(static, 32) num_threads(n_threads) if (n_threads > 1)
        for (int i = 0; i < expert_size; ++i) run_row(i);
    }
}

// ============================================================
// 纯计算内核: 【零 CUDA 调用】, 输入全部必须是 CPU 连续张量
// ============================================================
template <bool S8>
static void moe_int8_impl(const torch::Tensor& x_cpu,
                          const torch::Tensor& x_scales_cpu,
                          const torch::Tensor& idx_cpu,
                          const torch::Tensor& scores_cpu,
                          const torch::Tensor& w_cpu,
                          const torch::Tensor& w_scales_cpu,
                          float* out_ptr,
                          int gpu_expert_count,
                          int n_threads,
                          int tok_block,
                          int loop_order, int sched) {
    const int bs = x_cpu.size(0);
    const int hidden = x_cpu.size(1);
    const int expert_size = w_cpu.size(1);
    const int n_cpu_experts = (int)w_cpu.size(0);
    const int top_k = idx_cpu.size(1);

    const int8_t* x_ptr = x_cpu.data_ptr<int8_t>();
    // 权重两种打包都是 1 字节/元素(biased=uint8, signed=int8), 按 int8 视图取指针即可
    const int8_t* w_ptr = reinterpret_cast<const int8_t*>(w_cpu.data_ptr());
    const int64_t* idx_ptr = idx_cpu.data_ptr<int64_t>();
    const float* score_ptr = scores_cpu.data_ptr<float>();
    const float* x_scales_ptr = x_scales_cpu.data_ptr<float>();
    const float* w_scales_ptr = w_scales_cpu.data_ptr<float>();

    TORCH_CHECK(n_cpu_experts <= 256, "too many cpu experts");
    const int max_slots = bs * top_k;
    TORCH_CHECK(max_slots <= 8192, "static slot buffer too small");

    // ✅ T5-前置: 按"专家"分桶。同一个专家的 token 挤在一起, 才能共用一次权重搬运。
    //    老写法外层是 token, 每个 token 各扫一遍整张 2MB 权重 —— 那就是带宽墙本身。
    static thread_local int lst_b[8192], lst_k[8192];
    static thread_local int head[257], cnt[256], fill[256];
    for (int e = 0; e < n_cpu_experts; ++e) cnt[e] = 0;
    for (int b = 0; b < bs; ++b)
        for (int k = 0; k < top_k; ++k) {
            int64_t eid = idx_ptr[b * top_k + k];
            if (eid < gpu_expert_count) continue;
            // 越界检查: 老写法这里会直接算出越界的 w_mat 偏移, 静默读垃圾内存
            TORCH_CHECK(eid < gpu_expert_count + n_cpu_experts,
                        "expert id ", eid, " out of range");
            cnt[eid - gpu_expert_count]++;
        }
    head[0] = 0;
    for (int e = 0; e < n_cpu_experts; ++e) head[e + 1] = head[e] + cnt[e];
    for (int e = 0; e < n_cpu_experts; ++e) fill[e] = head[e];
    for (int b = 0; b < bs; ++b)
        for (int k = 0; k < top_k; ++k) {
            int64_t eid = idx_ptr[b * top_k + k];
            if (eid < gpu_expert_count) continue;
            int e = (int)(eid - gpu_expert_count);
            lst_b[fill[e]] = b; lst_k[fill[e]] = k; fill[e]++;
        }

    // sum_x 每 token 只算一次(老写法每个 slot 重算一遍); signed 打包不需要它
    static thread_local int32_t sumx[8192];
    if constexpr (!S8) {
        for (int b = 0; b < bs; ++b) {
            const int8_t* xr = x_ptr + (size_t)b * hidden;
            int32_t s = 0;
            for (int j = 0; j < hidden; ++j) s += (int32_t)xr[j];
            sumx[b] = s;
        }
    }

    const int tb = tok_block == 1 ? 1 : (tok_block == 2 ? 2 : (tok_block == 4 ? 4 : 8));

    // ============================================================
    // ❌ T14 否证: 每专家的 token 块循环, 放在【行循环外面】还是【里面】?
    //   当初的假设: order=0 每满 tb 个 token 就把 2MB 权重【整趟重读】, 配置C 专家5
    //   摊 29 token => 5 趟 = 10MB, 白搬 14MB DRAM => 应该有 -40%。
    //   实测(纯 CPU 卡尺, /tmp 探针): 单核 13.5 MAC/cyc, 6 线程 288 GMAC/s, 而且
    //   【MAC/s 与趟数无关】—— [8]=1趟 与 [64]=8趟 的 MAC/s 一样(55.4 vs 54.2 单核)。
    //   这个 kernel 是 AVX2 maddubs 的【算力墙】, 重扫的权重全在 L2/L3 里, 不要钱。
    //   order=1 实测: C 独占 459->506us(更差), 抢核时 564->591us(更差) => 默认关。
    //   order=0(默认): 外层把该专家的 token 切成 <=tb 的块, 每块一次 run_block
    //                = 一次【完整 2MB 权重扫描】+ 一次 OMP 并行区启动。
    //                命中该专家的 token 越多, 同一张权重被整趟重读的遍数越多:
    //                配置C(bs=32) 专家5=29 token -> 8+8+8+4+1 = 5 趟 = 10MB,
    //                专家6=26 -> 4 趟, 专家7=4 -> 1 趟, 合计 ~20MB, 而信息量只有 6MB。
    //   order=1(新)  : 一个专家只进一次并行区, token 块循环在 run_block 的【行循环
    //                里面】=> 一行权重(1KB)进 L1 后被这个专家的所有 token 复用,
    //                整张 2MB 每帧只从 L2/L3 过一遍。寄存器需求一点没变(还是 tb 个
    //                int32 累加器), 变的只是循环嵌套顺序。
    //   🚨 逐位等价: 每个 (行 i, token t) 的点积与它的累加顺序都没变, 变的只是
    //      尾部余数由谁处理(原来贪心 2 的幂切块, 现在 8 的整块 + 余数走 dot_int8_1),
    //      而 dot_int8_1 和 dot_int8_tb 对同一 token 是同一串指令 => 结果逐位相同。
    // ============================================================
    static thread_local const int8_t* lst_x[8192];
    static thread_local float lst_c[8192];
    static thread_local int lst_bi[8192];

    for (int e = 0; e < n_cpu_experts; ++e) {
        const int n_tok = cnt[e];
        if (n_tok == 0) continue;
        const int8_t* w_mat = w_ptr + (size_t)e * expert_size * hidden;
        const float w_scale = w_scales_ptr[e];
        // 🚨 槽位表是分专家连续存放的, 必须用 head[e] 做基址偏移
        const int* lb = lst_b + head[e];
        const int* lk = lst_k + head[e];

        if (loop_order) {
            const int he = head[e];
            for (int t = 0; t < n_tok; ++t) {
                const int b = lb[t], k = lk[t];
                lst_x[he + t] = x_ptr + (size_t)b * hidden;
                lst_bi[he + t] = b;
                lst_c[he + t] = x_scales_ptr[b] * w_scale * score_ptr[b * top_k + k];
            }
            const int8_t** xr = lst_x + he;
            const float* cb = lst_c + he;
            const int* bi = lst_bi + he;
            switch (tb) {
                case 8: run_block<8, S8>(w_mat, expert_size, hidden, xr, bi, cb, sumx, out_ptr, n_tok, n_threads, sched); break;
                case 4: run_block<4, S8>(w_mat, expert_size, hidden, xr, bi, cb, sumx, out_ptr, n_tok, n_threads, sched); break;
                case 2: run_block<2, S8>(w_mat, expert_size, hidden, xr, bi, cb, sumx, out_ptr, n_tok, n_threads, sched); break;
                default: run_block<1, S8>(w_mat, expert_size, hidden, xr, bi, cb, sumx, out_ptr, n_tok, n_threads, sched); break;
            }
            continue;
        }

        // ✅ 贪心切块: 每次取 <= 剩余额度 的最大 2 的幂。
        //    这样"某专家只摊到 5 个 token"时也是 4+1 两块(2 趟权重),
        //    而不是退化成 5 个孤立 token 各扫一遍权重(5 趟)。
        for (int t0 = 0; t0 < n_tok; ) {
            int n = n_tok - t0; if (n > tb) n = tb;
            while (n & (n - 1)) --n;          // 向下取整到 2 的幂
            const int8_t* x_rows[8];
            float comb[8];
            int bidx[8];
            for (int t = 0; t < n; ++t) {
                const int b = lb[t0 + t], k = lk[t0 + t];
                x_rows[t] = x_ptr + (size_t)b * hidden;
                bidx[t] = b;
                comb[t] = x_scales_ptr[b] * w_scale * score_ptr[b * top_k + k];
            }
            switch (n) {
                case 8: run_block<8, S8>(w_mat, expert_size, hidden, x_rows, bidx, comb, sumx, out_ptr, n, n_threads, sched); break;
                case 4: run_block<4, S8>(w_mat, expert_size, hidden, x_rows, bidx, comb, sumx, out_ptr, n, n_threads, sched); break;
                case 2: run_block<2, S8>(w_mat, expert_size, hidden, x_rows, bidx, comb, sumx, out_ptr, n, n_threads, sched); break;
                default: run_block<1, S8>(w_mat, expert_size, hidden, x_rows, bidx, comb, sumx, out_ptr, n, n_threads, sched); break;
            }
            t0 += n;
        }
    }
}

// 运行期选打包方式。signed_w: -1=按张量 dtype 自动判(int8=>signed, 其它=>biased),
// 1=强制 signed(可证明无饱和), 0=强制历史 biased。自动判定 = 不可能配错。
static bool want_signed(int64_t signed_w, const torch::Tensor& w) {
    if (signed_w >= 0) return signed_w != 0;
    return w.scalar_type() == torch::kInt8;
}

template <class... A>
static void dispatch_moe(int64_t signed_w, const torch::Tensor& w, A&&... a) {
    if (want_signed(signed_w, w)) moe_int8_impl<true>(std::forward<A>(a)...);
    else                          moe_int8_impl<false>(std::forward<A>(a)...);
}

static void check_cpu(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_cpu(), name, " must be on CPU (got ", t.device(), ")");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// ============================================================
// ✅ 乙-2: 真形状 expert = SwiGLU 三矩阵
//     g = x@Wg, u = x@Wu, h = silu(g)*u, out = h@Wd
//
// 【为什么这一刀不是"多写两行 GEMM"】
//   卖点(冷专家在 CPU 做 int8 GEMV)是【字节 bound】: 单矩阵→三矩阵 = 每张专家
//   2MB→6MB, 字节墙 49us/专家 直接变 147us/专家; 中间那个 h 还必须【再量化成 int8】
//   才能喂第二趟 GEMV => 精度上多一笔真实的量化代价, 必须单独定价(见 bench_swiglu.py)。
//
// 【布局约定】(与 python 侧一一对应, D=hidden, N=expert_size)
//   wq_g / wq_u : (Ecpu, N, D)  行 i = 第 i 个中间神经元的整行(长度 D)  <- 与 gemm 的 cpu_w 同布局
//   wq_d        : (Ecpu, D, N)  行 j = down 的第 j 个输出(长度 N)      <- 转置存, 保证 GEMV 连续
//   out         : (bs, D)       注意输出宽度从 N 变成 D(真模型 residual 维)
//
// 【数值约定】
//   只允许 signed(int8) 打包。biased 那条在 T10 已证明 maddubs 会 int16 饱和,
//   而且第二阶段的去偏项是 128*sum(h_row) —— 它按【slot】而不是按 token 定义,
//   同一 token 的两个 slot 撞同一专家时 sumx[b] 是错的。别把旧 bug 复活到新路径上。
//   h 的量化: 每 slot 对称格点 scale = amax/127, 取整用 rintf(= 半偶舍入, 与
//   torch.round 同规则) => 和 torch 侧的 int8 仿真能对上到 1~2 LSB 以内。
// ============================================================

// 通用 GEMV: n 个 int8 行向量 x 一张 (n_out x len) 的 int8 权重, 结果【原样 int32 点积】
// 写进 scr[t*scr_stride + i]。按输出行 i 切线程 => 每个 (t,i) 只由一条线程算 => 结果与
// 线程数无关(和 run_block 同一条确定性论证)。tok_block 的语义与 T5 一致: 一行权重进
// L1 后同时喂 TB 个 slot。
template <int TB, bool S8>
static void gemv_i32(const int8_t* w_mat, int n_out, int len,
                     const int8_t* const* x_rows, int32_t* scr, int scr_stride,
                     int n, int n_threads, int sched, const int32_t* sumx) {
    auto run_row = [&](int i) {
        const int8_t* wr = w_mat + (size_t)i * len;
        for (int off = 0; off < n; off += TB) {
            const int m = TB < n - off ? TB : (n - off);
            const int8_t* xr[TB];
            int32_t d[TB];
#pragma GCC unroll 8
            for (int t = 0; t < m; ++t) xr[t] = x_rows[off + t];
            if (m == TB) {
                dot_int8_tb<TB, S8>(wr, xr, len, d);
            } else {
#pragma GCC unroll 8
                for (int t = 0; t < m; ++t) dot_int8_1<S8>(wr, xr[t], len, &d[t]);
            }
            for (int t = 0; t < m; ++t) {
                const int32_t corr = S8 ? 0 : 128 * sumx[off + t];
                scr[(size_t)(off + t) * scr_stride + i] = d[t] - corr;
            }
        }
    };
    if (sched) {
#pragma omp parallel for schedule(guided) num_threads(n_threads) if (n_threads > 1)
        for (int i = 0; i < n_out; ++i) run_row(i);
    } else {
#pragma omp parallel for schedule(static, 32) num_threads(n_threads) if (n_threads > 1)
        for (int i = 0; i < n_out; ++i) run_row(i);
    }
}

// 阶段1: 同一个 slot 批同时算 g 和 u(两趟 GEMV, 共用同一批 x 行指针)
template <int TB, bool S8>
static void swg_stage_a(const int8_t* wg, const int8_t* wu, int n_mid, int hidden,
                        const int8_t* const* xr, int32_t* G, int32_t* U, int n,
                        int n_threads, int sched) {
    gemv_i32<TB, S8>(wg, n_mid, hidden, xr, G, n_mid, n, n_threads, sched, nullptr);
    gemv_i32<TB, S8>(wu, n_mid, hidden, xr, U, n_mid, n, n_threads, sched, nullptr);
}

template <bool S8>
static void moe_swiglu_impl(const torch::Tensor& x_cpu,
                            const torch::Tensor& x_scales_cpu,
                            const torch::Tensor& idx_cpu,
                            const torch::Tensor& scores_cpu,
                            const torch::Tensor& wg_cpu,
                            const torch::Tensor& wu_cpu,
                            const torch::Tensor& wd_cpu,
                            const torch::Tensor& sg_cpu,
                            const torch::Tensor& su_cpu,
                            const torch::Tensor& sd_cpu,
                            float* out_ptr,
                            int gpu_expert_count, int n_threads, int tok_block, int sched) {
    const int bs = x_cpu.size(0);
    const int hidden = x_cpu.size(1);              // D
    const int n_mid = (int)wg_cpu.size(1);         // N = 中间/门控维 = 第一/二张的输出宽
    const int n_cpu_experts = (int)wg_cpu.size(0);
    const int top_k = (int)idx_cpu.size(1);

    const int8_t* x_ptr = x_cpu.data_ptr<int8_t>();
    const int8_t* pg = reinterpret_cast<const int8_t*>(wg_cpu.data_ptr());
    const int8_t* pu = reinterpret_cast<const int8_t*>(wu_cpu.data_ptr());
    const int8_t* pd = reinterpret_cast<const int8_t*>(wd_cpu.data_ptr());
    const int64_t* idx_ptr = idx_cpu.data_ptr<int64_t>();
    const float* score_ptr = scores_cpu.data_ptr<float>();
    const float* x_scales_ptr = x_scales_cpu.data_ptr<float>();
    const float* sg_ptr = sg_cpu.data_ptr<float>();
    const float* su_ptr = su_cpu.data_ptr<float>();
    const float* sd_ptr = sd_cpu.data_ptr<float>();

    TORCH_CHECK(n_cpu_experts <= 256, "too many cpu experts");
    const int max_slots = bs * top_k;
    TORCH_CHECK(max_slots <= 8192, "static slot buffer too small");

    // ---- 按专家分桶(与 gemm 版同一段代码; 故意复制而不是抽公共函数: 那边是历史档位, 不动它)
    static thread_local int lst_b[8192], lst_k[8192], head[257], cnt[256], fill[256];
    for (int e = 0; e < n_cpu_experts; ++e) cnt[e] = 0;
    for (int b = 0; b < bs; ++b)
        for (int k = 0; k < top_k; ++k) {
            int64_t eid = idx_ptr[b * top_k + k];
            if (eid < gpu_expert_count) continue;
            TORCH_CHECK(eid < gpu_expert_count + n_cpu_experts, "expert id ", eid, " out of range");
            cnt[eid - gpu_expert_count]++;
        }
    head[0] = 0;
    for (int e = 0; e < n_cpu_experts; ++e) head[e + 1] = head[e] + cnt[e];
    for (int e = 0; e < n_cpu_experts; ++e) fill[e] = head[e];
    for (int b = 0; b < bs; ++b)
        for (int k = 0; k < top_k; ++k) {
            int64_t eid = idx_ptr[b * top_k + k];
            if (eid < gpu_expert_count) continue;
            int e = (int)(eid - gpu_expert_count);
            lst_b[fill[e]] = b; lst_k[fill[e]] = k; fill[e]++;
        }

    // ---- scratch: 一次最多处理 SWG_CHUNK 个 slot(1024 x 2048 x 4B x2 = 16MB 上限)
    const int SWG_CHUNK = 1024;
    const int chunk = max_slots < SWG_CHUNK ? max_slots : SWG_CHUNK;
    static thread_local std::vector<int32_t> G, U;
    static thread_local std::vector<int8_t> HQ;
    static thread_local std::vector<float> H;
    static thread_local std::vector<float> HS;
    static thread_local const int8_t* xr_a[SWG_CHUNK];
    static thread_local const int8_t* xr_b[SWG_CHUNK];
    static thread_local int bidx[SWG_CHUNK];
    static thread_local float comb[SWG_CHUNK];
    if ((int)G.size() < chunk * n_mid) {
        G.assign((size_t)chunk * n_mid, 0);
        U.assign((size_t)chunk * n_mid, 0);
        HQ.assign((size_t)chunk * n_mid, 0);
        H.assign((size_t)chunk * n_mid, 0.f);
    }
    if ((int)HS.size() < chunk) HS.assign(chunk, 0.f);

    // 🚨🚨 OpenMP x thread_local 的坑(乙-2 首版就是在这里段错误, gdb 里 gp=0x2000):
    //   在 parallel 区【里面】写 G.data(), 每条工作线程会自己去解析 G 这个
    //   static thread_local 符号 => 拿到的是【它自己那份从没分配过的副本】
    //   (主线程分配的那份它根本看不见) => data()==nullptr, 加上 t*n_mid 的偏移
    //   就变成 0x2000 这种野地址。数组型 lst_b[] 之所以没炸: 那边传的一直是
    //   【主线程算好的指针】, 区里没有再碰符号名。
    //   规矩: 并行区里只准用下面这些裸指针, TLS 容器只允许在区外触碰。
    int32_t* gbase  = G.data();
    int32_t* ubase  = U.data();
    int8_t*  hqbase = HQ.data();
    float*   hbase  = H.data();
    float*   hsbase = HS.data();
    const int* bidx_p = bidx;      // 🚨 同一条坑: bidx 也是 thread_local 数组, 区里只准用这个指针

    const int tb = tok_block == 1 ? 1 : (tok_block == 2 ? 2 : (tok_block == 4 ? 4 : 8));

    for (int e = 0; e < n_cpu_experts; ++e) {
        const int n_tok = cnt[e];
        if (n_tok == 0) continue;                  // 零流量专家: 三张权重一个字都不读
        const int8_t* wg_mat = pg + (size_t)e * n_mid * hidden;
        const int8_t* wu_mat = pu + (size_t)e * n_mid * hidden;
        const int8_t* wd_mat = pd + (size_t)e * hidden * n_mid;
        const float s_g = sg_ptr[e], s_u = su_ptr[e], s_d = sd_ptr[e];

        for (int base = 0; base < n_tok; base += chunk) {
            const int n = chunk < n_tok - base ? chunk : (n_tok - base);
            const int he = head[e] + base;
            for (int t = 0; t < n; ++t) {
                const int b = lst_b[he + t];
                xr_a[t] = x_ptr + (size_t)b * hidden;
                bidx[t] = b;
                comb[t] = score_ptr[b * top_k + lst_k[he + t]];
            }
            // ---- 阶段1: g = x@Wg, u = x@Wu (两趟 GEMV, 各读 2MB; x 行留在 L1 里被复用)
            // (TB 用普通 switch 分发 —— 泛型 lambda 里 `constexpr int TB = TBv` 不是常量表达式)
            switch (tb) {
                case 8: swg_stage_a<8, S8>(wg_mat, wu_mat, n_mid, hidden, xr_a, gbase, ubase, n, n_threads, sched); break;
                case 4: swg_stage_a<4, S8>(wg_mat, wu_mat, n_mid, hidden, xr_a, gbase, ubase, n, n_threads, sched); break;
                case 2: swg_stage_a<2, S8>(wg_mat, wu_mat, n_mid, hidden, xr_a, gbase, ubase, n, n_threads, sched); break;
                default: swg_stage_a<1, S8>(wg_mat, wu_mat, n_mid, hidden, xr_a, gbase, ubase, n, n_threads, sched); break;
            }
            // ---- 阶段2: h = silu(g)*u -> 每 slot 对称量化成 int8 (这就是那笔新精度代价)
#pragma omp parallel for schedule(static) num_threads(n_threads) if (n > 1)
            for (int t = 0; t < n; ++t) {
                const float xs = x_scales_ptr[bidx_p[t]];   // 🚨 bidx 走指针, 不能在区里读符号名
                const float ag = xs * s_g, au = xs * s_u;
                const int32_t* gp = gbase + (size_t)t * n_mid;   // 🚨 见上面 TLS 注释
                const int32_t* up = ubase + (size_t)t * n_mid;
                int8_t* hp = hqbase + (size_t)t * n_mid;
                float* hb = hbase + (size_t)t * n_mid;
                float amax = 0.f;
                // 🚨 单趟 expf + 存进 fp32 暂存, 而不是"两趟各算一次 silu":
                //    expf 是标量指令(~30 cyc), 两趟 = 每 slot 每专家多 2048 次 expf
                //    ≈ 15us, 在 bs=1(阶段2 只有 <=2 个 slot 可并行)时那是纯串行白送。
                for (int i = 0; i < n_mid; ++i) {
                    const float g = (float)gp[i] * ag;
                    const float h = g / (1.f + expf(-g)) * ((float)up[i] * au);
                    hb[i] = h;
                    const float a = h < 0.f ? -h : h;
                    if (a > amax) amax = a;
                }
                const float hs = amax > 1e-8f ? amax / 127.f : 1e-8f;
                const float inv = 1.f / hs;
                for (int i = 0; i < n_mid; ++i) {
                    float q = rintf(hb[i] * inv);     // rintf = 半偶舍入 = torch.round 同规则
                    q = q < -127.f ? -127.f : (q > 127.f ? 127.f : q);
                    hp[i] = (int8_t)q;
                }
                hsbase[t] = hs;
            }
            // ---- 阶段3: out[b] += (h@Wd) * (h_scale * wd_scale * 门控分)
            //   直接复用 run_block: 它本来就是"每输出行一次 GEMV + 按 comb 累加到 token",
            //   这里 w_mat=(D,N) 行=输出 j, dot 长度=N, 输出 stride=D(=expert_size 形参位)
            for (int t = 0; t < n; ++t) xr_b[t] = hqbase + (size_t)t * n_mid;
            const float sd_e = s_d;
            for (int t = 0; t < n; ++t) comb[t] *= hsbase[t] * sd_e;
            switch (tb) {
                case 8: run_block<8, S8>(wd_mat, hidden, n_mid, xr_b, bidx, comb, nullptr, out_ptr, n, n_threads, sched); break;
                case 4: run_block<4, S8>(wd_mat, hidden, n_mid, xr_b, bidx, comb, nullptr, out_ptr, n, n_threads, sched); break;
                case 2: run_block<2, S8>(wd_mat, hidden, n_mid, xr_b, bidx, comb, nullptr, out_ptr, n, n_threads, sched); break;
                default: run_block<1, S8>(wd_mat, hidden, n_mid, xr_b, bidx, comb, nullptr, out_ptr, n, n_threads, sched); break;
            }
        }
    }
}

// ============================================================
// 入口 C (乙-2): SwiGLU 三矩阵版, 与入口 B 同一条铁律 —— 【函数体内零 CUDA 调用】,
// 结果写进调用方给的(建议 pinned)CPU 缓冲区, 形状 (bs, hidden)。
// ============================================================
void avx2_moe_cpu_experts_int8_swiglu_into(
    torch::Tensor x, torch::Tensor x_scales, torch::Tensor gate_idx,
    torch::Tensor gate_scores,
    torch::Tensor wq_g, torch::Tensor wq_u, torch::Tensor wq_d,
    torch::Tensor s_g, torch::Tensor s_u, torch::Tensor s_d,
    torch::Tensor out, int64_t gpu_expert_count, bool release_gil,
    int64_t n_threads, int64_t tok_block, int64_t signed_w, int64_t omp_sched) {

    check_cpu(x, "x"); check_cpu(x_scales, "x_scales"); check_cpu(gate_idx, "gate_idx");
    check_cpu(gate_scores, "gate_scores");
    check_cpu(wq_g, "wq_g"); check_cpu(wq_u, "wq_u"); check_cpu(wq_d, "wq_d");
    check_cpu(s_g, "s_g"); check_cpu(s_u, "s_u"); check_cpu(s_d, "s_d");
    check_cpu(out, "out");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(out.dim() == 2 && out.size(1) == wq_d.size(1),
                "swiglu 的输出宽度是 hidden(=", wq_d.size(1), "), 不是 expert_size —— "
                "拿到 (bs, expert_size) 说明 python 侧的 out_dim 没跟着 expert_kind 改");
    // 🚨 见本节开头【数值约定】: biased 打包在第二阶段有两处说不通的地方, 直接挡住
    TORCH_CHECK(want_signed(signed_w, wq_g),
                "expert_kind=swiglu 只支持 signed(int8) 打包: biased 的 maddubs 有 T10 证明的"
                " int16 饱和, 且它的去偏项 128*sum_x 在 h 那一侧要按 slot 而不是按 token 定义"
                " —— 同一 token 两个 slot 撞同一专家时按 token 取 sum 是错的");

    out.zero_();
    float* out_ptr = out.data_ptr<float>();
    auto call = [&] {
        moe_swiglu_impl<true>(x, x_scales, gate_idx, gate_scores,
                              wq_g, wq_u, wq_d, s_g, s_u, s_d, out_ptr,
                              (int)gpu_expert_count, (int)n_threads,
                              (int)tok_block, (int)omp_sched);
    };
    if (release_gil) { py::gil_scoped_release g; call(); } else call();
}

// ============================================================
// 入口 A (老 API, 保留当基线): 内部自己做 .cpu() / .cuda()
// ============================================================
torch::Tensor avx2_moe_cpu_experts_int8(
    torch::Tensor x, torch::Tensor x_scales, torch::Tensor gate_idx,
    torch::Tensor gate_scores, torch::Tensor cpu_w_q, torch::Tensor cpu_w_scales,
    int64_t gpu_expert_count, bool release_gil, int64_t n_threads, int64_t tok_block,
    int64_t signed_w, int64_t loop_order, int64_t omp_sched) {

    auto x_cpu = x.cpu().contiguous();          // 🚨 CUDA 调用 + 阻塞
    auto x_scales_cpu = x_scales.cpu().contiguous();
    auto idx_cpu = gate_idx.cpu().contiguous();
    auto scores_cpu = gate_scores.cpu().contiguous();
    auto w_cpu = cpu_w_q.contiguous();
    auto w_scales_cpu = cpu_w_scales.contiguous();

    const int bs = x_cpu.size(0);
    const int expert_size = cpu_w_q.size(1);
    auto out = torch::zeros({bs, expert_size},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    float* out_ptr = out.data_ptr<float>();
    if (release_gil) {
        py::gil_scoped_release g;   // ✅ L1: 松开 GIL, 主线程才能继续推进
        dispatch_moe(signed_w, w_cpu, x_cpu, x_scales_cpu, idx_cpu, scores_cpu, w_cpu,
                     w_scales_cpu, out_ptr, (int)gpu_expert_count, (int)n_threads,
                     (int)tok_block, (int)loop_order, (int)omp_sched);
    } else {
        dispatch_moe(signed_w, w_cpu, x_cpu, x_scales_cpu, idx_cpu, scores_cpu, w_cpu,
                     w_scales_cpu, out_ptr, (int)gpu_expert_count, (int)n_threads,
                     (int)tok_block, (int)loop_order, (int)omp_sched);
    }
    return out.cuda();                          // 🚨 CUDA 调用 + 隐式同步
}

// ============================================================
// 入口 B (T3 正解): 【函数体内零 CUDA 调用】, 结果写进调用方给的(建议 pinned)缓冲区
// ============================================================
void avx2_moe_cpu_experts_int8_into(
    torch::Tensor x, torch::Tensor x_scales, torch::Tensor gate_idx,
    torch::Tensor gate_scores, torch::Tensor cpu_w_q, torch::Tensor cpu_w_scales,
    torch::Tensor out, int64_t gpu_expert_count, bool release_gil,
    int64_t n_threads, int64_t tok_block, int64_t signed_w, int64_t loop_order, int64_t omp_sched) {

    check_cpu(x, "x"); check_cpu(x_scales, "x_scales"); check_cpu(gate_idx, "gate_idx");
    check_cpu(gate_scores, "gate_scores"); check_cpu(cpu_w_q, "cpu_w_q");
    check_cpu(cpu_w_scales, "cpu_w_scales"); check_cpu(out, "out");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");

    out.zero_();                                // 纯 CPU 操作, 无 CUDA
    float* out_ptr = out.data_ptr<float>();

    if (release_gil) {
        py::gil_scoped_release g;               // ✅ L2: 全程松 GIL
        dispatch_moe(signed_w, cpu_w_q, x, x_scales, gate_idx, gate_scores, cpu_w_q,
                     cpu_w_scales, out_ptr, (int)gpu_expert_count, (int)n_threads,
                     (int)tok_block, (int)loop_order, (int)omp_sched);
    } else {
        dispatch_moe(signed_w, cpu_w_q, x, x_scales, gate_idx, gate_scores, cpu_w_q,
                     cpu_w_scales, out_ptr, (int)gpu_expert_count, (int)n_threads,
                     (int)tok_block, (int)loop_order, (int)omp_sched);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avx2_moe_cpu_experts_int8_swiglu_into", &avx2_moe_cpu_experts_int8_swiglu_into,
          "AVX2 INT8 SwiGLU MoE Experts (3 matrices, pure-CPU kernel, zero CUDA calls)",
          py::arg("x"), py::arg("x_scales"), py::arg("gate_idx"), py::arg("gate_scores"),
          py::arg("wq_g"), py::arg("wq_u"), py::arg("wq_d"),
          py::arg("s_g"), py::arg("s_u"), py::arg("s_d"),
          py::arg("out"), py::arg("gpu_expert_count"), py::arg("release_gil") = false,
          py::arg("n_threads") = 6, py::arg("tok_block") = 8, py::arg("signed_w") = -1,
          py::arg("omp_sched") = 1);
    m.def("avx2_moe_cpu_experts_int8", &avx2_moe_cpu_experts_int8,
          "AVX2 INT8 MoE Experts (legacy: CUDA transfers inside C++)",
          py::arg("x"), py::arg("x_scales"), py::arg("gate_idx"), py::arg("gate_scores"),
          py::arg("cpu_w_q"), py::arg("cpu_w_scales"), py::arg("gpu_expert_count"),
          py::arg("release_gil") = false, py::arg("n_threads") = 1,
          py::arg("tok_block") = 8, py::arg("signed_w") = -1,
          py::arg("loop_order") = 0, py::arg("omp_sched") = 1);
    m.def("avx2_moe_cpu_experts_int8_into", &avx2_moe_cpu_experts_int8_into,
          "AVX2 INT8 MoE Experts (pure-CPU kernel, zero CUDA calls, writes into out)",
          py::arg("x"), py::arg("x_scales"), py::arg("gate_idx"), py::arg("gate_scores"),
          py::arg("cpu_w_q"), py::arg("cpu_w_scales"), py::arg("out"),
          py::arg("gpu_expert_count"), py::arg("release_gil") = false,
          py::arg("n_threads") = 1, py::arg("tok_block") = 8, py::arg("signed_w") = -1,
          py::arg("loop_order") = 0, py::arg("omp_sched") = 1);
}
