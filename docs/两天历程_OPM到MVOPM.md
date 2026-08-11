# 两天历程:从 OPM v2 到 MV-OPM —— 设计、数学、代码、实验、结果与反思

> 这是一份供**思考**用的完整叙事文档。它把两天里做的所有实验和尝试串起来:每一步的**设计动机、
> 关键数学、核心代码、实验细节、真实结果**,以及每一次**诚实的转向和失败**。所有数字均来自真实运行,
> 可从 `results/` 下的原始文件重建。配套细节见:`DEEP_DIVE.md`(数学+代码)、`FULL_REPORT.md`(E1-E10)、
> `docs/mvopm/*`(MV-OPM)、`docs/reviews/*`(评审+门禁)。

**一句话主线**:
> 我们先造了一个"多治疗近端双重稳健 CATE 学习器"(OPM v2),把它做扎实并诚实评估后,发现它的真正价值
> 不在"更强的桥",而在"proximal 识别去混杂偏差"。这引出一个更本质的问题——**在没有反事实真值时,能不能用
> 观测数据判断哪个近端估计器可信?** 于是转向 MV-OPM(用留出集识别矩违背来选择/证伪近端估计器)。诚实的
> pilot 给出 HOLD,但同时暴露了一个理论一致、可救活的方向(偏差×方差),并在全新种子上把它去了风险。

---

## 目录
- [第一部分:起点——OPM v2 的方法与数学](#一)
- [第二部分:OPM v2 的十个实验(E1-E10)——设计/结果/诚实发现](#二)
- [第三部分:转折——为什么从"更好的学习器"转向 MV-OPM](#三)
- [第四部分:MV-OPM 的设计与实现(数学+代码)](#四)
- [第五部分:MV-OPM 的实验与结果(机制/选择/真实数据)](#五)
- [第六部分:HOLD、失败分析与探索性验证](#六)
- [第七部分:供思考的结论、教训与下一步](#七)

---
<a name="一"></a>
## 第一部分:起点——OPM v2 的方法与数学

### 1.1 要解决的问题:未观测混杂下的 CATE

标准 CATE 估计假设**可忽略性**:给定观测 X,处理 T 与潜在结果独立。现实里常有**未观测混杂 U**——同时影响
处理和结果,使可忽略性失效。此时任何"只调整 X"的方法都有**系统性偏差,且数据再多也消不掉**。

**近端因果推断**(Miao 2018;Cui 2023)用两类代理绕过 U:
- **结果诱导代理 W**:与 U 相关,给定 (U,X,T) 后不直接影响 Y,不受 T 影响。
- **处理诱导代理 V**:与 U 相关,给定 (U,X,T) 后不直接影响 Y。

### 1.2 两个桥函数与识别定理

对每个臂 k 定义**结果桥** `h_k(W,X)` 与**处理桥** `q_k(V,X)`,分别解条件矩方程:
```
(B1)  E[ Y − h_k(W,X) | V,X,T=t_k ] = 0
(B2)  E[ 1{T=t_k} q_k(V,X) − 1 | W,X ] = 0
```
**双重稳健伪结果**(STAR):`φ_k = 1{T=t_k} q_k(Y − h_k) + h_k`。

**定理 1(识别)**:若 h 解 B1 *或* q 解 B2(任一),则 `E[φ_k|X]=μ_k(X)=E[Y(t_k)|X]`,故
`τ_k(x)=E[φ_k−φ_0|X=x]`。**证明思路**(展示双重稳健):
- 若 h 对:`E[1{T=k}q_k(Y−h_k)|V,X]=q_k·P(T=k|V,X)·E[Y−h_k|V,X,T=k]=0`(B1)→ `E[φ_k|X]=E[h_k|X]=μ_k`。
- 若 q 对:处理桥识别给 `E[1{T=k}q_k Y|X]=μ_k`;而 `E[1{T=k}q_k h_k|X]=E[h_k E(1{T=k}q_k|W,X)|X]=E[h_k|X]`(B2),与 `+h_k` 项相消 → μ_k。

**定理 2(率双重稳健)**:两桥都估计时,记 `δ_h=ĥ−h, δ_q=q̂−q`。逐项算 `E[φ_k(ĥ,q̂)|X]−μ_k`:
- h 的一阶项 `E[1{T=k}q δ_h|X]=E[δ_h|X]` 与 `+E[δ_h|X]` 抵消 → 0(**Neyman 正交**);
- q 的一阶项 `E[1{T=k}δ_q(Y−h)|X]=0`(B1)→ 0;
- 剩 `−E[1{T=k}δ_q δ_h|X]`,由 Cauchy-Schwarz `≤ ‖δ_h‖·‖δ_q‖`。

**→ 条件偏差 ≤ 两桥 L2 误差之积。** 这个"乘积结构"贯穿一切:E2 验证它、E6(f) 展示混杂增强时的增益、
E10 覆盖率随 n 趋名义、最后成为 MV-OPM 的核心动机(**product 分数**)。

### 1.3 kernel-moment 桥:为什么不需要对抗

B1 是条件矩 `E[R|Z]=0`(R=1{T=k}(Y−h_k), Z=(V,X)),等价于对丰富函数类 `E[R f(Z)]=0`。取 RKHS 单位球,
最大矩限制目标有**闭式**:
```
J(h) = sup_{‖f‖≤1}(E[R f(Z)])² = E_{i,j 独立}[ R_i R_j k(Z_i,Z_j) ]
```
无需训练判别器 f。**无偏经验估计**是排除对角的 U 统计量(排除 i=j 才无偏),核心恒等式:
```
Σ_{i≠j} R_i R_j K_ij = Rᵀ(K − diag K)R
```
对应代码 `opm/bridges/kernel_moment.py`:
```python
def ustat_quadratic(vals, Kmat):
    Kz = Kmat - torch.diag(torch.diag(Kmat))   # K − diag(K)
    return (vals @ (Kz @ vals)) / (B*(B-1))     # = Σ_{i≠j} R_i R_j K_ij / (B(B-1))
```
q 参数化 `q_k=clip(1+softplus(g_k),max=50)` 保证 q≥1、有界。网络:共享主干 3×128 + 臂嵌入;中位数启发式带宽;
EMA(0.99);RFF 残差诊断早停。**这个 RFF 残差诊断,后来成了 MV-OPM 验证器的原型。**

### 1.4 交叉拟合 + Stage-2 + 影响函数 CI
5 折交叉拟合(leakage-proof)拼装 OOF `φ`;`D=φ_{1:}−φ_0` 回归到 X(MLP 3×256)得 τ̂;
ATE 影响函数 `ψ=φ_k−φ_0`,`CI=ATE±1.96·sd(ψ)/√n`。无代理时退化为 AIPW(`dr_fallback`,含 Hájek 稳定化)。

---
<a name="二"></a>
## 第二部分:OPM v2 的十个实验——设计/结果/诚实发现

> 里程碑纪律:**M1 硬门禁**(E1 必须先过)→ 基准 → 消融 → 生成器。所有实验写 REPORT.md 逐条 PASS/FAIL。

| 实验 | 设计 | 关键结果 | 判定 |
|---|---|---|---|
| **E1** 一致性/率 | 线性高斯闭式桥,n∈{500..16000}×10 seed | ATE 偏差单调→0.002(Spearman **−1.0**);h 探针 R²=**0.974**;覆盖 0.875 | ✅ 5/5(硬门禁) |
| **E2** 乘积双稳健 | 已知率腐蚀桥,拟合 log\|bias\| vs log n 斜率 | DR 性质确证(单桥腐蚀偏差递减);斜率 4/9 精确 | ✅ 3/3(诚实软判据) |
| **E3** 主基准 | S1/S2 + 腐蚀,10 seed,全基线 | **OPM 击败全部 8 个可忽略性基线**(BH<0.05);OPM 0.525 vs sieve-PDR 0.490 | ✅ 3/3 |
| **E4** RHC 真实 | 真实代理→完整 proximal;线性-POR 门禁 | POR **−1.94 天**,OPM **−1.32 天**,CI 重叠,En[1q]=0.998/0.972 | ✅ 3/3 |
| **E5** 半合成 | HAMD 67 维真实协变量 + 已知真值 | OPM 0.603 击败全部可忽略性(n=1892) | ✅ 2/2 |
| **E6** 消融 | (a-f) | **(f) c_U 扫描:fallback ATE 误差 0.14→0.57,proximal 优势随混杂扩大** | ✅ 3/3 |
| **E7** 分布反事实 | DDPM+蒸馏 vs factual-DDPM | 蒸馏均值偏差未 <50% factual | ❌ 0/1(可选模块,诚实记录) |
| **E8** 诊断有效性 | 矩残差/代理强度 vs PEHE 相关 | bridge_resid Spearman **+0.6**;rho_min 较噪 | ✅ 2/2 |
| **E9** 非线性混杂 | 强非线性多维代理 | **OPM ATE 误差 0.14 vs 可忽略性 ~0.55(4×)**;但 PEHE 0.56 vs **sieve1 0.44(sieve 更好)** | ✅ 2/2(ATE 判据) |
| **E10** 有效推断 | 局部线性 CATE CI,n 扫描 | 覆盖 **0.70→0.90→0.967** 趋名义 | ✅ 3/3 |

**这一部分最重要的三个诚实发现**(它们直接催生了 MV-OPM):
1. **E3/E5/E6a/E9:kernel 桥并不优于简单 sieve**——线性 sieve1 常追平或反超(E9 PEHE:sieve1 0.44 < kernel 0.56)。
2. **E9:proximal 的干净胜点在偏差/ATE(4×),不在 CATE/PEHE**——kernel 桥的方差抵消了低偏差。
3. **E9:高阶 sieve 不稳**(sieve2/3 PEHE 1.24、2.38 爆炸)。

→ **没有单一 solver 主导**。这不是尴尬,而是"**为什么需要验证/选择**"的最强动机。

---
<a name="三"></a>
## 第三部分:转折——为什么从"更好的学习器"转向 MV-OPM

诚实定位(`docs/POSITIONING.md`)承认:**kernel 桥不是卖点,proximal 识别框架 + 去偏才是**。但更深一层:
既然桥的灵活度不决定因果可靠性(E9),而真值 τ 又不可得(真实数据没有 PEHE),那么——

> **关键科学问题:在没有反事实真值时,能不能用观测数据判断哪个近端估计器可信?**

MV-OPM 的赌注:**留出集上的识别矩违背**(B1/B2 在没用于拟合该候选的数据上被违背多少)是一个**观测数据**信号,
且由定理 2,它应与桥误差(进而与因果偏差)相关。于是:
> **用 h/q 侧的留出矩违背 `D_h, D_q` 诊断、选择、并在必要时拒绝候选近端估计器——不用任何反事实标签。**
> 主分数(预注册):**product = D_h · D_q**(因为 DR 余项是乘积结构)。

**Phase 0 审计**(`docs/mvopm/PHASE0_AUDIT.md`)发现:桥矩残差诊断**已存在**(E8 就是单候选原型),但所有"选择相关"的
护栏都缺失——需新建**共享特征 bank 的 solver 无关验证器、嵌套交叉拟合、oracle 隔离、精确桥 DGP**。

---
<a name="四"></a>
## 第四部分:MV-OPM 的设计与实现(数学+代码)

### 4.1 候选库(含混合候选)—— `opm/candidates/library.py`
6 个候选:`kernel_kernel, sieve1_sieve1, sieve2_sieve2, sieve3_sieve3, kernel_sieve1, sieve1_kernel`。
效率关键:**每个唯一 solver 只拟合一次**,候选是缓存 solver 预测的组合。顺序固定(按名排序)保证候选顺序不影响结果。
**一个被测试捕获的真 bug**:kernel 网络初始化用 torch 全局 RNG → 同进程内顺序拟合不可复现。修:
```python
for si, sname in enumerate(needed):
    ssd = self.seed + 17*si + 1
    torch.manual_seed(ssd); np.random.seed(ssd)     # 隔离全局 RNG → 复现 + 顺序不变
    self._solvers[sname] = _make_solver(sname, ...).fit(view)
```

### 4.2 solver 无关矩验证器 —— `opm/validation/moments.py`
残差 `R_k=1{T=k}(Y−h_k)`,`S_k=1{T=k}q_k−1`。**全候选共享同一个固定特征 bank**(RFF 权重/标准化/带宽,由留出集
构建,种子与候选无关)。**残差按臂内 sd 归一化**(记录在案),使 h、q 两侧无量纲、可比、乘积有意义:
```
D_h_L2 = sqrt( mean_j ( E_n[ R/sd(R) · g_j(z) ] )² )   （z=(V,X) 的 RFF）
```
还实现 RFF-max 与 kernel/MMR 变体。**这个"归一化"是后来失败的关键**(见第六部分)。

### 4.3 选择器 —— `opm/validation/selector.py`
预注册分数(低=好):`product=log(D_h)+log(D_q)`(主)、`h_only`、`q_only`、`sum`、`max`。竞争者:
fixed_kernel/fixed_sieve1/random/ess/qbalance/oracle。**看到结果后绝不改主分数、绝不新造确认性分数。**

### 4.4 嵌套交叉拟合 —— `opm/estimator/nested_crossfit.py`
选择引入自适应,单层 OOF 不够。**外折**留出 OUTER_TEST;**内折**在 OUTER_TRAIN 内拟合候选、在 INNER_VALID 上算矩、
选候选;选中者在**全 OUTER_TRAIN** 重拟合、预测 OUTER_TEST → 外层 OOF 伪结果 → 全部外折后才拟合 τ 头。
**每折索引审计**:inner∩=∅、inner⊂outer_train、outer_test∉inner。

### 4.5 oracle 物理隔离 —— `opm/data/views.py`
`ObservedDatasetView` 只暴露 X/T/Y/W/V/K,读 `tau_true` 等**抛 AttributeError**(连 meta 走私也剥掉)。
验证器/选择器包**不 import** `opm.eval.oracle`(测试断言)。

### 4.6 精确桥 DGP —— `opm/dgp/finite_proxy_exact.py`(取代 E2 的债务)
有限离散 U + 满秩代理通道,**两个真桥都从总体线性方程解出**:
- 结果桥:`M_t h̃_t = r_t`,`M_t[v,w]=Σ_u B_W[u,w] P(U=u|V=v,T=t)`,`r_t[v]=Σ_u P(U|V,T) c_U f_U`。
- 处理桥:`N_t q_t = 1`,`N_t[w,v]=Σ_u B_V[u,v] Pt[u,t] P(U=u|W=w)`。
**数值验证**(n=2e5):`E[Y−h_t|V,T=t]` max\|resid\|≈0.007-0.015,`E[1{T=t}q_t|W]∈[0.994,1.007]`,
STAR-ATE=**2.01**。这给了一个**真 q_true**,可干净测乘积机制。

### 4.7 测试与集群
- 12 个科学不变量测试(oracle 隔离、嵌套索引、共享 bank、矩公式、product 公式、候选顺序不变、确定性)+
  **对抗 bug 注入**(泄漏/走私真值被检测)。**21/21 全过;测试捕获并逼出了确定性 bug 和 segfault。**
- 无 Slurm → 本地 manifest/worker(单行=单任务,带 git/host/seed/runtime/exit 溯源)/launcher(限并发+限线程)/
  aggregator(完整性检查)。**两遍隔离评审 + 交叉核对**(agent 撞额度→自审 fallback,已记录),**门禁看结果前冻结**。

---
<a name="五"></a>
## 第五部分:MV-OPM 的实验与结果

**冻结的 Go/No-Go 门禁**(主目标=**ATE 误差**,因矩测偏差):C1 排序 Spearman(product,ATEerr)>0.2 在 ≥3/4 家族;
C2 product oracle-ratio < fixed-kernel;C4 机制成立。

### Study A —— 精确桥乘积机制(180 行)
```
Spearman(D_h, ‖ĥ−h‖) = 0.86      Spearman(D_q, ‖q̂−q‖) = 0.90
Spearman(moment_product, true_err_product) = 0.83
```
**→ 留出矩差是有效的"桥偏差"度量(强确认)。** 但 `Spearman(|ATE bias|, moment_product)=0.13`——因为腐蚀率下
ATE 偏差埋在 MC 噪声里(设计局限,非机制失败)。

### Study B-E 选择(S1/S2/nonlinear/HAMD,各 10-12 seed)
- **C2 强 PASS**:product oracle-ratio(ATE)≈**1.2-1.5** vs fixed-kernel **3-14**(kernel 灾难);product 也胜
  h_only(1.6-20)/q_only(1.3-6.4)/random。
- **C1 FAIL**:逐实例 Spearman(product,ATEerr)=S1 0.13/S2 0.10/nonlinear 0.07/HAMD 0.45(仅 1/4 >0.2)。

### Study J —— RHC 真实数据(25 分裂)
**product 选择器 100% 选中 sieve1**;ATE=**−2.07 天**(sd 0.70,中位 CI [−2.93,−0.97]),ESS 2104,max_q 18.7
——契合文献(RHC 缩短生存)、稳定、权重有界。**修了一个 segfault**:sieve 在 67 维上 degree-3 展开≈57000 特征→
BLAS 崩溃;加了特征数护栏 + 高维用非爆炸候选集。

**门禁裁决:HOLD。**(C1 失败;主张"近 oracle 选择"不成立。)

---
<a name="六"></a>
## 第六部分:HOLD、失败分析与探索性验证

### 6.1 根因(定论,非功耗问题)—— 方差盲点
矩测**偏差**(`E[R|Z]=0`),残差按 sd 归一化 → **对方差盲**。过灵活、低偏差-高方差的候选满足留出矩却灾难。
nonlinear 每候选实测:
| 候选 | D_h | D_q | product | **ATE 误差** | ATE CI 宽度 |
|---|---|---|---|---|---|
| sieve1_sieve1(oracle-best) | 0.037 | 0.063 | 0.0023 | **0.20** | 0.75 |
| **sieve2_sieve2** | 0.034 | 0.070 | **0.0024** | **5.10** | 27.8 |
| **sieve3_sieve3** | 0.036 | 0.093 | 0.0033 | **11.22** | 35.6 |
| kernel_kernel | 0.188 | 0.085 | 0.016 | 0.95 | 0.45 |

**sieve2/3 矩最低(最"好")却 ATE 误差 5/11(最坏)**——product 分不开它们与 oracle-best。

### 6.2 探索性(post-hoc,非确认)—— 偏差×方差
CATE 误差 ≈ 偏差 + 方差。矩测偏差,**ATE CI 宽度/伪结果方差**是观测数据的方差代理(见上表:sieve2/3 宽度 28-36)。
在**全新种子 500-514**(与推导数据不相交)上:
| 分数 | 中位 oracle-ratio(ATE) | 平均 Spearman |
|---|---|---|
| product(预注册,只偏差) | 1.26 | 0.23 |
| **biasvar(偏差×方差)** | **1.04** | **0.73** |
| variance_only | 混杂(nonlinear/HAMD 崩) | 0.11-0.51 |
| fixed_sieve1(强基线) | 1.03 | 0.41 |

逐场景 Spearman 升到 **0.70/0.73/0.75/0.73**(全 ≥0.70),oracle-ratio 降到 ~1.04(**近 oracle**)。
**是"偏差+方差"的组合起作用,方差单独不行**——与理论一致。**但这是 post-hoc,不能当确认性**;它只把预注册确认
研究**去了大半风险**。诚实警告:测过的场景里 sieve1 几乎总是 oracle-best,要证明数据驱动选择器优于"固定选 sieve1",
确认研究必须含 **oracle-best 会变化的场景**。

---
<a name="七"></a>
## 第七部分:供思考的结论、教训与下一步

### 7.1 最强的、被证据支持的主张(分层)
1. **确认**:留出集近端识别矩违背是**无需反事实标签、可靠的"桥偏差"度量**——一个真正的桥充分性/证伪工具
   (机制 Spearman 0.83-0.90)。
2. **确认(pilot)**:单靠它(product,只测偏差)**选不出可靠 CATE 估计器**,因为它对方差盲。
3. **探索性(fresh-seed 去风险,非确认)**:**矩充分性 + 估计方差 ≈ 近 oracle 选择**(Spearman 0.73)。
4. **贡献类型**:一个经验的**模型验证/证伪框架**,而非"更强的桥"或"近 oracle 选择器"——诚实划界。

### 7.2 两天里最值得反思的三件事
- **诚实的定位胜过漂亮的主张**:E9 揭示 kernel 不优于 sieve、干净胜点在偏差不在 CATE。承认这点,反而找到了
  更本质的问题(验证/选择),也避免了会被自己数据打脸的"更强学习器"主张。
- **纪律产生可信度**:门禁看结果前冻结 → 诚实 HOLD;没为通过调参/藏 sieve 胜出/造 DGP/把 post-hoc 分数包装成
  确认性。**HOLD 本身是有价值的科学结果**,加上失败分析,比一个凑出来的 PASS 更有说服力。
- **失败里藏着理论一致的救活路径**:方差盲点 → 偏差×方差,是"误差=偏差+方差"的直接推论,fresh-seed 上强验证。

### 7.3 下一步(若继续)
写一份**预注册文档**冻结:主分数=`biasvar`(偏差×方差的具体形式)、**含 oracle-best 变化的场景设计**、全新种子库、
判据、多重比较控制;然后在与所有历史数据不相交的种子上跑**确认性**研究。若成立,主张升级为"矩充分性+方差信号可
实现近 oracle 的近端估计器选择";若不成立,退回"矩=有效桥偏差诊断"这个已确认的、较窄但扎实的贡献。

### 7.4 交付物索引(`/home/xqin5/进化的opm/`)
- **OPM v2**:`FULL_REPORT.md`、`DEEP_DIVE.md`、`results/{E1..E10}/REPORT.md`、`results/SUMMARY.md`、`POSITIONING.md`。
- **MV-OPM**:`docs/mvopm/PHASE0_AUDIT.md`、`FINAL_ASSESSMENT_MVOPM.md`;`docs/reviews/*`(A/B+交叉+冻结门禁);
  `results/mvopm/GATE_REPORT.md`、`FAILURE_ANALYSIS.md`、`EXPLORATORY_BIASVAR.md`;419+60 个带溯源原始运行 +
  `raw.csv`/`raw_candidates.csv`/`explore_candidates.csv`。
- **代码**:`opm/{candidates,validation}/`、`opm/estimator/nested_crossfit.py`、`opm/dgp/finite_proxy_exact.py`、
  `opm/data/views.py`、`scripts/cluster/*`;测试 `tests/test_mvopm_invariants.py`(+ 原 9 个)。全部可从原始文件重建。

---
*一切诚实:确认的说确认,探索的标探索,失败的写失败。乘积机制是真的(矩测偏差),product 选择器 HOLD(方差盲),
偏差×方差是去了风险但未确认的方向。这份文档留给你判断:是把"矩=桥偏差诊断"这个扎实的窄贡献写出去,还是投入一次
预注册确认研究去搏"矩+方差的近端估计器选择"这个更大的主张。*
