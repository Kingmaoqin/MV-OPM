# MV-OPM Round 2 双人独立复审综合报告

**审计日期：** 2026-08-24  
**审计提交：** aaef262；冻结运行 checkpoint：aa983a2  
**总判定：** MAJOR_REVISION_REQUIRED

## 一句话结论

Round 2 的 MSES NOT_SUPPORTED 负面结论仍然稳健，6,820 条任务的当前证据包也完整可重建；但现有代码和稿件不足以支持“已校准的 formal moment test”“公平的 kernel/sieve 候选对比”和“六种候选的结构性 oracle switching”三类表述。

最强可辩护结论应收窄为：在特定 exact-DGP 对齐扰动下，给定的 moment discrepancies 包含桥误差与二阶偏差信息；冻结的 MSES 实现未成为可靠的一般 ATE estimator selector。现有结果不能外推为 MSES 思想本身的一般否定。

## 独立性与证据完整性

两个隔离的 reviewer 分别完成了方法/统计审计和对抗性工程/证据链审计：

- [Reviewer C：方法、统计与算法复审](REVIEW_C_METHODS_REAUDIT.md)
- [Reviewer D：对抗性工程、复现与稿件复审](REVIEW_D_ADVERSARIAL_REAUDIT.md)

交叉复核结果：

- 仓库全部测试 36/36 通过，其中 Round-2 invariant tests 27/27 通过。
- Reviewer D 从全部 6,820 个 result.json 独立重聚合；raw.csv、raw_candidates.csv 和 AGGREGATION_AUDIT.json 的 SHA-256 与存档逐字节一致。
- 6,820 个任务的 config、array index、scientific/split/bootstrap/bank 四类 seed 和 git checkpoint 对 manifest 均为 0 mismatch。
- 未发现伪造、丢行、重复任务或当前存档被错路由的证据。
- 未发现能把 MSES 主要负面判定改成正面的 P0 错误。

以上只证明“这次存档完整”，不代表现有聚合器已 fail closed，也不代表统计检验已校准。

## P1：提交或新一轮实验前必须处理

### 1. Kernel q >= 1 限制排除了仓库自己的真桥

KernelBridges 使用 clip(1 + softplus(g), max=q_max)，强制 q 不小于 1（[kernel_moment.py:169](../../../opm/bridges/kernel_moment.py#L169)）。默认 exact finite-proxy DGP 的 population q 是：

    [[0.149765, 1.393577, 5.450256],
     [5.714657, 1.766135, 0.098341]]

其中 2/6 个单元小于 1，最小值 0.09834；sieve 却允许 [0,50]（[sieve.py:104](../../../opm/bridges/sieve.py#L104)）。这是候选函数类的不对称错配，不是增加 epoch 可以修复的收敛问题。

影响：kernel/sieve 公平性、fixed-kernel 性能及 mixed candidate 的 q 侧解释均受影响。A2 直接使用 exact q，故 A2 机制结论不被这一问题推翻。

### 2. Kernel 的 U-statistic 训练目标有限样本可为负，h 侧可无界向下

训练直接最小化 v' (K-diag(K)) v / [B(B-1)]（[kernel_moment.py:37](../../../opm/bridges/kernel_moment.py#L37)）。对 residuals (a,-a) 和正的 off-diagonal kernel 值 c，该目标等于 -a²c。h 输出无界，因此网络可以沿与满足 population moments 无关的负方向降低目标。AdamW 和 held-out early stopping 只能缓解，不能使目标有下界。

影响：convergence audit 最多证明 300 epochs 比 45 epochs 少欠训练，不能证明 kernel 优化了正确的非负 population norm 或已经收敛。

### 3. OOF multiplier bootstrap 暂不能称为 calibrated formal test

候选 h/q 由训练集高度重叠的 K-fold fits 产生（[oof.py:39](../../../opm/validation/oof.py#L39)），随后代码对汇总后的 observation rows 施加独立 Gaussian multipliers（[moment_tests.py:86](../../../opm/validation/moment_tests.py#L86)）。这没有处理重叠 nuisance fits 的共享训练变动；而 h/q 各自的桥矩对该 nuisance 误差并非 Neyman-orthogonal，一阶估计误差不会因普通 cross-fitting 自动消失。

补充 exact-truth 检查在 100 个种子、199 次 bootstrap 下观察到 0/100 global false rejection。这仅验证“固定 exact truth”实现很保守，不验证“估计后的 OOF candidate”检验尺寸。

影响：p_h、p_q、p_DR、oracle survival、rejection 和 abstention 应暂称 empirical compatibility scores。若保留 formal test 表述，需独立 train/test 检验、refit bootstrap，或有局部稳健矩与证明的 cross-fit 极限理论。

### 4. 同一候选的 bootstrap p 值会随 candidate library 成员变化

test_many_candidates 使用 bootstrap_seed + 10000 × sorted index 分配候选流（[moment_tests.py:142](../../../opm/validation/moment_tests.py#L142)）。只要加入一个字典序更早的候选，原候选的预测和 statistic 不变，p 值却会变化。两位审计均独立复现。

影响：当前方案是 order-deterministic，但不是 library-subset invariant。I2 不同 library 的 abstention 差异混入了 bootstrap Monte Carlo stream 变化。应改用候选名称的稳定哈希或共享 multiplier draws。

### 5. Oracle switching 与 oracle_not_sieve1 子组使用了带结果噪声的 task-level oracle

oracle_best 是每个 simulation seed 上实现绝对误差最小的候选（[core.py:117](../../../opm/experiments/mvopm_round2/core.py#L117)）。报告中的“六候选均会赢、70.4%/62.9% switching”是 per-seed winner instability，不是 regime expected-risk crossing。

按 20 个 R regimes 的 mean ATE risk 重构，只有四个 regime-level winners：kernel_sieve1 12、kernel_kernel 6、sieve1_kernel 1、sieve2_sieve2 1。Mean-risk winner 在八个 n 对比中改变两次，在八个 noise 对比中改变三次。

此外，[analyze_round2.py:213](../../../scripts/analyze_round2.py#L213) 用同一 task 的 realized oracle_best != sieve1_sieve1 定义 oracle_not_sieve1，再与 fixed sieve1 比较。这是按比较器自身实现误差做结果后筛选，不等于 preregistration 所说的“sieve1 不是 oracle 的 regimes”。

影响：基准确有至少四类候选的结构性 switching，但证据强度被明显夸大。改为 regime-level 定义后，MSES 相对 fixed sieve1 的负面均值结论不变。

### 6. D2 与其他 DGP 的 ATE truth 目标不一致

S1/S2/R 使用当前样本的 mean(tau(X))，D2 却固定 ate_true=2.0（[nonlinear_bridge.py:52](../../../opm/dgp/nonlinear_bridge.py#L52)），使 pooled selector metric 混合 sample-conditional ATE 和 population ATE。

审计性地把 D2 改按实现 sample ATE 重算后，MSES 仍严重失败：median oracle ratio 41.55、catastrophic fraction 0.953、最大误差 11.323。因此该问题违反 estimand 一致性，但不挽救 MSES。

### 7. 主分析 pooling、weighting 与 verdict rule 未完全预注册

Preregistration 未明确主 pool 是 B2–E2、B2–E2+R 还是全部 simulation，也未冻结 study/regime 权重、主 gate 区间估计和完全确定的 verdict rule。当前主描述混合 200 个 flagship rows 与 600 个 R rows，使 R 占 75%；R 的 20 个 regimes 又共享 30 个 scientific-seed blocks。

预先承诺的 Holm arm adjustment 与 maximum contrast variance sensitivity 也未实现或报告。当前负面结果并不临界，但这些分析自由度不允许把方案称为完全封闭的 confirmatory statistical analysis plan。

### 8. RHC 区间是同数据选模后的 naive Wald CI

RHC 先用 full-sample OOF outcomes 选择 candidate（[studies.py:170](../../../opm/experiments/mvopm_round2/studies.py#L170)），再对同一选中 candidate 报告 mean ± 1.96SE。区间未包含 candidate selection、overlapping nuisance fits 或 split search 的不确定性；报告 ATE 来自 full-sample adaptive choice，也不是已计算的 nested estimator。

影响：30/30 负向只可说明一个数据集上的算法方向稳定性；“所有 95% CI 排除 0”不能作为选模后因果显著性证据。

### 9. I2 的 library_truth 不是已证明的 bridge-adequacy truth

Manifest 直接把 full easy、sieve1-only nonlinear 与 overregularized kernel 标成 adequate/inadequate，但相应 DGP 没有为这些 candidate classes 提供 exact bridge 或 population approximation-error bound。“false abstention 0”和“true rejection 70%”因此只是相对人工 scenario label 的指标，应改称 scenario-specific abstention。

I2 的 70% 还是“任一 outer fold abstains 即整 task abstains”的 nested 口径；同一 overregularized cell 的 full-sample deployment abstention 是 58%。两者不应混称一个部署拒绝率。

## P2：工程、测试与措辞问题

### 聚合与 resume 未 fail closed

[aggregate_round2.py:37](../../../scripts/cluster/aggregate_round2.py#L37) 只检查 path 存在且 status=ok，再用 manifest 标签结果；它没有断言 result 内 config、array id、四类 seed 和 checkpoint 与 manifest 相等。[launch.py:23](../../../scripts/cluster/launch.py#L23) 的 resume 也会跳过任意 path 上的 status=ok 文件。当前 corpus 已通过独立全量核对，但未来 stale/cross-copied result 可能被静默误标。

### 环境复现性不完整

pyproject.toml 对多数依赖只给下界或不限定版本；worker 未记录包版本、CUDA/BLAS、git dirty state 和 manifest hash。同一 commit 未必能在未来重建数值一致环境。

### Result 写入不是原子的，重复 worker 可能竞争

Worker 直接覆盖 canonical result.json，进程中断可能留下半写文件，两个相同 task worker 也可能同时归档和覆盖。应在同目录临时文件中写入，flush/fsync 后 atomic rename，并使用 per-row lock 和内容哈希。当前 corpus 未发现这类损坏，但框架没有主动防护。

### ObservedDatasetView 不是真正 read-only，meta 也不是 allowlist

[views.py:24](../../../opm/data/views.py#L24) 保留原 NumPy arrays 的可写引用，meta 仅排除一组已知名称。当前 selector 未发现读取 oracle truth，但文稿中“物理隐藏、read-only”的强表述超过代码保证。应使用不可写 copies/views 与元数据 allowlist。

### 其他分析口径

- “Catastrophic”仅表示 selected_error > 1.5 × oracle_error。在 335 个 core/switching flags 中，153 个 selected absolute error 小于 0.05，204 个小于 0.10。应改称 relative >1.5× oracle failure 并同时报绝对 tail。
- Candidate 与 nested-MSES PEHE 在同一批 X 上拟合 Stage-2 head 后立即评估，属于样本内 secondary diagnostic，不是 out-of-sample CATE risk。
- A2 在 25 个 cells 中复用同一组 200 seed labels；5,000 rows 不是 5,000 个独立 replications。
- Convergence audit 的 300-epoch 预测与 180-epoch 仍未达到冻结的稳定阈值；选 300 是合规的 cap 决定，不是“已收敛”证明。
- F2 改变 c_U 时同时改变 hidden-confounding 与 treatment overlap/positivity；差异不能全归因于单一混杂维度。
- G2 severity=0 基线在四类 corruption 中重复四次；pooled summary 应去重或按 seed/cell 分块。
- Sieve2 的 q 在 767/1,520 个 candidate rows 达到硬上限 50，sieve3 为 1,145/1,520；sieve3 全部 1,520 rows 的 q-balance error 大于 0.1，最大 pseudo-outcome variance 达 2.81e8。“六候选都赢过”包含这些数值病态候选的少量 noisy wins，不能解释为六种科学上有用的结构区间。
- summary_selectors.csv 的独立重构仅在约 1e-16 级标准差尾数上不同；若声称 byte reproducibility，应固定数值栈并按文档精度舍入，否则应称 numerical reproducibility。

## 最新稿件逐份审计

### FINAL_REPORT.md

NOT_SUPPORTED 头条和主要数字可重建，应保留。必须修改：

1. 将“六候选结构性 switching”改成“四个 regime-level mean-risk winners；六个候选至少赢过一个带噪 replicate”。
2. 将 formal screening/p-values 改为 empirical moment compatibility，除非补齐 nuisance-aware 校准。
3. 将“公平收敛”改成“按冻结规则把 kernel cap 提到 300，但未证明收敛，且 q 类与训练损失有结构性问题”。
4. 重写 oracle_not_sieve1、RHC CI、I2 truth/abstention 和 catastrophic 命名。
5. 将“held-out moments measure bridge error”限定到 exact aligned-perturbation DGP 与预注册扰动方向。
6. 披露 D2 estimand 不一致、缺失的 Holm/max-variance sensitivities 以及主权重的描述性。

### ROUND2_PREREG.md

它确实在 final seeds 前冻结了方法、seeds 和广义成功条件；但 pooling/weighting、estimand table、主区间、确定 verdict rule 与两个 sensitivity 未封闭。应保留冻结原文，以 deviation/addendum 披露，不能回溯改写。

### EXACT_DGP_BIAS_DERIVATION.md

对齐二阶偏差的符号与 Gauss-Hermite 常数正确。需要补充：推导只保证 q 扰动因子为正，并不证明 learned kernel q 的 q>=1 参数化正确。

### CATE_SELECTION_ANALYSIS.md

这是当前边界最谨慎的稿件，应保留“未证明、未提升为确认性方法”的结论。需追加 Stage-2 PEHE 样本内评估，以及 pairwise relative-risk SE 未包含重叠 nuisance-fit 变动的警告。

### LITERATURE_BOUNDARY.md

文件正确禁止 first 声称，但仍是 pre-result 条件成功叙事，并遗漏最接近的 conditional-moment specification 文献：[2020 KCM/MMR 检验](https://proceedings.mlr.press/v124/muandet20a.html)，以及 2026-07-27 提出 Neyman-orthogonal moments、cross-fitting 与 multiplier bootstrap 的 [locally robust kernel specification test](https://arxiv.org/abs/2607.24382)。后者正好说明“cross-fitting + 普通 multiplier bootstrap”本身不足以自动消除 ML nuisance 的一阶影响。文献稿应改成 post-result boundary。

### RHC 数据来源与 proxy 定义

RHC loader 顶部仍保留“VERIFY against the published paper”标记，而稿件把 V=(pafi1,paco21)、W=(ph1,hema1) 称为已确认的真实 proxies。公开前应补齐论文精确页码/表格、预处理出处、数据 checksum 与列字典；否则把 RHC 降级为 provisional real-data demonstration。

### 三份中文长稿

- 技术全志_1 和 两天历程 把逐点条件偏差用全局 L2 桥误差乘积约束。逐点 Cauchy–Schwarz 需要给定 X 的 conditional L2 norms；全局 L2 乘积一般只控制积分 ATE bias，不能在缺少 uniform/conditional assumptions 时约束任意 x 的 CATE bias。
- 技术全志_1 又从 orthogonality 直接推出 ATE/CATE CI 渐近名义覆盖；这仍需要 nuisance rates、smoothing bias、fold dependence 和有效方差证明，尤其不能自动推出 CATE 区间有效。
- 技术全志_2 仍写 final seeds 未运行并以探索性 biasvar 近 oracle 结束；两天历程 仍把未来确认为下一步。

三份文档实质上是 2026-08-11 的 Round-1/Pilot 历史记录，却使用“全志/完整结果”标题。应加“冻结的 pre-Round-2 历史版，已被 Round-2 NOT_SUPPORTED 报告补充/取代”标幅，或追加 Round-2 终章；数学错误另做勘误。

## 现有测试没有覆盖的关键问题

36/36 通过不等于代码与统计设计通过。必须新增：

1. 完整 fitted-candidate moment-test pipeline 的 null size/power。
2. Exact q 在每个 solver support 中的可表达性。
3. Kernel 训练目标非负/有下界，或当前负方向的 expected-failure test。
4. 同一 candidate 的 bootstrap stream 对 library composition 不变。
5. 所有 DGP 遵守同一 ate_true 约定。
6. Aggregator/resume 对 manifest/config/index/seeds/checkpoint 不等的 fail-closed tests。
7. Candidate-order invariance 同时比较 h 和 q；当前只比 h。
8. Out-of-sample Stage-2 CATE-head 评估。
9. Full-deployment 与 nested-any-fold abstention 语义分离。
10. ObservedDatasetView 数组不可写与 metadata allowlist。

## 仍可保留的科学结论

1. 当前 6,820-row corpus 完整、可重建并与 manifest 一致。
2. 在 exact finite-proxy DGP 和预注册 aligned bounded perturbation 下，moment discrepancy 跟踪桥扰动与预测二阶 bias。
3. 在冻结代码、library、screen 和评估设计下，MSES 未达到 near-oracle 标准；D2 estimand sensitivity 不改变这一点。
4. R 矩阵中 MSES 相对 fixed kernel 的 mean gain 约 0.01274，seed-block 95% CI [0.00933, 0.01546]；这是较窄的自适应优势，不是全局 near-oracle 成功。

## 处置建议

1. 永久保留 aa983a2 raw corpus、当前分析和 NOT_SUPPORTED 判定；不在 final seeds 上重调方法。
2. 不需重跑即可修正文稿：switching、relative-failure 命名、RHC post-selection CI、I2 口径、D2 estimand deviation、缺失 sensitivity、历史文档版本标幅与数学勘误。
3. 必须改代码并使用新 preregistration/新 seeds：q 参数化、kernel 训练目标、nuisance-aware moment test、candidate-stable bootstrap stream、一致 ATE estimand 和有真值的 inadequacy benchmark。
4. 工程加固：aggregator/resume fail closed，固定依赖并记录环境/dirty/manifest hash，强化 observed-data immutability。

本次复审没有修改科学代码、原始结果、冻结预注册或最终判定；只新增独立审计文档。
