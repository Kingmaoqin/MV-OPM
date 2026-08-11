# OPM v2 深度技术文档:算法 · 数学 · 代码

本文档在 `FULL_REPORT.md`（实验总览）之上,把**每个算法组件的数学推导、精确实现代码、以及两者的
逐段对应**讲透。阅读顺序建议:先本文档（懂"方法为什么对、代码怎么实现"）,再 `FULL_REPORT.md`
（看"实验证据"）。

记号约定:`O=(X,T,Y,W,V)`;`1{·}` 指示函数;`E_n[·]` 样本均值;`R_i, S_i` 为桥的矩残差;
`φ_k` 为伪结果;`τ_k=μ_k−μ_0` 为 CATE。所有代码块均逐字摘自当前源码。

---

## 目录
- [Part 1 数学基础:proximal 识别与双重稳健](#part-1)
- [Part 2 Stage-1:kernel-moment 桥（含 MMR 推导与训练代码）](#part-2)
- [Part 3 Stage-2:交叉拟合 · 伪结果 · CATE 头 · ATE 置信区间](#part-3)
- [Part 4 dr_fallback:插件 AIPW + Hájek 稳定化](#part-4)
- [Part 5 sieve 桥:两阶段最小二乘（2SLS）推导与代码](#part-5)
- [Part 6 CATE 逐点推断:局部线性 + 三明治方差（E10）](#part-6)
- [Part 7 Stage-3:条件 DDPM + 均值蒸馏（E7）](#part-7)

---
<a name="part-1"></a>
## Part 1 数学基础:proximal 识别与双重稳健

### 1.1 两个桥方程

在潜在可忽略性 `Y(t)⟂T|(U,X)` 下,若能"代理"未观测混杂 U,即可识别 `μ_k(x)=E[Y(t_k)|X=x]`。
proximal 引入两类代理 W（结果诱导）、V（处理诱导）,并定义两个**桥函数**:

- **结果桥** `h_k(W,X)`,解条件矩方程 (B1):
  ```
  E[ Y − h_k(W,X) | V, X, T=t_k ] = 0
  ```
- **处理桥** `q_k(V,X)`,解 (B2):
  ```
  E[ 1{T=t_k}·q_k(V,X) − 1 | W, X ] = 0
  ```

直觉:`h_k` 用 W 把"给定 (V,X,T=k) 的结果条件期望"表出;`q_k` 是近端版逆倾向权重,使得
加权后的处理指示在 (W,X) 上"配平"为 1。**完备性假设 A5**（W、V 对 U 足够丰富）保证这两个积分方程
有解。

### 1.2 定理 1（识别）:伪结果的条件期望就是 μ_k

定义**双重稳健伪结果** (STAR):
```
φ_k(O) = 1{T=t_k}·q_k(V,X)·( Y − h_k(W,X) ) + h_k(W,X)
```
**命题**:若 h_k 解 (B1) *或* q_k 解 (B2)（任一即可）,则 `E[φ_k|X] = μ_k(X)`。

**证明（population,展示双重稳健）**。记 `δ` 项为 0 的两种情形:

**(i) h 正确（解 B1）**。看修正项,对 (V,X) 取条件:
```
E[ 1{T=k} q_k(V,X)(Y−h_k) | V,X ]
  = q_k(V,X)·E[ 1{T=k}(Y−h_k) | V,X ]
  = q_k(V,X)·P(T=k|V,X)·E[ Y−h_k | V,X,T=k ]
  = q_k(V,X)·P(T=k|V,X)·0  = 0      （由 B1）
```
故 `E[φ_k|X] = E[h_k(W,X)|X]`,而结果桥识别（POR）给出 `E[h_k(W,X)|X]=E[Y(t_k)|X]=μ_k`。✓

**(ii) q 正确（解 B2）**。展开
```
E[φ_k|X] = E[1{T=k}q_k Y|X] − E[1{T=k}q_k h_k|X] + E[h_k|X]
```
- 处理桥识别（PIPW）给出 `E[1{T=k}q_k(V,X)Y|X] = μ_k`。
- 对 `E[1{T=k}q_k h_k|X]`,先对 (W,X) 取条件:`h_k(W,X)` 可提出,`E[1{T=k}q_k(V,X)|W,X]=1`（由 B2）,
  故 `E[1{T=k}q_k h_k|X]=E[h_k(W,X)|X]`,与最后一项相消。
- 剩 `μ_k`。✓ ∎

于是 `τ_k(x)=E[φ_k−φ_0|X=x]`——**把因果目标转成一个可回归的伪结果条件期望**。这是 Stage-2 的依据。

### 1.3 定理 2（率双重稳健）:偏差 = 两个桥误差之积

现实里两个桥都是**估计的** `ĥ_k, q̂_k`。记 `δ_h=ĥ_k−h_k`,`δ_q=q̂_k−q_k`。伪结果之差:
```
φ_k(ĥ,q̂) − φ_k(h,q)
  = 1{T=k}[ (q+δ_q)(Y−h−δ_h) − q(Y−h) ] + δ_h
  = 1{T=k}[ −q·δ_h + δ_q(Y−h) − δ_q·δ_h ] + δ_h
```
逐项取 `E[·|X]`:
- **h 的一阶项**:`E[1{T=k}q·δ_h|X] = E[δ_h·E(1{T=k}q|W,X)|X] = E[δ_h|X]`（B2）,与 `+E[δ_h|X]` 抵消 → **0**。
- **q 的一阶项**:`E[1{T=k}δ_q(Y−h)|X] = E[δ_q·P(T=k|V,X)·E(Y−h|V,X,T=k)|X] = 0`（B1）→ **0**。
- **剩余唯一项**:`−E[1{T=k}·δ_q·δ_h|X]`。由 Cauchy–Schwarz:
  ```
  | 条件偏差 |  ≤  E[1{T=k}|δ_q||δ_h|]  ≤  ‖δ_q‖_{L2} · ‖δ_h‖_{L2}
  ```

**结论**:φ_k 的条件偏差被**两个桥 L2 误差之积**界住。两个一阶项消失 = **Neyman 正交性**——单个桥的
估计误差不进入一阶。这带来三个后果,贯穿全实现:
1. **单桥腐蚀不产生偏差**（E2 的 DR 对照:h-only 或 q-only 腐蚀,偏差仍随 n 消失）;
2. **双桥腐蚀的偏差 ~ n^{−(a_h+a_q)}**（E2 的乘积率）;
3. **Stage-2/推断中桥误差是二阶的** → ATE-CI 与 CATE-CI 的覆盖率随 n 趋于名义（E1、E10）。

---
<a name="part-2"></a>
## Part 2 Stage-1:kernel-moment 桥

### 2.1 从条件矩到无条件 MMR（为什么不需要对抗内循环）

(B1) 是条件矩 `E[R|Z]=0`,其中 `R=1{T=k}(Y−h_k)`,`Z=(V,X)`。它等价于:对某个足够丰富的函数类 F,
```
E[ R·f(Z) ] = 0  ,  ∀ f ∈ F
```
取 F = RKHS H 的单位球,**最大矩限制（MMR）**目标为
```
J(h) = sup_{‖f‖_H ≤ 1} ( E[R·f(Z)] )²
```
由核均值嵌入,这个 sup 有**闭式**:
```
J(h) = ‖ E[ R·k(Z,·) ] ‖²_H  =  E_{(i,j) 独立} [ R_i R_j k(Z_i,Z_j) ]
```
即"两份独立拷贝的 R·R·核"的期望。**因此无需对抗训练一个判别器 f**——sup 已被核内积消掉
（Dikkala et al. 2020;Kompa 2022 NMMR）。最小化 `J(h)` 把条件矩推向 0。q 的 (B2) 完全同理。

### 2.2 U 统计量:无偏经验估计

`J` 的**无偏**经验估计是排除对角（i=j）的 U 统计量:
```
Ĵ = 1/(B(B−1)) · Σ_{i≠j} R_i R_j k(z_i,z_j)
```
排除 i=j 是关键:对角项 `R_i² k(z_i,z_i)` 会引入 `E[R²k(Z,Z)]` 这一有偏成分;去掉后 Ĵ 恰是
`E_{i≠j}[R_i R_j k]` 的无偏估计。

**核心恒等式（代码向量化的依据）**:
```
Σ_{i≠j} R_i R_j K_ij = Rᵀ K R − Σ_i R_i² K_ii = Rᵀ (K − diag(K)) R
```
对应代码 `opm/bridges/kernel_moment.py`:
```python
def gaussian_kernel(z, bw):
    d2 = torch.cdist(z, z) ** 2
    return torch.exp(-d2 / (2.0 * bw * bw))          # K_ij = exp(-‖z_i-z_j‖²/(2 bw²))

def ustat_quadratic(vals, Kmat):
    B = vals.shape[0]
    Kz = Kmat - torch.diag(torch.diag(Kmat))          # 对角置零 = K − diag(K)
    quad = vals @ (Kz @ vals)                         # = Σ_{i≠j} R_i R_j K_ij
    return quad / (B * (B - 1))                       # 无偏归一
```
单测 1（`tests/test_kernel_ustat.py`）用朴素双循环验证这个向量化与定义在 1e-6 内一致。

### 2.3 两个桥的损失、参数化与网络

对臂 k:
- `R_i = 1{T_i=k}(Y_i − h_k(W_i,X_i))`,核工具 `z_i = std(V_i,X_i)` → `L_h = ustat(R, K_z)`。
- `S_i = 1{T_i=k}·q_k(V_i,X_i) − 1`,核工具 `u_i = std(W_i,X_i)` → `L_q = ustat(S, K_u)`。

注意**工具与网络输入不同**:h 的**网络输入**是 (W,X),但**核工具**是 (V,X)（因为 B1 条件在 V,X 上）;
q 反之。代码里 `_feat_h=(W,X)`、`_inst_h=(V,X)`;`_feat_q=(V,X)`、`_inst_q=(W,X)`。

**q 的参数化**保证 `q≥1` 且有界:`q_k = clip(1 + softplus(g_k), max=50)`。

**网络**（`opm/bridges/networks.py`）:一个共享主干跨臂,臂用嵌入区分（稀有臂借力参数共享）:
```python
class TrunkNet(nn.Module):
    def __init__(self, feat_dim, K, arm_emb=8, hidden=128, depth=3):
        self.arm = nn.Embedding(K, arm_emb)
        dims = [feat_dim + arm_emb] + [hidden]*depth        # 3×128
        ... ReLU MLP ...
        self.head = nn.Linear(hidden, 1)
    def forward(self, feat, arm):
        z = torch.cat([feat, self.arm(arm)], dim=-1)        # concat(输入, arm_embedding)
        return self.head(self.trunk(z)).squeeze(-1)         # 标量
```
每个样本对每个臂 k 调用一次 `net(feat, arm=k)` 得到 `h_k` / `g_k`。

### 2.4 单步训练:整批核矩阵 + 逐臂累加

训练循环（摘自 `KernelBridges.fit`）:
```python
Kh = gaussian_kernel(zh[bt], self.bw_h)      # B×B 核矩阵（h 的工具 z=(V,X)）
Kq = gaussian_kernel(zq[bt], self.bw_q)      # B×B（q 的工具 u=(W,X)）
loss = 0
for k in range(self.K):                       # 逐臂累加（K 个臂共享主干）
    arm  = torch.full((B,), k)
    mask = (Tb == k).float()                  # 1{T=k}
    hk   = self.net_h(fh[bt], arm)
    R    = mask * (Yb - hk)                    # R_i = 1{T=k}(Y-h_k)
    loss += ustat_quadratic(R, Kh)            # L_h^k
    gk   = self.net_q(fq[bt], arm)
    qk   = torch.clamp(1.0 + F.softplus(gk), max=cfg.q_max)
    S    = mask * qk - 1.0                     # S_i = 1{T=k}q_k − 1
    loss += ustat_quadratic(S, Kq)            # L_q^k
opt.zero_grad(); loss.backward(); opt.step()
ema_h.update(self.net_h); ema_q.update(self.net_q)
```
要点:
- **总损失 = Σ_k (L_h^k + L_q^k)**。h 网与 q 网**参数不相交**,所以合并成一个 backward 与分别训练**等价**——
  这不是"跨阶段联合目标",也无损失平衡（符合不可协商原则 #1）。
- **批 ≥ 512**,保证 U 统计量的成对项充足;整批 B×B 核矩阵,对角在 `ustat_quadratic` 内置零。
- **AdamW**（lr 1e-3, wd 1e-4）。

### 2.5 中位数启发式带宽

带宽一次性在训练折上取**标准化工具的中位成对距离**（`opm/utils.py::median_bandwidth`,大 n 时抽样成对以
控成本）。理由:把高斯核置于"相似度既非全 1、也非全 0"的信息量最大的尺度上——MMD/核方法的标准选择。
代码:`self.bw_h = median_bandwidth(inst_h, seed=...)`。

### 2.6 EMA:稳定噪声化的 U 统计量梯度

U 统计量梯度方差大,故对权重取 **EMA(0.99)**,并用 EMA 影子权重做所有下游预测:
```python
def update(self, model):
    for s, p in zip(self.shadow.parameters(), model.parameters()):
        s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)   # θ_ema ← 0.99 θ_ema + 0.01 θ
```

### 2.7 早停诊断:随机傅里叶特征（RFF）残差

条件矩 `E[R|Z]=0 ⟺ E[R g(Z)]=0 ∀g`。用 100 个 RFF `g_j(z)=cos(ω_j·z+b_j)`（ω_j~N(0,1/bw²)）近似"丰富
函数类",在**验证折**上算残差分数:
```
score = mean_k  max_j | E_n[ R_i^k · g_j(z_i) ] | / sd(R^k)      （h,q 各一半,取平均）
```
分数越低 = 矩越接近满足。以此早停（patience 30），保留分数最低的 EMA 权重。代码 `_val_residual`:
```python
gz_h = torch.cos(zh[vidx] @ omega_h.T + b_h)      # (nv, 100)
...
m_h = (R[:,None]*gz_h).mean(0).abs().max() / sdR   # max_j |En[R g_j]| / sd(R)
```
这就是 spec §6.1 的"桥残差诊断",也在 E8 里被验证为 PEHE 的强预测量（bridge_resid Spearman +0.6）。

### 2.8 预测 API

`predict_h(W,X)` / `predict_q(V,X)` 各返回 `(n, K)`——对**所有臂**给出桥值（注意 h_k 对所有样本有定义,
不只 T=k 的样本;这是 (STAR) 需要的）。q 输出再次 `clip(1+softplus, max=50)`。

---
<a name="part-3"></a>
## Part 3 Stage-2:交叉拟合 · 伪结果 · CATE 头 · ATE 置信区间

### 3.1 交叉拟合(leakage-proof)

`opm/estimator/crossfit.py` 把 [0,n) 洗牌后均分 L=5 折;对每折,训练集=其余折,预测集=该折,保证
`train ∩ test = ∅`。`assignments` 记录每个样本被哪一折预测,供单测 5 断言"无泄漏"。为什么必须交叉拟合:
让**桥的估计误差与被预测样本独立**,从而 Theorem 2 的二阶偏差论证成立、伪结果是真正 OOF 的。

### 3.2 (STAR) 伪结果装配

`opm/estimator/pseudo_outcome.py`:
```python
def compute_phi(h_pred, q_pred, ds):                 # h_pred,q_pred: (n,K)
    onehot = 0; onehot[arange(n), ds.T] = 1.0         # 1{T_i=k}
    residual = onehot * q_pred * (Y - h_pred)         # 1{T=k} q_k (Y - h_k)
    return residual + h_pred                          # + h_k  →  φ_k
```
逐字对应 (STAR) `φ_k = 1{T=k}q_k(Y−h_k)+h_k`。

### 3.3 顶层编排 `OPM.fit`

`opm/estimator/learner.py`:交叉拟合 → 拼 OOF φ → Stage-2。关键片段:
```python
if c.n_folds <= 1:                                    # E6(b) 消融:关闭交叉拟合(样本内,有泄漏)
    br = self._make_bridge(seed=...).fit(ds)
    h_oof = br.predict_h(ds.W, ds.X); q_oof = br.predict_q(ds.V, ds.X)
    phi   = compute_phi(h_oof, q_oof, ds)
else:
    self.cf = CrossFitter(n, c.n_folds, c.seed)
    for fold, (tr, te) in enumerate(self.cf):
        br  = self._make_bridge(seed=c.seed+100*fold+1).fit(ds.subset(tr))   # 在其余折训桥
        sub = ds.subset(te)
        h = br.predict_h(sub.W, sub.X); q = br.predict_q(sub.V, sub.X)       # 预测该折
        h_oof[te], q_oof[te] = h, q
        phi[te] = compute_phi(h, q, sub)                                     # OOF φ
# Stage-2:
D = phi[:, 1:] - phi[:, [0]]                          # D_i = (φ_1−φ_0,…) ∈ R^{K-1}
self.tau_head = MLPRegressor(HeadConfig(...)).fit(ds.X, D)     # τ̂: MLP 3×256
self.g0_head  = MLPRegressor(...).fit(ds.X, phi[:, 0])         # g0̂ = 回归 φ_0 于 X
self.ate = {k: ate_with_ci(phi[:,k]-phi[:,0]) for k in 1..K-1}
```
`mode` 决定桥工厂:`proximal`→KernelBridges,`dr_fallback`→PlugInBridges,`sieve`→SieveBridges——**同一套
交叉拟合/伪结果/Stage-2 复用**,只换桥。`predict_cate(X)=τ̂(X)`;`predict_mu(X)=g0̂(X)+τ̂(X)`。

### 3.4 CATE 头（`tau_head.py`）

`MLPRegressor`:标准化输入 → MLP(3×256,ReLU) → 对验证 MSE 早停（10% split,patience 20）,保留最优权重。
`fit_cate_and_ate` 把 `D=φ_{1:}−φ_0` 回归到 X 得 τ̂,并对每个臂算 ATE-CI。

### 3.5 ATE 影响函数置信区间

`ATE_k = E[φ_k−φ_0]`。在该识别下,**efficient influence function** 就是中心化的伪结果差
`ψ_i = φ_k(O_i)−φ_0(O_i) − ATE_k`。故
```
√n ( ATÊ_k − ATE_k )  →  N( 0, Var(ψ) )
```
用样本方差估 `Var(ψ)`,得 `SE = sd(ψ)/√n`,`CI95 = ATÊ ± 1.96·SE`。交叉拟合 + Neyman 正交保证桥误差不进入
一阶,CI 渐近有效（E1 覆盖率 0.875;E10 覆盖率随 n→0.97）。代码 `ate_with_ci`:
```python
ate = mean(psi); se = std(psi, ddof=1)/sqrt(n)
return {ate, se, ci_low=ate-1.96*se, ci_high=ate+1.96*se}
```

### 3.6 q 诊断:配平、最大权重、ESS

`OPM._diag_q` 对每个臂算:
- `En[1{T=k}q_k]`（目标 1;proximal 由 B2 天然≈1,E4/RHC 实测 0.998/0.972）;
- `max q`;
- **有效样本量** `ESS_k = (Σ w)² / Σ w²`（Kish ESS,w 为 treated 臂-k 单元的 q 权重）。
E4/RHC 实测 ESS=3443/1796。

---
<a name="part-4"></a>
## Part 4 dr_fallback:插件 AIPW + Hájek 稳定化

无代理时,桥退化为标准 AIPW 讨厌参数:`h_k→m̂_k(X)`（逐臂结果回归）,`q_k→1/π̂_k(X)`（softmax 倾向,
π 裁剪到 [1/50,1]）。代入 (STAR) 就是经典 AIPW:
```
φ_k = 1{T=k}/π̂_k(X)·(Y − m̂_k(X)) + m̂_k(X)
```
**Hájek 稳定化**:插件倾向不满足 B2,故 `En[1{T=k}/π̂_k]` 未必=1。在训练集上自归一化:
```
q̃_k(X) = (1/π̂_k(X)) / En_train[ 1{T=k}/π̂_k ]      →  En_train[1{T=k}q̃_k] = 1
```
这就是**自归一化/Hájek IPW**。代码 `opm/bridges/plugin.py`:
```python
q_raw = self._raw_q(X)                                # 1/clip(π, 1/50, 1)
self.norm_k = [ (onehot[:,k]*q_raw[:,k]).mean() for k in range(K) ]   # En[1{T=k}q_k]
...
def predict_q(self, V, X):
    return clip( self._raw_q(X) / self.norm_k[None,:], 0, q_max )     # 稳定化
```
`linear=True`（E4 线性门禁）用 Ridge + 多项 logistic;否则用 HistGradientBoosting。**注意**:E4/RHC 因有真实
代理走的是 proximal（B2 天然配平,严格权重带成立）;HAMD 无代理才走 dr_fallback,稀有臂 Fluoxetine(n=50) 用
了有记录的放宽带（见 `DECISIONS.md`）。

---
<a name="part-5"></a>
## Part 5 sieve 桥:两阶段最小二乘（2SLS）

sieve 用有限维基展开逼近桥,给出 POR/PIPW/PDR 基线 + E6(a) 消融 + E4 线性门禁。

### 5.1 结果桥 h 的 2SLS

设 `h_k(W,X)=Φ_h(W,X)ᵀβ_k`,工具基 `Ψ(V,X)`。(B1) 的样本版 `E_n[(Y−Φ_hβ)Ψ]=0`（在 T=k 子样本上）。
sieve 最小距离 / 2SLS 解:
```
β_k = ( Φ_hᵀ P_Ψ Φ_h )^{-1} Φ_hᵀ P_Ψ Y ,   P_Ψ = Ψ(ΨᵀΨ)^{-1}Ψᵀ  （投影到工具张成的空间）
```
代码 `_twosls`（加 ridge 稳定）:
```python
Pinv    = solve(Ψᵀ Ψ + ridge·I, Ψᵀ)          # (ΨᵀΨ)^{-1} Ψᵀ
Phi_hat = Ψ @ (Pinv @ Φ)                       # P_Ψ Φ  （投影后的回归子）
A = Phi_hatᵀ Φ + ridge·I;  b = Phi_hatᵀ Y
β = solve(A, b)                                # (Φᵀ P_Ψ Φ)^{-1} Φᵀ P_Ψ Y
```
（因 `Phi_hatᵀΦ = ΦᵀP_ΨᵀΦ = ΦᵀP_ΨΦ`,P 对称幂等,与 2SLS 公式一致。）

### 5.2 处理桥 q 的线性矩解

设 `q_k(V,X)=Φ_q(V,X)ᵀα_k`,工具 `ξ(W,X)`。(B2):`E[(1{T=k}Φ_qᵀα−1)ξ]=0` →
```
E[ 1{T=k} ξ Φ_qᵀ ] α_k = E[ ξ ]     即  A α = b
```
代码 `_q_solve` 用正规方程 `(AᵀA)α=Aᵀb` 求最小二乘解:
```python
A = (Xi * 1{T=k}).T @ PhiQ / n      # A = E_n[1{T=k} ξ Φ_q']
b = Xi.mean(0)                      # b = E_n[ξ]
α = solve(AᵀA + ridge·I, Aᵀ b)
```
predict_q 再 clip 到 [0, q_max]。

### 5.3 基 `_Basis`:先标准化→多项式→再标准化→裁剪→加偏置

关键工程细节:**多项式展开后必须再标准化**（交叉项 x_i·x_j 尺度差异极大,否则 2SLS 正规方程病态）,
并 `clip(|·|≤8σ)` 驯服 U² 类重尾高阶特征:
```python
P = scaler_out.transform( poly.transform( scaler_in.transform(A) ) )
P = np.clip(P, -8.0, 8.0)
return concat([ones, P])            # 加偏置列
```
degree=1（线性 sieve）稳定且常最优;degree=2/3 在强非线性下高方差/不稳(E9:PEHE 1.24、2.38)——这正是
"kernel 桥的价值是免选阶的鲁棒性"的实证来源。

---
<a name="part-6"></a>
## Part 6 CATE 逐点推断:局部线性 + 三明治方差（E10 新增）

### 6.1 D 是 τ 的无偏信号

由定理 1,OOF 伪结果差 `D_i = φ_1(O_i) − φ_0(O_i)` 满足 `E[D_i|X_i]=τ(X_i)`,即
`D_i = τ(X_i) + 噪声_i`（噪声条件均值 0）。于是**把 D 对一维效应修饰变量 m=x_j 做非参回归 = 估计 τ(m)**。

### 6.2 局部线性 WLS

在目标点 x0,以高斯核权重 `w_i=exp(−½((m_i−x0)/bw)²)` 做加权最小二乘,设计阵 `Z_i=[1, m_i−x0]`:
```
β̂ = argmin Σ_i w_i ( D_i − Z_iᵀβ )²  = (ZᵀWZ)^{-1} ZᵀW D
τ̂(x0) = β̂_0   （截距 = x0 处的估计;斜率 β̂_1 是局部导数,自动修掉一阶偏差）
```
**局部线性对局部线性的 τ 无偏、无边界偏差**——这是选它（而非局部常数/NW）的原因:τ(x)=2+x_1 是线性的,
故任意带宽都近似无偏,带宽只影响方差,可放大以收紧 CI（`silverman_bw` 乘 1.5）。

### 6.3 异方差稳健三明治方差（完整推导）

把核权重视为固定,残差 `r_i=D_i−Z_iᵀβ̂`。WLS 的 Huber–White 稳健方差:
```
Var(β̂) = A^{-1} · ( Σ_i w_i² r_i² Z_i Z_iᵀ ) · A^{-1},   A = ZᵀWZ = Σ_i w_i Z_i Z_iᵀ
```
中间"肉" `Σ_i w_i² r_i² Z_iZ_iᵀ` 允许每点方差不同（异方差,proximal 伪结果确实异方差）。
`τ̂` 的方差 = `Var(β̂)[0,0]`,`SE=√Var(β̂)[0,0]`,`CI95 = τ̂ ± 1.96·SE`。

代码 `opm/estimator/cate_inference.py`（逐行对应上式）:
```python
w   = np.exp(-0.5*((m-x0)/bw)**2)              # 核权重
Z   = column_stack([ones, m-x0])              # [1, m-x0]
A   = Z.T @ (Z*w[:,None])                     # ZᵀWZ  (2×2)
b   = (Z*w[:,None]).T @ D                     # ZᵀW D
beta= Ainv @ b ; tau_hat = beta[0]            # β̂,τ̂=截距
r   = D - Z @ beta                            # 残差
meat= (Z*(w*r)[:,None]).T @ (Z*(w*r)[:,None]) # Σ_i (w_i r_i)² Z_iZ_iᵀ = Σ w²r² ZZᵀ
Vbeta = Ainv @ meat @ Ainv                    # 三明治
se  = sqrt(Vbeta[0,0])
```
（`(Z*(w r)[:,None])` 的第 i 行是 `w_i r_i Z_i`,其 `MᵀM = Σ_i w_i²r_i² Z_iZ_iᵀ`,正是"肉"。）
这是 Kennedy(2020) DR-learner 的逐点推断,施于 proximal 伪结果。

### 6.4 子组 ATE:精确 CLT 均值区间（E10 主判据）

更稳的一档:按 x_j 分位分箱,箱内 `est=mean(D)`,`CI=est±1.96·sd/√n_bin`,覆盖箱内 τ 的均值。这是**普通
样本均值 CI**——无可争议正确,且 SE 略偏保守（含箱内 τ 变差）,故覆盖 ≥ 名义。E10 主判据用它。

### 6.5 为什么覆盖率随 n 趋于名义

CI 只捕捉伪结果的采样方差,不含桥估计的额外方差;但由 Neyman 正交（定理 2）,桥误差是**二阶**的,随 n 以
更快的率消失。故小 n 时有轻微、可缩小的欠覆盖,`n→∞` 趋于名义 95%。E10 实测:子组 0.70→0.90→0.967,
逐点 0.60→0.90→0.967——正是这个渐近有效性。

---
<a name="part-7"></a>
## Part 7 Stage-3:条件 DDPM + 均值蒸馏（可选,E7）

目标:在 Stage-1/2 **冻结**下,训一个条件生成器,其**样本均值被 proximal 的 μ̂_k 去偏**（生成式基线在混杂下
均值有偏）。

### 7.1 前向扩散与 eps-预测损失

余弦调度 `ᾱ_t`（`cosine_alpha_bar`）。前向 `q(y_t|y_0)=N(√ᾱ_t·y_0, (1−ᾱ_t))`,即
`y_t=√ᾱ_t·y_0+√(1−ᾱ_t)·ε`。训练**事实对** (X,T,Y) 的去噪损失（ε-预测）:
```python
def _denoise_loss(self, y0, x, arm, rng):
    t   = randint(0, T);  eps = randn_like(y0)
    y_t = sqrt_ab[t]*y0 + sqrt_1mab[t]*eps
    eps_hat = self.model(y_t, t, x, arm)          # EpsMLP(4×256), 条件 c=(embed X, arm, t-embed)
    return mse(eps_hat, eps)                       # ‖ε − ε_θ(y_t,t,c)‖²
```
条件 `c = concat(MLP_embed(X), arm_embedding, sin_embed(t), y_t)`。

### 7.2 反向采样（两种）

**祖先采样（ancestral,E7 评估用,更准）**:DDPM 后验 `q(y_{t-1}|y_t,ŷ_0)` 的均值 + 方差 β_t:
```
ŷ_0 = ( y_t − √(1−ᾱ_t)·ε_θ ) / √ᾱ_t
μ   = √ᾱ_{t-1}·β_t/(1−ᾱ_t)·ŷ_0 + √α_t·(1−ᾱ_{t-1})/(1−ᾱ_t)·y_t ,  α_t=ᾱ_t/ᾱ_{t-1}, β_t=1−α_t
y_{t-1} = μ + √β_t·z         （t>0;t=0 取 ŷ_0）
```
代码 `_ancestral_sample` 逐字实现该式。
**DDIM 确定性采样（蒸馏梯度用,少步、可微）**:`y_{t-1}=√ᾱ_{t-1}·ŷ_0 + √(1−ᾱ_{t-1})·ε_θ`（无随机项）。

### 7.3 均值蒸馏损失

每 `distill_every` 步,对小批子集抽 M 样本/(X_i,臂 k),用**可微 DDIM** 采样,把样本均值拉向 μ̂_k:
```python
def _distill_loss(self, Xt, mu_t, subset, M, n_steps, rng):
    for k in range(K):
        samp   = self._ddim_sample(x_subset, arm=k, M, n_steps)   # 反传经过 DDIM
        mean_M = samp.mean(dim=1)
        total += mse(mean_M, mu_t[idx, k])                        # (mean_M G − μ̂_k)²
    return total / K
# 总损失:  L = L_ddpm + λ_distill · L_distill      （λ_distill 是核心方法唯一可调超参）
```

### 7.4 已知局限（E7 未达标的诚实解释）

蒸馏**只去偏均值**,高阶矩继承事实条件形状（spec 明示,E7 的 W1 反映方差失配）。且在 CPU 可行的 DDPM 质量下,
factual-DDPM 的均值偏差本就小（0.5–0.86,由生成器自身条件均值误差主导,而非可去除的混杂）,故"蒸馏减 50%"
不可达,through-sampling 梯度还噪声化。E6(d) 显示 λ=10 能把均值偏差降到 0.71（机制方向性为真）,但数值判据
在此规模不过。**Stage 3 是 spec 明示的可选模块;核心估计器 (Stage 1-2) 已由 E1-E6、E8、E9、E10 完整验证。**

---

## 附:数学 ↔ 代码 ↔ 实验 对照速查

| 数学对象 | 代码位置 | 相关实验/验证 |
|---|---|---|
| U 统计量 MMR 损失 | `bridges/kernel_moment.py::ustat_quadratic` | 单测 1;E1 桥收敛 |
| 闭式结果桥 `h_t=2t+1.25W+β'X` | `dgp/linear_gaussian.py::h_true` | 单测 2;E1 探针 R²=0.974 |
| 定理 1(识别,STAR) | `estimator/pseudo_outcome.py::compute_phi` | 单测 2/3;E1 |
| 定理 2(乘积偏差,Neyman 正交) | 由交叉拟合 + STAR 保证 | 单测 4;**E2** 乘积率 |
| ATE 影响函数 CI | `estimator/tau_head.py::ate_with_ci` | E1 覆盖 0.875;E4 CI |
| Hájek 稳定化权重 | `bridges/plugin.py::norm_k` | E4 权重带 |
| sieve 2SLS | `bridges/sieve.py::_twosls/_q_solve` | E3/E6a;E4 线性门禁 |
| 局部线性三明治 CI | `estimator/cate_inference.py::local_linear_ci` | **E10** 覆盖 0.70→0.97 |
| DDPM 祖先/DDIM 采样 + 蒸馏 | `generator/ddpm.py` | E6(d/e);E7 |
| c_U=0 时 fallback≡proximal | `estimator/learner.py`(两模式) | 单测 7;**E6(f)** |

> 与本文档配套:`FULL_REPORT.md`(实验总览)、`POSITIONING.md`(定位)、`REVIEW.md`(交叉评审)、
> `DECISIONS.md`(所有决策/偏离)、`results/SUMMARY.md`(PASS/FAIL 总板)。
