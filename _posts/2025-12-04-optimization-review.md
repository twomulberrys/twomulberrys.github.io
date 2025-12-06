---
title: 第一章：凸集与凸函数
date: 2025-12-04 17:00:00 +0800
categories: [学习笔记, 最优化]
tags: [数学, 凸优化, 期末复习]
math: true
---

本章主要梳理关于凸集与凸函数的核心定义、重要定理（如分离定理、Farkas 引理）以及典型习题的解题思路。

## 1. 凸集 (Convex Sets)

### 1.1 核心定义
**凸集**：集合 $S \subseteq \mathbb{R}^n$ 是凸集，若对任意 $x^{(1)}, x^{(2)} \in S$ 及 $\lambda \in [0, 1]$，都有：
$$
\lambda x^{(1)} + (1-\lambda)x^{(2)} \in S
$$
即集合中任意两点的连线仍在集合内。

> **注**：线性规划的可行域（多面集）是典型的凸集。

### 1.2 运算性质
设 $S_1$ 和 $S_2$ 是两个凸集，$\beta$ 为实数，则以下集合均为凸集：
1. **数乘**：$\beta S_1 = \{ \beta x \mid x \in S_1 \}$
2. **交集**：$S_1 \cap S_2$
3. **和集**：$S_1 + S_2 = \{ x^{(1)} + x^{(2)} \mid x^{(1)} \in S_1, x^{(2)} \in S_2 \}$
4. **差集**：$S_1 - S_2 = \{ x^{(1)} - x^{(2)} \mid x^{(1)} \in S_1, x^{(2)} \in S_2 \}$

### 1.3 极点与方向
* **极点 (Extreme Point)**：设 $S$ 是非空凸集，$x \in S$。若 $x$ 不能表示成 $S$ 中两个不同点的凸组合（即若 $x = \lambda x^{(1)} + (1-\lambda)x^{(2)}$ 且 $\lambda \in (0,1)$，必推出 $x = x^{(1)} = x^{(2)}$），则称 $x$ 是极点。

> **结论：多面集的方向**
>
> 设 $S = \{x \mid Ax = b, x \ge 0\}$ 为非空集合，$d$ 是非零向量。则 $d$ 是 $S$ 的方向的充要条件是：
>
> $$d \ge 0 \quad \text{且} \quad Ad = 0$$
>
> *(注：此处指一般方向，而非极方向。极方向是指不能被其他两个方向正线性组合的方向)。*

---

## 2. 凸集分离定理及其应用

这一部分是理论证明的核心，特别是 Farkas 定理，是后续对偶理论的基础。

### 2.1 投影定理
**投影定理**：设 $S$ 是 $\mathbb{R}^n$ 中的闭凸集，$y \notin S$。则存在**唯一**的点 $\bar{x} \in S$，使得到 $y$ 的距离最小，即：
$$
\| y - \bar{x} \| = \inf_{x \in S} \| y - x \|
$$
$\bar{x}$ 称为 $y$ 在 $S$ 上的投影。

### 2.2 分离定理
**凸集分离定理**：设 $S_1, S_2$ 是非空凸集且 $S_1 \cap S_2 = \emptyset$，则存在非零向量 $p$，使得：
$$
\inf \{ p^T x \mid x \in S_1 \} \ge \sup \{ p^T x \mid x \in S_2 \}
$$
即存在超平面分离这两个集合。

**点与闭凸集分离定理**：设 $S$ 是闭凸集，$y \notin S$。则存在非零向量 $p$ 及 $\epsilon > 0$，使得对任意 $x \in S$：
$$
p^T y \ge \epsilon + p^T x
$$
这意味着存在一个超平面**严格分离**点 $y$ 和集合 $S$。



> **证明思路**
>
> 1. **寻找最近点**：由于 $S$ 是闭凸集且 $y \notin S$，由投影定理可知，存在 $S$ 中唯一的点 $\bar{x}$ 使得 $\|y - \bar{x}\| = \inf_{x\in S} \|y - x\| > 0$。
>
> 2. **利用凸性证明变分不等式**：由于 $\bar{x}$ 是 $y$ 在 $S$ 上的投影。对于任意 $x \in S$，构造线段上的点 $z(\lambda) = (1 - \lambda)\bar{x} + \lambda x$ ($\lambda \in [0, 1]$)。距离平方函数 $\phi(\lambda) = \|y - z(\lambda)\|^2$ 在 $\lambda = 0$ 处取最小值，故导数非负：
>    $$
>    \frac{d}{d\lambda}\|y - (\bar{x} + \lambda(x - \bar{x}))\|^2 \Big|_{\lambda=0} \ge 0
>    $$
>    展开求导可得：
>    $$
>    -2(y - \bar{x})^T(x - \bar{x}) \ge 0 \implies (y - \bar{x})^T(x - \bar{x}) \le 0
>    $$
>    整理符号即得：$(y - \bar{x})^T(\bar{x} - x) \ge 0$。
>
> 3. **得出分离结论**：令 $p = y - \bar{x}$。由上一步可知 $p^T \bar{x} \ge p^T x$。因为 $y = \bar{x} + p$，所以：
>    $$
>    p^T y = p^T(\bar{x} + p) = p^T \bar{x} + \|p\|^2
>    $$
>    代入不等式中：
>    $$
>    p^T y - \|p\|^2 \ge p^T x \implies p^T y \ge \|p\|^2 + p^T x
>    $$
>    令 $\epsilon = \|p\|^2 = \|y - \bar{x}\|^2 > 0$，即证 $p^T y \ge \epsilon + p^T x$。

### 2.3 Farkas 定理 (择一性定理)
**Farkas 定理**：设 $A$ 为 $m \times n$ 矩阵，$c$ 为 $n$ 维向量。则下列两个系统**有且仅有一个**有解：

* **系统 1**：$Ax \le 0, \quad c^T x > 0$ 有解。
* **系统 2**：$A^T y = c, \quad y \ge 0$ 有解。

> **证明思路**
>
> 令 $S = \{z \mid z = A^T y, y \ge 0\}$，这是一个闭凸锥。
>
> * 若 $c \in S$，则系统 2 有解，系统 1 无解（易证）。
> * 若 $c \notin S$，由点与凸集分离定理，存在非零向量 $x$ 及 $\epsilon > 0$，使得对任意 $z \in S$，有 $x^T c \ge \epsilon + x^T z$。
>     * 取 $z=0$ 得 $x^T c > 0$。
>     * 由于 $z$ 可以是 $A^T y$ ($y \ge 0$)，推导出 $x^T A^T y$ 有上界，进而推出 $Ax \le 0$。
>     * 从而构造出系统 1 的解。

> **进阶：为什么满足 Slater 条件时 $w_0 \neq 0$？**
>
> 这是一个基于反证法的经典证明，连接了分离定理与 KKT 条件。
>
> **证明**：假设 $w_0 = 0$。
>
> 1. **分离不等式退化**：
>    如果 $w_0 = 0$，Fritz John 条件中的分离超平面变得与目标函数 $f(x)$ 无关。对于数值集合 $\mathcal{A}$ 中的所有点，分离不等式退化为：
>    $$
>    \sum_{i=1}^m w_i g_i(x) + \sum_{j=1}^l v_j h_j(x) \ge 0
>    $$
>
> 2. **引入 Slater 点**：
>    根据 Slater 条件，存在一个严格内点 $\hat{x}$ 使得 $g_i(\hat{x}) < 0$ 且 $h_j(\hat{x}) = 0$。
>
> 3. **导出矛盾**：
>    将 $\hat{x}$ 代入退化后的不等式中：
>    $$
>    \sum_{i=1}^m w_i \underbrace{g_i(\hat{x})}_{<0} + \sum_{j=1}^l v_j \underbrace{h_j(\hat{x})}_{0} \ge 0
>    $$
>    由于 $w_i \ge 0$ 且不全为零，上述求和项必然 **严格小于 0**，这与 $\ge 0$ 矛盾！
>
> **结论**：假设不成立，故必然有 $w_0 > 0$。

### 2.4 Gordan 定理
**Gordan 定理**：设 $A$ 为 $m \times n$ 矩阵。那么，$Ax < 0$ 有解的充要条件是：不存在非零向量 $y \ge 0$，使得 $A^T y = 0$。

这也可以表述为下列两个系统**有且仅有一个**有解：
1. $Ax < 0$
2. $A^T y = 0, \quad y \ge 0, \quad y \neq 0$

> **证明**
>
> **1. 先证必要性 ($\implies$)**
>
> 假设 $Ax < 0$ 有解，即存在 $\bar{x}$，使 $A\bar{x} < 0$。
> 假设反面成立，即存在非零向量 $y \ge 0$ 使得 $A^T y = 0$，即 $y^T A = 0$。
>
> 在上式两端右乘 $\bar{x}$，得到：
> $$y^T A \bar{x} = 0$$
>
> 然而，由于 $A\bar{x} < 0$ 且 $y \ge 0$（且 $y$ 非零），两个向量的点积 $y^T (A\bar{x})$ 必然小于 0（或者说，为了使点积为 0，在 $A\bar{x}$ 全为负数的情况下，$y$ 必须包含负分量）。这与 $y \ge 0$ 矛盾。
>
> **2. 再证充分性 ($\impliedby$)**
>
> 我们证明其等价命题（逆否命题）：**若 $Ax < 0$ 无解，则存在非零向量 $y \ge 0$ 使 $A^T y = 0$。**
>
> 设 $Ax < 0$ 无解。构造以下两个集合：
> $$S_1 = \{ z \mid z = Ax, x \in \mathbb{R}^n \} \quad (\text{线性子空间})$$
> $$S_2 = \{ z \mid z < 0 \} \quad (\text{负开象限，凸集})$$
>
> 由于 $Ax < 0$ 无解，因此 $S_1 \cap S_2 = \emptyset$。
> 根据凸集分离定理，存在非零向量 $y$，使得对所有 $x \in \mathbb{R}^n$ 及 $z \in S_2$，有：
> $$y^T Ax \ge y^T z$$
>
> **(1) 确定 $y$ 的符号**：
> 特别地，取 $x = 0$（此时 $Ax=0$），不等式变为：
> $$0 \ge y^T z \implies y^T z \le 0$$
> 由于 $z$ 是 $S_2$ 中的任意点（即 $z$ 的每个分量都可以是任意负数），要保证 $y^T z \le 0$ 恒成立，**$y$ 的各分量必须非负**，即：
> $$y \ge 0$$
>
> **(2) 确定 $A^T y$ 的值**：
> 在分离不等式 $y^T Ax \ge y^T z$ 中，令 $z \to 0$（因为 0 是 $S_2$ 的边界点），得到对任意 $x \in \mathbb{R}^n$ 均有：
> $$y^T Ax \ge 0$$
>
> 令 $x = -A^T y$，代入上式：
> $$y^T A (-A^T y) = - (y^T A) (A^T y) = - \| A^T y \|^2 \ge 0$$
>
> 因为范数平方非负，所以必须有 $\| A^T y \| = 0$，即：
> $$A^T y = 0$$
>
> 综上，我们找到了非零向量 $y \ge 0$ 满足 $A^T y = 0$。证毕。

---

## 3. 凸函数 (Convex Functions)

### 3.1 定义与判定
**定义**：定义在凸集 $S$ 上的函数 $f(x)$ 是凸函数，若对任意 $x^{(1)}, x^{(2)} \in S$ 及 $\lambda \in [0, 1]$：
$$
f(\lambda x^{(1)} + (1-\lambda)x^{(2)}) \le \lambda f(x^{(1)}) + (1-\lambda)f(x^{(2)})
$$

> **凸函数的判定定理**
>
> 设 $S$ 是 $\mathbb{R}^n$ 中的非空开凸集。
>
> 1. **一阶条件**：设 $f \in C^1(S)$，则 $f$ 是凸函数的充要条件是：
>    $$f(x) \ge f(\bar{x}) + \nabla f(\bar{x})^T(x - \bar{x}), \quad \forall x, \bar{x} \in S$$
>
> 2. **二阶条件**：设 $f \in C^2(S)$，则 $f$ 是凸函数的充要条件是 Hesse 矩阵 $\nabla^2 f(x)$ 在 $S$ 上半正定。
>
> 3. **严格凸判别**：若 $\nabla^2 f(x)$ 在 $S$ 上正定，则 $f$ 是严格凸函数。

### 3.2 凸规划
**模型**：求凸函数在凸集上的极小点。

$$
\begin{aligned}
\min \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \ge 0, \quad i=1,\dots,m \\
& h_j(x) = 0, \quad j=1,\dots,l
\end{aligned}
$$

其中 $f(x)$ 为凸函数，$g_i(x)$ 为凹函数（保证可行域是凸集），$h_j(x)$ 为线性函数。

> **凸规划基本定理**
>
> 设 $f(x)$ 是凸函数，$S$ 是凸集。在凸规划问题 $\min\{f(x) \mid x \in S\}$ 中：
> 1. 局部极小点就是全局极小点。
> 2. 极小点的集合是凸集。

---

## 4. 重点习题详解

### 习题 1：凸集的验证

**题目**：验证下列各集合是凸集：
1. $S = \{(x_1, x_2) \mid x_1 + 2x_2 \ge 1, x_1 - x_2 \ge 1\}$
2. $S = \{(x_1, x_2) \mid x_2 \ge \lvert x_1 \rvert \}$
3. $S = \{(x_1, x_2) \mid x_1^2 + x_2^2 \le 10\}$

**解**：

**(1)** 该集合是两个半平面的交集。设 $S_1 = \{x \mid a_1^T x \ge b_1\}, S_2 = \{x \mid a_2^T x \ge b_2\}$。半平面（半空间）是凸集，凸集的交集仍为凸集，故 $S$ 是凸集。

**(2)** 利用定义证明。设 $y, z \in S$，即 $y_2 \ge \lvert y_1 \rvert, z_2 \ge \lvert z_1 \rvert$。对于 $\lambda \in [0, 1]$，令 $w = \lambda y + (1 - \lambda)z$。
$$
\begin{aligned}
w_2 &= \lambda y_2 + (1 - \lambda)z_2 \\
&\ge \lambda \lvert y_1 \rvert + (1 - \lambda)\lvert z_1 \rvert \\
&\ge \lvert \lambda y_1 + (1 - \lambda)z_1 \rvert \quad (\text{三角不等式}) \\
&= \lvert w_1 \rvert
\end{aligned}
$$
故 $w \in S$，集合为凸集。

**(3)** 该集合是圆盘。定义 $g(x) = x_1^2 + x_2^2 - 10$。$\nabla^2 g(x) = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$ 正定，故 $g(x)$ 是凸函数。凸函数的水平集 $S = \{x \mid g(x) \le 0\}$ 是凸集。

### 习题 2 & 3：线性变换下的凸性

**题目 (习题 2)**：设 $C \subset \mathbb{R}^n$ 是凸集，证明 $S = \{x \in \mathbb{R}^n \mid x = Ay, y \in C\}$ 是凸集。

**解**：
设 $x^{(1)}, x^{(2)} \in S$，则存在 $y^{(1)}, y^{(2)} \in C$ 使得 $x^{(i)} = Ay^{(i)}$。对于任意 $\lambda \in [0, 1]$：
$$
\lambda x^{(1)} + (1 - \lambda)x^{(2)} = \lambda Ay^{(1)} + (1 - \lambda)Ay^{(2)} = A[\lambda y^{(1)} + (1 - \lambda)y^{(2)}]
$$
因为 $C$ 是凸集，所以括号内的项属于 $C$。因此其线性变换后的项属于 $S$。

**题目 (习题 3)**：证明 $S = \{x \mid x = Ay, y \ge 0\}$ 是凸集。
**解**：这是习题 2 的特例。集合 $C = \{y \mid y \ge 0\}$ 是凸锥（凸集），其线性变换像 $S$ 也是凸集。

### 习题 4 & 8：凸组合性质与 Jensen 不等式

**题目**：设 $S$ 是凸集，证明对任意整数 $k \ge 2$，若 $x^{(i)} \in S$，则 $\sum_{i=1}^k \lambda_i x^{(i)} \in S$ ($\sum \lambda_i = 1, \lambda_i \ge 0$)。

**解**：使用数学归纳法。
1. $k=2$ 时，由凸集定义成立。
2. 假设 $k=m$ 时成立。对于 $k=m+1$：
   若 $\lambda_{m+1}=1$ 则显然成立。若 $\lambda_{m+1} < 1$，令 $\mu = 1 - \lambda_{m+1}$。
   $$
   x = \mu \underbrace{\sum_{i=1}^m \frac{\lambda_i}{\mu} x^{(i)}}_{\in S (\text{归纳假设})} + \lambda_{m+1} x^{(m+1)}
   $$
   此为两点的凸组合，故 $x \in S$。

*(注：习题 8 为该性质在凸函数上的应用，即 Jensen 不等式，证明逻辑完全一致，仅将 $\in S$ 替换为函数值不等式 $f(\dots) \le \dots$)*

### 习题 5：凸函数的判定

**题目**：判别下列函数是否为凸函数：
1. $f = x_1^2 - 2x_1 x_2 + x_2^2 + x_1 + x_2$
2. $f = x_1^2 - 4x_1 x_2 + x_2^2 + x_1 + x_2$

**解**：
**(1)** Hesse 矩阵 $\nabla^2 f = \begin{pmatrix} 2 & -2 \\ -2 & 2 \end{pmatrix}$。$D_1 > 0, D_2 = 0$，半正定 $\implies$ **凸函数**。
**(2)** Hesse 矩阵 $\nabla^2 f = \begin{pmatrix} 2 & -4 \\ -4 & 2 \end{pmatrix}$。$D_2 = -12 < 0$，不定 $\implies$ **非凸非凹**。

### 习题 7：二次函数的严格凸性

**题目**：证明 $f(x) = \frac{1}{2}x^T Ax + b^T x$ 为严格凸函数的充要条件是 $A$ 正定。

**解**：
$$
\nabla f(x) = Ax + b, \quad \nabla^2 f(x) = A
$$
根据判定定理，$f(x)$ 为严格凸 $\iff \nabla^2 f(x)$ 正定 $\iff A$ 正定。

### 习题 9：凸函数的极大值性质

**题目**：设 $f$ 是 $\mathbb{R}^n$ 上的凸函数，证明：如果在全空间某点 $\bar{x}$ 处具有全局极大值，则 $f(x)$ 为常数。

**解**：反证法。
假设 $f(x)$ 不是常数，则存在点 $y$ 使得 $f(y) < f(\bar{x})$。
取点 $z = 2\bar{x} - y$（即 $\bar{x}$ 是 $y, z$ 的中点）。由凸性：
$$
f(\bar{x}) \le \frac{1}{2}f(y) + \frac{1}{2}f(z)
$$
因为 $f(y) < f(\bar{x})$ 且 $f(z) \le f(\bar{x})$（极大值性质）：
$$
\frac{1}{2}f(y) + \frac{1}{2}f(z) < \frac{1}{2}f(\bar{x}) + \frac{1}{2}f(\bar{x}) = f(\bar{x})
$$
导出 $f(\bar{x}) < f(\bar{x})$，矛盾。故 $f(x)$ 必为常数。
