---
title: 最优化理论复习笔记：凸集与凸函数
date: 2025-12-04 17:00:00 +0800
categories: [学习笔记, 最优化]
tags: [数学, 凸优化, 期末复习]
math: true
---

最近在复习最优化理论，整理了第一章关于凸集与凸函数的核心定义、重要定理（如分离定理、Farkas 引理）以及典型习题的解题思路。

## 1. 凸集 (Convex Sets)

### 1.1 核心定义
**定义 1.1 (凸集)**：集合 $S \subseteq \mathbb{R}^n$ 是凸集，若对任意 $x^{(1)}, x^{(2)} \in S$ 及 $\lambda \in [0, 1]$，都有：
$$\lambda x^{(1)} + (1-\lambda)x^{(2)} \in S$$
[cite_start]即集合中任意两点的连线仍在集合内 [cite: 6, 7, 8]。

> [cite_start]**注**：线性规划的可行域（多面集）是典型的凸集 [cite: 9]。

### 1.2 运算性质
[cite_start]设 $S_1$ 和 $S_2$ 是两个凸集，$\beta$ 为实数，则以下集合均为凸集 [cite: 10]：
1.  [cite_start]**数乘**：$\beta S_1 = \{ \beta x \mid x \in S_1 \}$ [cite: 11]
2.  [cite_start]**交集**：$S_1 \cap S_2$ [cite: 12]
3.  [cite_start]**和集**：$S_1 + S_2 = \{ x^{(1)} + x^{(2)} \mid x^{(1)} \in S_1, x^{(2)} \in S_2 \}$ [cite: 14]
4.  [cite_start]**差集**：$S_1 - S_2 = \{ x^{(1)} - x^{(2)} \mid x^{(1)} \in S_1, x^{(2)} \in S_2 \}$ [cite: 16]

### 1.3 极点与方向
* [cite_start]**极点 (Extreme Point)**：设 $S$ 是非空凸集，$x \in S$。若 $x$ 不能表示成 $S$ 中两个不同点的凸组合（即若 $x = \lambda x^{(1)} + (1-\lambda)x^{(2)}$ 且 $\lambda \in (0,1)$，必推出 $x = x^{(1)} = x^{(2)}$），则称 $x$ 是极点 [cite: 18]。
* **多面集的方向**：设 $S = \{ x \mid Ax = b, x \ge 0 \}$，非零向量 $d$ 是 $S$ 的方向的充要条件是：
    $$d \ge 0, \quad Ad = 0$$
    [cite_start]如果可行域有方向 $d$ 且目标函数在该方向减少（$c^T d < 0$），则问题无界 [cite: 19, 20, 22]。

---

## 2. 凸集分离定理及其应用

这一部分是理论证明的核心，特别是 Farkas 定理，是后续对偶理论的基础。

### 2.1 投影定理
**定理 2.3 (投影定理)**：设 $S$ 是 $\mathbb{R}^n$ 中的闭凸集，$y \notin S$。则存在**唯一**的点 $\bar{x} \in S$，使得到 $y$ 的距离最小，即：
$$\| y - \bar{x} \| = \inf_{x \in S} \| y - x \|$$
[cite_start]$\bar{x}$ 称为 $y$ 在 $S$ 上的投影 [cite: 51, 52, 53]。

### 2.2 分离定理
**定理 2.1 (凸集分离定理)**：设 $S_1, S_2$ 是非空凸集且 $S_1 \cap S_2 = \emptyset$，则存在非零向量 $p$，使得：
$$\inf \{ p^T x \mid x \in S_1 \} \ge \sup \{ p^T x \mid x \in S_2 \}$$
[cite_start]即存在超平面分离这两个集合 [cite: 28, 29]。

**定理 2.2 (点与闭凸集分离定理)**：设 $S$ 是闭凸集，$y \notin S$。则存在非零向量 $p$ 及 $\epsilon > 0$，使得对任意 $x \in S$：
$$p^T y \ge \epsilon + p^T x$$
[cite_start]这意味着存在一个超平面**严格分离**点 $y$ 和集合 $S$ [cite: 33, 34]。

### 2.3 Farkas 定理 (择一性定理)
[cite_start]**定理 2.4**：设 $A$ 为 $m \times n$ 矩阵，$c$ 为 $n$ 维向量。则下列两个系统**有且仅有一个**有解 [cite: 59]：

* [cite_start]**系统 1**：$Ax \le 0, \quad c^T x > 0$ 有解 [cite: 60]。
* [cite_start]**系统 2**：$A^T y = c, \quad y \ge 0$ 有解 [cite: 61]。

> [cite_start]**证明思路**：构造闭凸锥 $S = \{ z \mid z = A^T y, y \ge 0 \}$。若 $c \notin S$，利用点与凸集分离定理构造分离超平面，从而推导出系统 1 的解 [cite: 63, 65, 68]。

---

## 3. 凸函数 (Convex Functions)

### 3.1 定义与判定
**定义**：定义在凸集 $S$ 上的函数 $f(x)$ 是凸函数，若对任意 $x^{(1)}, x^{(2)} \in S$ 及 $\lambda \in [0, 1]$：
[cite_start]$$f(\lambda x^{(1)} + (1-\lambda)x^{(2)}) \le \lambda f(x^{(1)}) + (1-\lambda)f(x^{(2)})$$ [cite: 76, 77]

**判定定理 (二阶条件)**：
[cite_start]设 $f \in C^2(S)$，则 $f$ 是凸函数的充要条件是 Hesse 矩阵 $\nabla^2 f(x)$ 在 $S$ 上**半正定**。若 $\nabla^2 f(x)$ 正定，则 $f$ 是严格凸函数 [cite: 81, 82]。

### 3.2 凸规划
**模型**：求凸函数在凸集上的极小点。
$$\begin{aligned} \min \quad & f(x) \\ \text{s.t.} \quad & g_i(x) \ge 0, \quad i=1,\dots,m \\ & h_j(x) = 0, \quad j=1,\dots,l \end{aligned}$$
[cite_start]其中 $f(x)$ 为凸函数，$g_i(x)$ 为凹函数（保证可行域是凸集），$h_j(x)$ 为仿射函数 [cite: 86, 91, 92, 93]。

[cite_start]**基本定理**：凸规划的局部极小点就是全局极小点，且极小点的集合是凸集 [cite: 95, 96, 97]。

---

## 4. 重点习题详解

### 习题 1：Hesse 矩阵判定法
[cite_start]**题目**：判别 $f = x_1^2 - 2x_1 x_2 + x_2^2 + x_1 + x_2$ 是否为凸函数 [cite: 135, 136]。

**解**：
计算 Hesse 矩阵：
$$\nabla^2 f(x) = \begin{pmatrix} 2 & -2 \\ -2 & 2 \end{pmatrix}$$
计算主子式：
* 一阶主子式 $D_1 = 2 > 0$
* 二阶主子式 $D_2 = 4 - 4 = 0$
[cite_start]矩阵半正定，故该函数是凸函数（非严格凸） [cite: 141, 142]。

### 习题 2：Jensen 不等式证明
[cite_start]**题目**：设 $f$ 是凸函数，证明 $f(\sum_{i=1}^k \lambda_i x^{(i)}) \le \sum_{i=1}^k \lambda_i f(x^{(i)})$，其中 $\sum \lambda_i = 1, \lambda_i \ge 0$ [cite: 157]。

[cite_start]**解**：使用数学归纳法 [cite: 158]。
1.  [cite_start]$k=2$ 时，由凸函数定义显然成立 [cite: 159]。
2.  假设 $k=m$ 时成立。对于 $k=m+1$，令 $\mu = 1 - \lambda_{m+1}$，构造 $y = \sum_{i=1}^m \frac{\lambda_i}{\mu} x^{(i)}$。
    利用凸函数性质拆分：
    $$f\left(\sum_{i=1}^{m+1} \lambda_i x^{(i)}\right) = f(\mu y + \lambda_{m+1} x^{(m+1)}) \le \mu f(y) + \lambda_{m+1} f(x^{(m+1)})$$
    [cite_start]再对 $f(y)$ 利用归纳假设即可证得 [cite: 161-165]。

### 习题 3：极大值性质
[cite_start]**题目**：证明若 $f$ 是 $\mathbb{R}^n$ 上的凸函数，且在 $\bar{x}$ 处取得全局极大值，则 $f(x)$ 为常数 [cite: 167]。

**解**：反证法。
假设存在 $y$ 使得 $f(y) < f(\bar{x})$。取 $z = 2\bar{x} - y$（即 $\bar{x}$ 是 $y$ 和 $z$ 的中点）。
由凸性：
$$f(\bar{x}) \le \frac{1}{2}f(y) + \frac{1}{2}f(z)$$
[cite_start]因为 $f(z) \le f(\bar{x})$ 且 $f(y) < f(\bar{x})$，代入后导出 $f(\bar{x}) < f(\bar{x})$ 的矛盾。故 $f(x)$ 必为常数 [cite: 169-173]。
