---
title: 1.线性回归与优化基础
date: 2025-12-06 21:00:00 +0800
categories: [学习笔记, 机器学习]
tags: [线性回归, 梯度下降, 数学推导, 优化算法]
math: true
mermaid: true
---

上一章我们建立了机器学习的宏观框架。本章我们将深入最基础但也最核心的算法——**线性回归 (Linear Regression)**。同时，我们还将引出机器学习中最重要的两种求解思路：**闭式解 (Closed-form Solution)** 与 **梯度下降 (Gradient Descent)**。

## 1. 单变量线性回归 (Univariate Linear Regression)

这是最基础的回归模型，用于预测连续值（例如：根据房屋面积预测房价）。

### 1.1 模型定义
试图学得一个属性的线性组合来进行预测：

$$f(x_i) = wx_i + b, \quad \text{使得 } f(x_i) \simeq y_i$$

* **离散属性处理**：若属性有“序”（order），则连续化（如：高/中/低 $\rightarrow$ 1.0/0.5/0.0）；否则转化为 $k$ 维向量（One-hot 编码）。

### 1.2 策略：最小二乘法 (Least Squares)
我们需要衡量预测值与真实值之间的差距。最常用的指标是**均方误差 (MSE)**。目标是找到一组 $(w^*, b^*)$ 使得损失函数最小：

$$
\begin{aligned}
(w^*, b^*) &= \underset{(w, b)}{\arg \min } \sum_{i=1}^{m}\left(f\left(x_{i}\right)-y_{i}\right)^{2} \\
&= \underset{(w, b)}{\arg \min } \sum_{i=1}^{m}\left(y_{i}-w x_{i}-b\right)^{2}
\end{aligned}
$$

### 1.3 求解：闭式解
令损失函数为 $E_{(w,b)}$，这是一个关于 $w$ 和 $b$ 的凸函数。我们可以通过求偏导并令其为 0 来求解。

**对 $w$ 和 $b$ 分别求导：**

$$
\begin{aligned}
\frac{\partial E_{(w, b)}}{\partial w} &= 2\left(w \sum_{i=1}^{m} x_{i}^{2}-\sum_{i=1}^{m}\left(y_{i}-b\right) x_{i}\right) \\
\frac{\partial E_{(w, b)}}{\partial b} &= 2\left(m b-\sum_{i=1}^{m}\left(y_{i}-w x_{i}\right)\right)
\end{aligned}
$$

**令导数为 0，得到闭式解 (Closed-form Solution)：**

$$
w = \frac{\sum_{i=1}^{m} y_{i}\left(x_{i}-\bar{x}\right)}{\sum_{i=1}^{m} x_{i}^{2}-\frac{1}{m}\left(\sum_{i=1}^{m} x_{i}\right)^{2}}, \quad b = \frac{1}{m} \sum_{i=1}^{m}\left(y_{i}-w x_{i}\right)
$$

---

## 2. 多元线性回归 (Multivariate Linear Regression)

在现实中，样本通常包含多个特征（如：房价取决于面积、房龄、距离地铁距离等）。此时 $x_i$ 变成了一个向量。

### 2.1 向量化表示
$$f(\boldsymbol{x}_i) = \boldsymbol{w}^T \boldsymbol{x}_i + b$$

为了简化计算，利用 **"Bias Trick"**，将 $w$ 和 $b$ 吸收入向量形式，令 $\hat{\boldsymbol{w}} = (\boldsymbol{w}; b)$，相应的输入数据矩阵 $\mathbf{X}$ 增加一列全是 1 的列。

数据集表示为：
$$
\mathbf{X}=\left(\begin{array}{cccc|c}
x_{11} & x_{12} & \cdots & x_{1 d} & 1 \\
x_{21} & x_{22} & \cdots & x_{2 d} & 1 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
x_{m 1} & x_{m 2} & \cdots & x_{m d} & 1
\end{array}\right), \quad \boldsymbol{y}=\left(y_{1} ; y_{2} ; \ldots ; y_{m}\right)
$$

### 2.2 矩阵求导
同样采用最小二乘法求解，目标函数变为矩阵形式：

$$\hat{\boldsymbol{w}}^* = \underset{\hat{\boldsymbol{w}}}{\arg \min } (\boldsymbol{y} - \mathbf{X}\hat{\boldsymbol{w}})^T (\boldsymbol{y} - \mathbf{X}\hat{\boldsymbol{w}})$$

令 $E_{\hat{\boldsymbol{w}}} = (\boldsymbol{y} - \mathbf{X}\hat{\boldsymbol{w}})^T (\boldsymbol{y} - \mathbf{X}\hat{\boldsymbol{w}})$，对向量 $\hat{\boldsymbol{w}}$ 求导：

$$
\frac{\partial E_{\hat{\boldsymbol{w}}}}{\partial \hat{\boldsymbol{w}}} = 2\mathbf{X}^T (\mathbf{X}\hat{\boldsymbol{w}} - \boldsymbol{y})
$$

令其为零，可得：

> **正规方程 (Normal Equation):**
>
> $$\hat{\boldsymbol{w}}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \boldsymbol{y}$$

**潜在问题**：
这里涉及矩阵求逆。
* 若 $\mathbf{X}^T \mathbf{X}$ **满秩**或正定，则可求出唯一解。
* 若 $\mathbf{X}^T \mathbf{X}$ **不满秩**（不可逆），则可能有多个解。此时需要引入**正则化 (Regularization)**（如 L1 Lasso 或 L2 Ridge）来约束参数，这将在后续章节详细讨论。

---

## 3. 广义线性模型 (Generalized Linear Models)

如果我们不想直接预测 $y$，而是预测 $y$ 的变形（例如 $y$ 的对数），可以使用**联系函数 (Link Function)** $g(\cdot)$。

$$y = g^{-1}(\boldsymbol{w}^T \boldsymbol{x} + b)$$

**典型例子：对数线性回归**
$$\ln y = \boldsymbol{w}^T \boldsymbol{x} + b$$
实际上是用 $e^{\boldsymbol{w}^T \boldsymbol{x} + b}$ 来逼近 $y$。这为后续的逻辑回归（Logistic Regression）奠定了基础。

---

## 4. 优化方法：从闭式解到梯度下降

在上面的线性回归中，我们要么直接推导出公式（闭式解），要么就需要通过迭代的方式逼近最优解（数值解）。

### 4.1 为什么要用梯度下降？
1.  **闭式解可遇不可求**：很多复杂模型（如神经网络）无法直接写出解析解（求导令为0解不出来）。
2.  **计算成本**：对于大规模数据，计算矩阵的逆 $(\mathbf{X}^T \mathbf{X})^{-1}$ 计算复杂度极高。

### 4.2 梯度下降的数学原理
核心思想是利用**泰勒展开 (Taylor Expansion)** 近似函数。

考虑函数在当前点 $x_0$ 走了一小步 $u$：
$$f(x_0 + u) \approx f(x_0) + \nabla f(x_0) \cdot u$$

* 想要函数值**变大**：$u$ 应与 $\nabla f(x_0)$ 方向一致（夹角0度）。
* 想要函数值**变小**：$u$ 应与 $\nabla f(x_0)$ 方向**相反**（夹角180度，$\cos 180^{\circ} = -1$）。

### 4.3 算法公式
为了寻找损失函数 $f(x)$ 的最小值，我们需要沿着梯度的**反方向**更新参数：

$$x_{t+1} = x_t - \eta \nabla f(x_t)$$

* **$\nabla f(x_t)$**：梯度，决定了更新的方向。
* **$\eta$ (Eta)**：步长（学习率 Learning Rate），决定了更新的幅度。

```mermaid
graph TD
    A[初始化参数 w, b] --> B[计算损失函数的梯度]
    B --> C[沿梯度反方向更新参数]
    C --> D{梯度是否趋近于0?}
    D -- No --> B
    D -- Yes --> E[停止迭代，输出最优参数]
