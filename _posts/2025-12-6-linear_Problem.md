---
title: 1. 线性回归与优化基础：从数学推导到梯度下降
date: 2025-12-06 21:00:00 +0800
categories: [学习笔记, 机器学习]
tags: [线性回归, 梯度下降, 数学推导, 优化算法, 最小二乘法]
math: true
mermaid: true
---

上一章我们建立了机器学习的宏观框架。本章我们将深入最基础但也最核心的算法——**线性回归 (Linear Regression)**。它是整个机器学习大厦的地基，理解了它，就理解了“模型、策略、算法”这三要素是如何运作的。

[cite_start]同时，我们还将引出机器学习中最重要的两种求解思路：**闭式解 (Closed-form Solution)** 与 **梯度下降 (Gradient Descent)** [cite: 1, 30-31, 48]。

## 1. 单变量线性回归 (Univariate Linear Regression)

这是最基础的回归模型，用于预测连续值。

### 1.1 直观理解
想象你在买房。影响房价 ($y$) 的因素有很多，假设我们只考虑**房屋面积** ($x$)。我们试图用一条直线去拟合这些数据：

$$f(x_i) = wx_i + b, \quad \text{使得 } f(x_i) \simeq y_i$$

* **$w$ (Weight)**：权重，代表面积对房价的影响力（斜率）。
* **$b$ (Bias)**：偏置，代表基础房价（截距）。



[Image of linear regression scatter plot with best fit line]


> [cite_start]**💡 数据处理小贴士** [cite: 72-73]：
> 如果属性是离散的怎么办？
> * **有“序” (Order)**：如“高/中/低”，可转化为连续值 $1.0/0.5/0.0$。
> * **无“序”**：如“西瓜/南瓜/黄瓜”，则需转化为 $k$ 维向量（One-hot 编码）。

### 1.2 策略：最小二乘法 (Least Squares)
我们如何衡量这条直线“准不准”？我们需要一把尺子，即**损失函数 (Loss Function)**。最常用的是**均方误差 (MSE)**。

$$
E_{(w,b)} = \sum_{i=1}^{m}\left(f\left(x_{i}\right)-y_{i}\right)^{2} = \sum_{i=1}^{m}\left(y_{i}-w x_{i}-b\right)^{2}
$$

**为什么是“均方” (Squared)？**
1.  **消除符号影响**：误差 $-10$ 和 $+10$ 应该被同等对待，平方将其转为正数。
2.  **惩罚大误差**：平方会放大大的错误。错 $1$ 块钱惩罚 $1$，错 $10$ 块钱惩罚 $100$。这迫使模型拼命避免犯大错。
3.  **数学便利**：平方函数是光滑的凸函数，处处可导，方便求极值。

### 1.3 求解：闭式解 (Closed-form Solution)
[cite_start]目标是找到一组 $(w^*, b^*)$ 使得损失函数最小 [cite: 74]。这是一个典型的求极值问题。

[cite_start]**对 $w$ 和 $b$ 分别求导并令其为 0：** [cite: 76-78]

$$
\begin{cases}
\frac{\partial E_{(w, b)}}{\partial w} = 2\left(w \sum_{i=1}^{m} x_{i}^{2}-\sum_{i=1}^{m}\left(y_{i}-b\right) x_{i}\right) = 0 \\
\frac{\partial E_{(w, b)}}{\partial b} = 2\left(m b-\sum_{i=1}^{m}\left(y_{i}-w x_{i}\right)\right) = 0
\end{cases}
$$

[cite_start]**解方程组得到闭式解：** [cite: 79]

$$
w = \frac{\sum_{i=1}^{m} y_{i}\left(x_{i}-\bar{x}\right)}{\sum_{i=1}^{m} x_{i}^{2}-\frac{1}{m}\left(\sum_{i=1}^{m} x_{i}\right)^{2}}, \quad b = \frac{1}{m} \sum_{i=1}^{m}\left(y_{i}-w x_{i}\right)
$$

[cite_start]这种能够直接用公式算出来的解，称为**闭式解**。但在现实中，这种完美情况“可遇而不可求” [cite: 31, 44]。

---

## 2. 多元线性回归 (Multivariate Linear Regression)

现实中，影响房价的除了面积，还有房龄、距离地铁距离等。此时 $x_i$ 变成了一个向量。

### 2.1 向量化与 Bias Trick
$$f(\boldsymbol{x}_i) = \boldsymbol{w}^T \boldsymbol{x}_i + b$$

[cite_start]为了简化计算，我们使用 **"Bias Trick"**：将 $b$ 吸收到 $w$ 中（$w \leftarrow (w; b)$），并在输入 $x$ 的最后加一列全为 1 的特征（$x \leftarrow (x; 1)$）[cite: 82-85]。

数据集表示为矩阵 $\mathbf{X}$（$m$ 行 $d+1$ 列）：
$$
\mathbf{X}=\left(\begin{array}{cccc|c}
x_{11} & x_{12} & \cdots & x_{1 d} & 1 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
x_{m 1} & x_{m 2} & \cdots & x_{m d} & 1
\end{array}\right), \quad \boldsymbol{y}=\left(y_{1} ; \ldots ; y_{m}\right)
$$

### 2.2 矩阵求导与正规方程
[cite_start]损失函数变为矩阵范数形式 [cite: 87-88]：

$$E_{\hat{\boldsymbol{w}}} = (\boldsymbol{y} - \mathbf{X}\hat{\boldsymbol{w}})^T (\boldsymbol{y} - \mathbf{X}\hat{\boldsymbol{w}})$$

[cite_start]利用矩阵求导公式 $\frac{\partial (\boldsymbol{x}^T \mathbf{A} \boldsymbol{x})}{\partial \boldsymbol{x}} = (\mathbf{A} + \mathbf{A}^T)\boldsymbol{x}$，对 $\hat{\boldsymbol{w}}$ 求导并令其为 0 [cite: 89-90]：

$$
\frac{\partial E_{\hat{\boldsymbol{w}}}}{\partial \hat{\boldsymbol{w}}} = 2\mathbf{X}^T (\mathbf{X}\hat{\boldsymbol{w}} - \boldsymbol{y}) = 0
$$

得到著名的**正规方程 (Normal Equation)**：

$$
\mathbf{X}^T \mathbf{X} \hat{\boldsymbol{w}} = \mathbf{X}^T \boldsymbol{y}
$$

[cite_start]最终解为 [cite: 94]：

$$
\hat{\boldsymbol{w}}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \boldsymbol{y}
$$

### 2.3 现实的骨感：矩阵不可逆怎么办？
这个公式看起来很完美，但它隐含了一个巨大的坑：**求逆矩阵 $(\mathbf{X}^T \mathbf{X})^{-1}$**。

1.  **计算成本**：矩阵求逆的复杂度约为 $O(d^3)$。当特征维度 $d$ 很大时，计算机根本算不动。
2.  [cite_start]**奇异矩阵 (Singular Matrix)** [cite: 91, 95]：如果 $\mathbf{X}^T \mathbf{X}$ 不满秩（不可逆），方程就有无穷多解。
    * **物理意义**：这通常意味着特征之间存在**多重共线性 (Multicollinearity)**。例如，你的特征里同时有“房屋面积（平方米）”和“房屋面积（平方尺）”，这两个特征完全线性相关，导致方程无法通过单一解来区分权重。
    * [cite_start]**解决方案**：引入**正则化 (Regularization)**（如 L2 正则化/Ridge 回归），给矩阵对角线加一个小数值，使其可逆 [cite: 96]。

---

## 3. 广义线性模型 (Generalized Linear Models)

如果我们不想直接预测 $y$，而是预测 $y$ 的变形呢？
[cite_start]例如，房价通常随着面积指数级增长，或者我们做的是分类任务（输出 0 或 1）。这时我们可以使用**联系函数 (Link Function)** $g(\cdot)$ [cite: 110-112]。

$$y = g^{-1}(\boldsymbol{w}^T \boldsymbol{x} + b)$$

[cite_start]**典型案例：对数线性回归 (Log-Linear Regression)** [cite: 104-106]
$$\ln y = \boldsymbol{w}^T \boldsymbol{x} + b$$
[cite_start]这实际上是让 $e^{\boldsymbol{w}^T \boldsymbol{x} + b}$ 逼近 $y$ [cite: 107-108]。这一思想非常关键，它为下一章的**逻辑回归 (Logistic Regression)** 奠定了基础——逻辑回归本质上就是用 Sigmoid 函数作为联系函数的广义线性模型。

---

## 4. 优化方法：梯度下降 (Gradient Descent)

既然闭式解计算量大且可能无解，我们需要一种通用的数值优化方法：**梯度下降**。

### 4.1 为什么要用梯度下降？
1.  [cite_start]**通用性**：绝大多数复杂模型（如深度学习）根本没有闭式解 [cite: 31, 44]。
2.  **避开矩阵求逆**：直接通过迭代逼近最优解。

### 4.2 直观理解：下山的故事
想象你被困这就好比你在山上（高误差），想要下到山谷（低误差）。
1.  **环顾四周**：找到坡度最陡的方向（梯度的反方向）。
2.  **迈出一步**：这一步的大小由**学习率 (Learning Rate, $\eta$)** 决定。
3.  **重复**：直到到达谷底。



### 4.3 数学原理：泰勒展开
[cite_start]核心思想利用了**泰勒展开 (Taylor Expansion)** 近似 [cite: 21]。
[cite_start]若想让函数值 $f(x)$ 下降最快，更新方向 $u$ 应与梯度 $\nabla f(x_0)$ 方向**相反**（夹角 180 度）[cite: 36-39]。

[cite_start]**迭代公式：** [cite: 48]

$$x_{t+1} = x_t - \eta \nabla f(x_t)$$

* **$\nabla f(x_t)$**：告诉我们往哪里走。
* **$\eta$ (Eta)**：告诉我们走多远。

```mermaid
graph TD
    A[初始化参数 w, b] --> B[计算损失函数的梯度]
    B --> C[沿梯度反方向更新参数]
    C --> D{梯度是否趋近于0?}
    D -- No --> B
    D -- Yes --> E[停止迭代，输出最优解]
