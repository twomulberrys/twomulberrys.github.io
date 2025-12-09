---
title: 1. 线性回归与优化基础：从数学推导到梯度下降
date: 2025-12-06 21:00:00 +0800
categories: [学习笔记, 机器学习]
tags: [线性回归, 梯度下降, 数学推导, 优化算法, 最小二乘法]
math: true
mermaid: true
---

上一章我们建立了机器学习的宏观框架。本章我们将深入最基础但也最核心的算法——**线性回归 (Linear Regression)**。它是整个机器学习大厦的地基，理解了它，就理解了“模型、策略、算法”这三要素是如何运作的。

## 1. 什么是线性模型？

### 1.1 直观理解
想象你在买房。影响房价的因素有很多：面积、离地铁距离、房龄等。如果我们假设房价 ($y$) 与这些因素 ($x$) 之间存在一种简单的“加权求和”关系，这就是线性模型。

$$
f(x) = w_1x_1 + w_2x_2 + \dots + w_d x_d + b
$$

* **$w$ (Weight)**：权重，代表每个特征的重要性（比如面积的权重肯定比朝向大）。
* **$b$ (Bias)**：偏置，代表一种基础水平（即使面积为0，地皮也值个价）。

### 1.2 向量化形式 (The Vector Form)
为了数学上的简洁（以及计算机计算的便利），我们通常使用向量形式表达。令 $w = (w_1; w_2; \dots; w_d)$，$x = (x_1; x_2; \dots; x_d)$，则：

$$
f(x) = w^T x + b
$$

这种形式简单、易懂，且具有极好的**可解释性 (Interpretability)**。如果你求出的 $w_{\text{面积}}$ 是正数，你就可以确信地说：面积越大，房价越高。

---

## 2. 策略：如何衡量模型好坏？

有了模型 $f(x)$，我们如何知道它通过了一组参数 $(w, b)$ 预测出来的结果准不准？我们需要一把尺子，这就是**损失函数 (Loss Function)**。

在回归任务中，最常用的尺子是**均方误差 (Mean Squared Error, MSE)**。

$$
E_{(w,b)} = \sum_{i=1}^{m} (y_i - f(x_i))^2 = \sum_{i=1}^{m} (y_i - (w x_i + b))^2
$$

> **💡 为什么要用平方？**
> * **消除符号影响**：误差 -10 和 +10 应该被同等对待，平方将其转为正数。
> * **惩罚大误差**：平方会放大大的错误。错 1 块钱惩罚 1，错 10 块钱惩罚 100。这迫使模型拼命避免犯大错。
> * **数学便利**：平方函数是光滑的凸函数，处处可导，方便我们求极值。

---

## 3. 算法：最小二乘法 (Least Squares)

我们的目标很明确：找到一组 $w^*$ and $b^*$，让上面的均方误差 $E$ 最小。这就变成了一个数学上的**最优化问题**。

### 3.1 简单的标量推导 (Scalar Form)
对于最简单的情况（只有一个特征），我们可以利用微积分中的求导。均方误差 $E$ 是关于 $w$ 和 $b$ 的凸函数，**导数为 0 的点就是极值点**。

分别对 $w$ 和 $b$ 求偏导并令其为 0：

$$
\begin{cases}
\frac{\partial E}{\partial w} = 2 \sum_{i=1}^{m} (w x_i + b - y_i) x_i = 0 \\
\frac{\partial E}{\partial b} = 2 \sum_{i=1}^{m} (w x_i + b - y_i) = 0
\end{cases}
$$

通过联立方程组求解（具体代数运算略），我们可以直接得到 $w$ 和 $b$ 的**闭式解 (Closed-form Solution)**。这意味着我们不需要像神经网络那样一轮轮训练，直接套公式就能算出答案。

### 3.2 优雅的矩阵推导 (Matrix Form) —— 核心部分
在实际应用中，数据通常是多维的。为了推导更加优雅，我们将参数 $b$ 吸收到 $w$ 中（在 $x$ 向量最后加一个恒为 1 的特征）。

* **数据集 $X$**：$m \times (d+1)$ 的矩阵（每一行是一个样本）。
* **标签 $y$**：$m \times 1$ 的向量。
* **参数 $\hat{w}$**：$(d+1) \times 1$ 的向量。

此时，损失函数变为矩阵范数形式：

$$
E_{\hat{w}} = (y - X\hat{w})^T (y - X\hat{w})
$$

**🔥 高能预警：矩阵求导**
为了求极值，我们对 $\hat{w}$ 求导：

1.  **展开公式**：
    $$E_{\hat{w}} = y^Ty - y^TX\hat{w} - \hat{w}^TX^Ty + \hat{w}^TX^TX\hat{w}$$
2.  利用矩阵求导公式 $\frac{\partial (x^TAx)}{\partial x} = (A+A^T)x$，对 $\hat{w}$ 求导并令其为 0：
    $$
    \frac{\partial E_{\hat{w}}}{\partial \hat{w}} = 2X^T(X\hat{w} - y) = 0
    $$
3.  化简得到**正规方程 (Normal Equation)**：
    $$
    X^T X \hat{w} = X^T y
    $$
4.  最终得到最优解：
    $$
    \hat{w}^* = (X^T X)^{-1} X^T y
    $$

> **⚠️ 现实的骨感：**
> 这个公式看起来完美，但它有一个前提：矩阵 $X^T X$ 必须是**可逆**（满秩）的。
> * **物理意义**：如果特征比样本还多（$d > m$），或者特征之间存在**多重共线性**（比如“左脚鞋码”和“右脚鞋码”两个特征同时存在，它们完全线性相关），矩阵就会不可逆。
> * **解决方案**：引入**正则化 (Regularization)**，即 Ridge 回归（加入 $\lambda I$ 扰动项）或 Lasso 回归，给矩阵对角线加一个小数值，使其可逆。

---

## 4. 另一种视角：梯度下降 (Gradient Descent)

虽然线性回归有完美的公式解（最小二乘法），但在深度学习中，我们几乎从不用它。为什么？

* **计算代价**：矩阵求逆 $(X^T X)^{-1}$ 的计算复杂度约为 $O(d^3)$。当特征维度 $d$ 达到万级、亿级时，计算机根本算不动。
* **通用性**：绝大多数复杂的模型（如神经网络）根本没有闭式解。

因此，我们引入了**梯度下降** —— 一种数值迭代优化的方法。

> **⛰️ 比喻**：
> 不仅是想一步登天（直接求闭式解），而是像下山一样。环顾四周，找到坡度最陡的方向（梯度的反方向），往下走一步。重复这个过程，终点就是山谷（损失函数最小值）。

**迭代公式：**

$$
x_{t+1} = x_t - \eta \nabla f(x_t)
$$

其中 $\eta$ 是**学习率 (Learning Rate)**，决定了我们下山的步子迈多大。步子太大容易“跨过”山谷，步子太小则下山太慢。

---

## 5. 深度学习的伏笔：从线性回归到神经元

最后，让我们把视角拉高。如果我们将线性回归的结构画出来，它其实就是**神经网络中最简单的神经元**。

```mermaid
graph LR
    x1((x1)) -- w1 --> Sum((Σ))
    x2((x2)) -- w2 --> Sum
    xd((xd)) -- wd --> Sum
    b -- 1 --> Sum
    Sum -- Linear --> Output(y)
    
    style Sum fill:#f9f,stroke:#333,stroke-width:2px

## 6. 多元线性回归 (Multivariate Linear Regression)

现实中，影响房价的除了面积，还有房龄、距离地铁距离等。此时 $x_i$ 变成了一个向量。

### 6.1 向量化与 Bias Trick
$$f(\boldsymbol{x}_i) = \boldsymbol{w}^T \boldsymbol{x}_i + b$$

为了简化计算，我们使用 **"Bias Trick"**：将 $b$ 吸收到 $w$ 中（$w \leftarrow (w; b)$），并在输入 $x$ 的最后加一列全为 1 的特征（$x \leftarrow (x; 1)$）。

数据集表示为矩阵 $\mathbf{X}$（$m$ 行 $d+1$ 列）：
$$
\mathbf{X}=\left(\begin{array}{cccc|c}
x_{11} & x_{12} & \cdots & x_{1 d} & 1 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
x_{m 1} & x_{m 2} & \cdots & x_{m d} & 1
\end{array}\right), \quad \boldsymbol{y}=\left(y_{1} ; \ldots ; y_{m}\right)
$$

### 6.2 矩阵求导与正规方程
损失函数变为矩阵范数形式：

$$E_{\hat{\boldsymbol{w}}} = (\boldsymbol{y} - \mathbf{X}\hat{\boldsymbol{w}})^T (\boldsymbol{y} - \mathbf{X}\hat{\boldsymbol{w}})$$

利用矩阵求导公式 $\frac{\partial (\boldsymbol{x}^T \mathbf{A} \boldsymbol{x})}{\partial \boldsymbol{x}} = (\mathbf{A} + \mathbf{A}^T)\boldsymbol{x}$，对 $\hat{\boldsymbol{w}}$ 求导并令其为 0：

$$
\frac{\partial E_{\hat{\boldsymbol{w}}}}{\partial \hat{\boldsymbol{w}}} = 2\mathbf{X}^T (\mathbf{X}\hat{\boldsymbol{w}} - \boldsymbol{y}) = 0
$$

得到著名的**正规方程 (Normal Equation)**：

$$
\mathbf{X}^T \mathbf{X} \hat{\boldsymbol{w}} = \mathbf{X}^T \boldsymbol{y}
$$

最终解为：

$$
\hat{\boldsymbol{w}}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \boldsymbol{y}
$$

### 6.3 现实的骨感：矩阵不可逆怎么办？
这个公式看起来很完美，但它隐含了一个巨大的坑：**求逆矩阵 $(\mathbf{X}^T \mathbf{X})^{-1}$**。

1.  **计算成本**：矩阵求逆的复杂度约为 $O(d^3)$。当特征维度 $d$ 很大时，计算机根本算不动。
2.  **奇异矩阵 (Singular Matrix)** [cite: 91, 95]：如果 $\mathbf{X}^T \mathbf{X}$ 不满秩（不可逆），方程就有无穷多解。
    * **物理意义**：这通常意味着特征之间存在**多重共线性 (Multicollinearity)**。例如，你的特征里同时有“房屋面积（平方米）”和“房屋面积（平方尺）”，这两个特征完全线性相关，导致方程无法通过单一解来区分权重。
    * [cite_start]**解决方案**：引入**正则化 (Regularization)**（如 L2 正则化/Ridge 回归），给矩阵对角线加一个小数值，使其可逆 [cite: 96]。

---

## 7. 广义线性模型 (Generalized Linear Models)

如果我们不想直接预测 $y$，而是预测 $y$ 的变形呢？
例如，房价通常随着面积指数级增长，或者我们做的是分类任务（输出 0 或 1）。这时我们可以使用**联系函数 (Link Function)** $g(\cdot)$ 。

$$y = g^{-1}(\boldsymbol{w}^T \boldsymbol{x} + b)$$

**典型案例：对数线性回归 (Log-Linear Regression)** 
$$\ln y = \boldsymbol{w}^T \boldsymbol{x} + b$$
这实际上是让 $e^{\boldsymbol{w}^T \boldsymbol{x} + b}$ 逼近 $y$ 。这一思想非常关键，它为下一章的**逻辑回归 (Logistic Regression)** 奠定了基础——逻辑回归本质上就是用 Sigmoid 函数作为联系函数的广义线性模型。

---

## 8. 优化方法：梯度下降 (Gradient Descent)

既然闭式解计算量大且可能无解，我们需要一种通用的数值优化方法：**梯度下降**。

### 8.1 为什么要用梯度下降？
1.  **通用性**：绝大多数复杂模型（如深度学习）根本没有闭式解。
2.  **避开矩阵求逆**：直接通过迭代逼近最优解。

### 8.2 直观理解：下山的故事
想象你被困这就好比你在山上（高误差），想要下到山谷（低误差）。
1.  **环顾四周**：找到坡度最陡的方向（梯度的反方向）。
2.  **迈出一步**：这一步的大小由**学习率 (Learning Rate, $\eta$)** 决定。
3.  **重复**：直到到达谷底。



### 8.3 数学原理：泰勒展开
核心思想利用了**泰勒展开 (Taylor Expansion)** 近似。
若想让函数值 $f(x)$ 下降最快，更新方向 $u$ 应与梯度 $\nabla f(x_0)$ 方向**相反**（夹角 180 度）。

**迭代公式：** 

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
