---
title: 0. 机器学习宏观架构全景图：从古典统计到大模型
date: 2025-12-06 20:00:00 +0800
categories: [学习笔记, 机器学习]
tags: [AI, 基础理论, 架构图谱, Transformer, ResNet]
math: true
mermaid: true
pin: true
---

在深入具体的算法（如 SVM、神经网络）之前，建立一个清晰的**宏观框架**至关重要。作为机器学习系列的第一篇笔记，本文旨在理清机器学习的核心定义、主要流派，并梳理从古典统计模型到现代大模型的技术演进脉络。

## 1. 什么是机器学习？

从数学角度看，机器学习的本质是寻找一个函数映射：

$$y = f(x; \theta)$$

* $x$：输入数据（特征向量）。
* $y$：输出结果（预测标签或数值）。
* $f$：模型（Model），即假设空间。
* $\theta$：参数（Parameters），是机器需要“学习”的对象。

**核心思想**：我们不再像传统编程那样手动编写规则（Rule-based），而是构建一个包含参数的框架，通过数据驱动（Data-driven）的方式，利用优化算法找到最优的参数 $\theta^*$，使得预测误差最小。

---

## 2. 核心流派与任务分类

根据**数据标签（Label）的存在与否**以及**反馈机制**的不同，可以将机器学习的版图划分为以下几大块：

### 2.1 监督学习 (Supervised Learning)
**特点**：数据集中包含明确的标签 $(x, y)$，就像老师教学生。
* **回归 (Regression)**：预测连续数值。
  * *典型场景*：房价预测、股票趋势。
  * *算法*：线性回归 (Linear Regression)。
* **分类 (Classification)**：预测离散类别。
  * *典型场景*：垃圾邮件识别、图像分类 (ImageNet)。
  * *算法*：逻辑回归 (Logistic Regression)、SVM、决策树。

### 2.2 无监督学习 (Unsupervised Learning)
**特点**：数据集中没有标签，只有 $x$。模型需要自己发现数据内部的结构。
* **聚类 (Clustering)**：将相似的数据归堆。
  * *算法*：K-Means、**EM 算法** (如高斯混合模型 GMM)。
* **降维 (Dimensionality Reduction)**：提取关键特征，压缩数据。
  * *算法*：**PCA** (主成分分析，保留最大方差)、t-SNE (可视化)。

### 2.3 半监督学习 (Semi-supervised Learning)
**特点**：少量有标签数据 + 大量无标签数据。
* *场景*：医学影像分析（请医生标注很贵，但医院有大量未标注片子）。

---

## 3. 技术演进：从统计学到大模型

为了更好地理解各种名词（SVM, ResNet, Transformer...），我们将机器学习的发展分为三个阶段。

### 第一阶段：古典机器学习 (Classical ML)
> **关键词：严谨数学、表格数据、特征工程**

这一阶段主要处理结构化数据（Excel 表格类），是数据挖掘竞赛（如 Kaggle）的基石。

1.  **线性模型**：
    * **逻辑回归**：虽然叫回归，实际是分类。通过 Sigmoid 函数将输出映射到 $(0,1)$ 概率区间。
    * **LDA (线性判别分析)**：一种监督降维技术，目的是让同类数据更紧凑，异类数据分得更开。

2.  **支持向量机 (SVM)**：
    * **核心逻辑**：寻找一个超平面 (Hyperplane)，最大化不同类别之间的**间隔 (Margin)**。
    * **核技巧 (Kernel Trick)**：SVM 的灵魂。能将低维不可分的数据映射到高维空间使其线性可分。

3.  **决策树与集成学习 (Ensemble Learning)**：
    * **决策树 (DT)**：像人类思维一样的 `if-then` 规则链。
    * **集成学习**：这是目前处理表格数据的最强王者。
        * **Bagging**：并行训练（如**随机森林**），少数服从多数，降低方差。
        * **Boosting**：串行训练（如 XGBoost, LightGBM），专门修正前一个模型的错误，降低偏差。

### 第二阶段：深度学习 (Deep Learning)
> **关键词：神经网络、感知、非结构化数据**

这一阶段主要解决图像、声音等非结构化数据，核心是**特征的自动提取**。

1.  **基础组件**：
    * **MLP (多层感知机)**：全连接网络，深度学习的原子结构。
    * **CNN (卷积神经网络)**：视觉领域的霸主。通过**卷积核**提取局部特征（边缘、纹理），具有平移不变性。

2.  **里程碑模型：ResNet (残差网络)**：
    * **背景**：深层网络面临梯度消失和退化问题。
    * **核心**：引入**Shortcut Connection**，让网络学习残差 $F(x) = H(x) - x$。
    * **意义**：使得训练 100 层甚至 1000 层的网络成为可能，是现代视觉模型的标配 Backbone。

3.  **图神经网络 (GNN)**：
    * 处理非欧几里得数据（社交网络、分子结构）。通过**消息传递 (Message Passing)** 聚合邻居节点信息。

4.  **序列模型**：
    * **HMM (隐马尔可夫模型)**：前深度学习时代的序列霸主。
    * **RNN/LSTM**：通过循环结构处理时间序列数据。

### 第三阶段：生成式 AI 与大模型 (GenAI & LLM)
> **关键词：Transformer、注意力机制、通用智能**

这是当下的最前沿，核心是从“判别”转向“生成”。

1.  **Transformer 架构**：
    * **QKV 机制 (Query, Key, Value)**：即**自注意力 (Self-Attention)**。让模型能够动态捕捉序列中任意两个位置的关联，解决了长距离依赖问题。
    * 抛弃了循环结构，实现了完全并行计算。

2.  **视觉的新星：ViT (Vision Transformer)**：
    * 将图片切分为 Patch（小方块），视为单词序列输入 Transformer。证明了 CNN 的归纳偏置并不是必须的。

3.  **大语言模型 (LLM)**：
    * 基于 Transformer Decoder (如 GPT) 或 Encoder-Decoder (如 T5)。
    * 通过海量数据的 **Next Token Prediction** 训练，涌现出了推理、代码生成等能力。

---
mindmap
  root((机器学习<br/>模型全景))
    [:fa-book: 监督学习<br/>(古典方法)]
      (回归任务<br/>Regression)
        **线性回归**
        多项式回归
        Ridge/Lasso回归
      (分类任务<br/>Classification)
        逻辑回归
        支持向量机(SVM)
        k近邻(k-NN)
        朴素贝叶斯
      (集成学习<br/>Ensemble)
        决策树(基分类器)
        随机森林(Bagging)
        XGBoost/LightGBM(Boosting)
    [:fa-connectdevelop: 无监督学习<br/>(数据挖掘)]
      (聚类<br/>Clustering)
        K-Means
        高斯混合模型(GMM)
        DBSCAN(密度聚类)
      (降维<br/>Dim Reduction)
        PCA(主成分分析)
        LDA(线性判别)
        t-SNE(可视化)
    [:fa-brain: 深度学习<br/>(神经网络)]
      (基础架构)
        MLP(多层感知机)
      (计算机视觉<br/>CV)
        CNN(卷积神经网络)
        ResNet(残差网络)
        ViT(Vision Transformer)
      (自然语言处理<br/>NLP)
        RNN / LSTM
        Transformer(基石)
        BERT / GPT(大模型)
      (图与生成)
        GNN(图神经网络)
        Diffusion(扩散模型)
    [:fa-gamepad: 强化学习<br/>(决策控制)]
      Q-Learning
      DQN(深度Q网络)
      PPO(策略梯度)
      
## 4. 机器学习的标准流程 (Pipeline)

一个完整的机器学习项目通常遵循以下流水线，这也将是我后续博客更新的顺序：

```mermaid
graph LR
    A[数据获取] --> B[数据预处理]
    B --> C[特征工程]
    C --> D[模型选择与训练]
    D --> E[模型评估]
    E --> F{效果达标?}
    F -- Yes --> G[部署上线]
    F -- No --> C
