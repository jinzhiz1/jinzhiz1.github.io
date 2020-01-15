---
title: 线性回归 (linear regression)
tags:
  - ML
  - Algorithm
categories:
  - Tech
description: 线性回归的基本概念
date: 2019-12-01 11:24:32
---


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 定义
回归(Regression): 预测连续的输出值，比如房价和股票价格。
分类(Classification): 预测离散的输出值，比如经典的二分分类，预测有没有得癌症(0, 1). 也有超过两个结果的分类，比如预测癌症的类型(0, 1, 2 ...)。

# 单变量线性回归
## Notation
\\(m\\) = Number of training examples.
\\(x's\\) = "input" variabl / features.
\\(y's\\) = "output" variable / "target" variable.
\\((x, y)\\) - one training example. \\((x^{(i)}, y^{(i)})\\) - \\(i^{th}\\) traininng example.

## Model
假设(hypothesis): \\(h_{\theta}(x) = \theta_{0}+\theta_{1}x\\)

## cost function
\\(J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i=1}^m(h_{\theta}(x^{(i)}) - y^{(i)})^2\\)
找到一个\\(\theta_0\\)和\\(\theta_1\\)使对于所有的训练集，让\\(h_{\theta}(x)\\)最接近\\(y\\)。即 minimize \\(J(\theta_0, \theta_1)\\).
[图解cost function](https://d18ky98rnyall9.cloudfront.net/_ec21cea314b2ac7d9e627706501b5baa_Lecture2.pdf?Expires=1575331200&Signature=SZjP9VQb7rJ1MYapyVl1AGHoeX1d0WQpAU19cMPIkQlI2SqTeNH6gGRfwDelGi5ehRrSi8nVshNiepmAdlJASTfk4zE-kvGYkAE4K18fm8jW9bYAdPH1ll3d94K3o22VepdbqAag5xz1y3QtRSWfV8BOXrZxXPazfAofIPibCMA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)

# 多变量线性回归(mulitiple feature)
## Notation
\\(m\\) = Number of training examples.
\\(n\\) = Number of features.
\\(x^{(i)}\\) = input(features) of \\(i^{th}\\) training example.
\\(x_j^{(i)}\\) = value of feature \\(j\\) in \\(i^{th}\\) training example.
\\(y\\) = "output" variable / "target" variable.

## Model
假设(hypothesis): \\(h_{\theta}(x) = \theta_{0}+\theta_{1}x+\theta_{2}x+... = \theta^TX\\) which \\(x_0^{(i)} = 1\\)

# 多项式回归(polynomial regression)
有时候数据集里面只有一个变量，我们可以根据这个变量去定义一些新的变量从而得到更好的模型。
\\(h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_1x_2\\) 其中(\\(x_1=x\\), \\(x_2=x^2\\)). \\(h_\theta(x)\\)是房子的价格，\\(x\\)是房子的大小。

# Gradient descent
## 基本步骤：
1. 从\\(\theta_0, \theta_1\\)开始。
2. 持续更新\\(\theta_0, \theta_1\\)的值去减小\\(J(\theta_0, \theta_1)\\)直到得到一个最小值。

## 算法：
repeat util convergence { \\(\theta_j := \theta_j - \alpha\frac{d}{d_{\theta_j}}J(\theta_0, \theta_1)\\) }
注意，这里面的\\(\theta_j\\)是要同时更新：
\\(temp_0 := \theta_0 - \alpha\frac{d}{d_{\theta_0}}J(\theta_0, \theta_1)\\)
\\(temp_1 := \theta_1 - \alpha\frac{d}{d_{\theta_1}}J(\theta_0, \theta_1)\\)
\\(\theta_0 := temp_0\\)
\\(\theta_1 := temp_1\\)
而不是先更新\\(\theta_0\\), 再用新的\\(\theta_0\\)去更新\\(\theta_1\\)。

## 选参:
梯度下降算法中，我们需要去选择参数\\(\alpha\\). 如果\\(\alpha\\)太小，梯度下降收敛的速度会很慢。如果\\(\alpha\\)太大，梯度下降可能反而会发散。但是在计算过程中我们不需要去改变\\(\alpha\\)的值。只需要选到一个固定的\\(\alpha\\)值。
一般情况下，梯度下降只能找到局部最优。

## Gradient descent for single variable linear regression
repeat until convergence {
\\(\quad \theta_0 := \theta_0 - \alpha\frac{1}{m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})\\)
\\(\quad \theta_1 := \theta_1 - \alpha\frac{1}{m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})x^{(i)}\\)
}

梯度下降的优点是适用于数据集很大的问题，能够得到一个很好的模型。但是缺点是很慢。因为梯度下降的每次计算都要使用到所有的数据集。

## Gradient descent for multiple variables linear regression
repeat until convergence (\\(n \geq 1\\)) {
\\(\quad \theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}\\)
}
同时更新 \\(\theta_j (j = 0, ..., n)\\)

## Feature Scaling
目标：使变量(features)在一个相似的范围内. 比如使变量都大约在(\\(-1 \leq x_i \leq 1\\))。

### Mean normalization
\\(x_i = \frac{(x_i - \mu)}{\sigma}\\) 期中\\(\mu\\)是平均值，\\(\sigma\\)可以是方差，也可以是最大值减去最小值的差值。
通过用 (\\(x_i - \mu_i\\)) 替换 \\(x_i\\) 来使变量的平均值近似为0.（除了\\(x_0=1\\).

# 正态方程(normal equation)
用解析的方法得到\\(\theta\\). \\(\frac{d}{d\theta_j}J(\theta)=0\\).
通过计算 \\(\theta = (X^TX)^{-1}X^Ty\\) 其中有两种情况\\(X^TX\\)不可逆：
1. Redundant features (linearly dependent)
2. Too many features (e.g. \\(m \leq n\\)). Delete some features or use regularization.

|       normal equation         |            gradient descent            |
|            ----               |                 ----                   |
| Need to choose \\(\alpha\\)   |    Don't need to choose  \\(\alpha\\)  |
| Needs many iterations         |         Don't need to iterate          |
| works well when n is large    |       Need to compute  \\(X^TX\\)      |
|                               |         Slow if n is very large        |














