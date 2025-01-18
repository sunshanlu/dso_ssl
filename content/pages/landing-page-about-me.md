---
author: "孙善路"
title: SSL的SLAM系列 - DSO
layout: page
date: 2025-01-07 21:04:47 +0800
status: hidden
slug: landing-page-about-hidden
---

`DSO` 是 `vSLAM` 的一个重要的开源实现，它基于直接法，考虑了相机成像过程，使用光度标定或者仿射参数进行相机测量的直接矫正，使用相对高效的帧管理和点管理策略来维护整个滑动窗口。在优化阶段，使用自定义的优化器，充分使用了intel芯片的`SIMD`指令集来对整个优化过程进行加速。除此之外，它还制定了不同的策略，来防止解在零空间上的漂移，例如系统的正交投影、解的正交投影、SVD分解置零和固定初始帧的策略。除此之外，由于它的整个系统涉及到滑动窗口的点边缘化和帧边缘化，它使用部分`FEJ`的方式来防止这种边缘化的误操作行为导致的部分不客观的维度变得可观情况的发生。


## DSO的代码解读

<img src="../theme/images/apple-touch-icon-180x180.png" alt="RobotLU" style="float: right; margin-left: 10px; width: 150; height: 150;">

我使用了一个多月的时间，完成了`DSO`的源码解读，我想使用记录博客的方式对整个`DSO`的源码按照我的理解进行梳理。在这个博客随笔中，你可以从我的角度了解到 Direct和 Indirect的区别，Sparse 和 Dense的区别、`DSO`中的相机标定内容（畸变模型到pinhole模型的转换、光度去畸变）、`DSO`中的点选策略（像素点选择）、`DSO`考虑平移距离的初始化策略、`DSO`的跟踪策略、`DSO`的滑窗优化（涉及`FEJ`，local转global以及HM和bM的维护策略）。

## DSO的源码重写

我计划对整个`DSO`进行重写，重写的`DSO`源码应该比较符合我自己的代码风格，感兴趣的朋友可以关注我的DSO重写仓库：[DSO_SSL](https://github.com/sunshanlu/dso_ssl)。

## 相关链接

- [我的DSO重写仓库：dso_ssl](https://github.com/sunshanlu/dso_ssl)
- [我的bilibili主页：rookie-lu](https://space.bilibili.com/489032586)
- [我的github主页：rookie-lu](https://github.com/sunshanlu)