---
authors: 孙善路-github, 孙善路-bilibili
title: DSO中的去畸变
tags: vSLAM, DSO, Robot
date: 2025-1-09
slug: De-distortion-in-DSO
Category: DSO
description: 光度去畸变和像素去畸变
---

[TOC]


```c++
void CoarseInitializer::makeGradients(Eigen::Vector3f **data) {
    for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) {
        int lvlm1 = lvl - 1;
        int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

        Eigen::Vector3f *dINew_l = data[lvl];
        Eigen::Vector3f *dINew_lm = data[lvlm1];
        // 使用上一层得到当前层的值
        for (int y = 0; y < hl; y++)
            for (int x = 0; x < wl; x++)
                dINew_l[x + y * wl][0] = 0.25f * (dINew_lm[2 * x + 2 * y * wlm1][0] + dINew_lm[2 * x + 1 + 2 * y * wlm1][0] +
                                                  dINew_lm[2 * x + 2 * y * wlm1 + wlm1][0] + dINew_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
        // 根据像素计算梯度
        for (int idx = wl; idx < wl * (hl - 1); idx++) {
            dINew_l[idx][1] = 0.5f * (dINew_l[idx + 1][0] - dINew_l[idx - 1][0]);
            dINew_l[idx][2] = 0.5f * (dINew_l[idx + wl][0] - dINew_l[idx - wl][0]);
        }
    }
}
```

