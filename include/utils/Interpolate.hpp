#pragma once

#include <algorithm>
#include <execution>
#include <numeric>
#include <xmmintrin.h>

#include <opencv2/opencv.hpp>

namespace interp
{

/**
 * @brief 对图像数据执行双线性插值。
 *
 * 此函数对给定的一组图像数据集执行双线性插值，返回指定位置的插值像素值。
 * 双线性插值是一种通过两次线性插值来估计四个已知数据点之间某个点的值的技术。
 * 这是图像处理中常用的一种估计非整数坐标处像素值的方法。
 *
 * @param data   包含单通道 32 位浮点图像数据的 cv::Mat 对象。
 * @param u      用于插值的 x 坐标，可以是浮点数。
 * @param v      用于插值的 y 坐标，可以是浮点数。
 * @return float 插值的像素值。
 *
 * @note 该函数假定输入的图像数据为CV_32F类型，且坐标（u，v）在图像的有效范围内。如果不满足条件，函数将抛出异常或断言失败。
 */
float BilinInterp(const cv::Mat &data, float u, float v);

/**
 * @brief 使用SSE指令集进行双线性插值的函数
 *
 * @param data  输入的数据矩阵，必须是32位浮点类型
 * @param u     包含4个插值点的x坐标的数组
 * @param v     包含4个插值点的y坐标的数组
 * @return std::vector<float> 包含4个插值结果的vector
 */
std::vector<float> BilinInterpSSE(const cv::Mat &data, float u[4], float v[4]);

/**
 * @brief 使用双线性插值处理图像数据
 *
 * @param data 输入的图像数据，必须是单通道32位浮点数格式
 * @param u 插值的横向坐标映射，必须是32位浮点数格式，存储source的u坐标
 * @param v 插值的纵向坐标映射，必须是32位浮点数格式，存储source的v坐标
 * @return 返回插值后的图像数据
 *
 * 该函数通过双线性插值方法，根据输入的u和v坐标对图像数据进行插值处理
 * 主要利用了Intel TBB库和SSE指令集来并行处理和优化计算性能
 */
cv::Mat BilinInterpCVMat(const cv::Mat &data, const cv::Mat &u, const cv::Mat &v);

} // namespace interp
