#pragma once

#include <execution>
#include <immintrin.h>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <tbb/parallel_for.h>

namespace mat_op
{

/// 输入检查
void InputCheck(const cv::Mat &src1, const cv::Mat &src2);

/// 定义cv::Mat的操作类型
enum class Operators
{
    Add, ///< 逐个元素相加
    Sub, ///< 逐个元素相减
    Mul, ///< 逐个元素相乘
    Div  ///< 逐个元素相除
};

/**
 * 对单通道cv::Mat进行逐元素操作的函数
 *
 * 该函数支持的操作包括加法、减法、乘法和除法
 *
 * @param src1 第一个输入矩阵，必须为单通道浮点类型
 * @param src2 第二个输入矩阵，必须为单通道浮点类型
 * @param Operator 操作类型，可以是加法、减法、乘法或除法
 *
 * @return 结果矩阵，与输入矩阵具有相同的尺寸和类型
 *
 * @throws std::runtime_error 如果任一输入矩阵不是单通道，或者在除法操作中第二个矩阵包含零值
 */
cv::Mat MatOperatorCorrOne(const cv::Mat &src1, const cv::Mat &src2, Operators Operator);

/**
 * @brief 对矩阵的逐元素进行某项操作，操作类型由 Operators定义
 *
 * @tparam Operator 指定的操作类型
 * @param src1      输入的矩阵1
 * @param src2      输入的矩阵2
 * @return cv::Mat  输出的操作后矩阵
 */
template <Operators Operator>
cv::Mat MatOperatorCorr(const cv::Mat &src1, const cv::Mat &src2)
{
    InputCheck(src1, src2);

    std::vector<cv::Mat> src1_channels, src2_channels;
    if (src1.channels() == 1)
    {
        src1_channels.push_back(src1);
        src2_channels.push_back(src2);
    }
    else
    {
        cv::split(src1, src1_channels);
        cv::split(src2, src2_channels);
    }

    std::vector<int> indices(src1_channels.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<cv::Mat> dst_channels(indices.size());

    std::for_each(std::execution::par, indices.begin(), indices.end(),
                  [&](const int &i) { dst_channels[i] = MatOperatorCorrOne(src1_channels[i], src2_channels[i], Operator); });

    cv::Mat dst;
    if (src1.channels() == 1)
        dst = dst_channels[0];
    else
        cv::merge(dst_channels, dst);

    return dst;
}

} // namespace mat_op
