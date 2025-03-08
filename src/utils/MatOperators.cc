#include "utils/MatOperators.hpp"

namespace mat_op
{

/**
 * 检查两个输入矩阵是否满足特定的类型条件
 *
 * 该函数旨在确保参与后续处理的两个矩阵（src1和src2）都是浮点类型，并且具有相同的类型
 * 这对于保证后续处理步骤的正确性和有效性至关重要
 *
 * @param src1 第一个输入矩阵，其类型将被检查
 * @param src2 第二个输入矩阵，其类型将被检查
 *
 * @throws std::runtime_error 如果任一矩阵不是浮点类型，或者两个矩阵类型不同，则抛出运行时错误
 */
void InputCheck(const cv::Mat &src1, const cv::Mat &src2)
{
    if (!(src1.type() == CV_32F || src1.type() == CV_32FC3))
        throw std::runtime_error("MatOperatorCorr: src1 and src2 must be float type");

    if (src1.type() != src2.type())
        throw std::runtime_error("MatOperatorCorr: src1 and src2 must have the same type");
}

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
cv::Mat MatOperatorCorrOne(const cv::Mat &src1, const cv::Mat &src2, Operators Operator)
{
    InputCheck(src1, src2);

    if (src1.channels() != 1 || src2.channels() != 1)
        throw std::runtime_error("MatDivideCorrOne: src1 and src2 must be single channel");

    int allpixels = src1.rows * src1.cols;
    cv::Mat dst(src1.rows, src1.cols, src1.type());

    auto process_func = [&](const tbb::blocked_range<int> &range)
    {
        float remains = (range.end() - range.begin()) % 4;

        for (int start = range.begin(); start < range.end() - 3; start += 4)
        {
            alignas(16) float src1_data[4], src2_data[4];
            alignas(16) float dst_data[4];
            int rows[4], cols[4];

            for (int idx = start; idx < start + 4; ++idx)
            {
                int row = idx / src1.cols;
                int col = idx % src1.cols;
                rows[idx - start] = row;
                cols[idx - start] = col;

                src1_data[idx - start] = src1.at<float>(row, col);
                src2_data[idx - start] = src2.at<float>(row, col);
                if (Operator == Operators::Div && src2_data[idx - start] == 0)
                    throw std::runtime_error("MatDivideCorrOne: src2 has zero value");
            }

            auto src1_sse = _mm_load_ps(src1_data);
            auto src2_sse = _mm_load_ps(src2_data);

            switch (Operator)
            {
            case Operators::Add:
                _mm_store_ps(dst_data, _mm_add_ps(src1_sse, src2_sse));
                break;

            case Operators::Sub:
                _mm_store_ps(dst_data, _mm_sub_ps(src1_sse, src2_sse));
                break;

            case Operators::Mul:
                _mm_store_ps(dst_data, _mm_mul_ps(src1_sse, src2_sse));
                break;

            case Operators::Div:
                _mm_store_ps(dst_data, _mm_div_ps(src1_sse, src2_sse));
                break;

            default:
                break;
            }

            for (int i = 0; i < 4; ++i)
                dst.at<float>(rows[i], cols[i]) = dst_data[i];
        }

        for (int i = 1; i <= remains; ++i)
        {
            int idx = range.end() - i;
            int row = idx / src1.cols;
            int col = idx % src1.cols;

            dst.at<float>(row, col) = src1.at<float>(row, col) / src2.at<float>(row, col);
        }
    };

    tbb::parallel_for(tbb::blocked_range<int>(0, allpixels, 4), process_func);

    return dst;
}

} // namespace mat_op