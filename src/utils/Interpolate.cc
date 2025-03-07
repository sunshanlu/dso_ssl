#include "utils/Interpolate.hpp"

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
float BilinInterp(const cv::Mat &data, float u, float v)
{
    assert(data.type() == CV_32F && "input data mast be CV_32F");

    int ix = static_cast<int>(u);
    int iy = static_cast<int>(v);

    if (ix < 0 || iy < 0 || ix >= data.cols - 1 || iy >= data.rows - 1)
        throw std::out_of_range("coordinate out of range");

    float dx = u - ix;
    float dy = v - iy;

    const float &Ip0 = data.at<float>(iy + 1, ix + 1);
    const float &Ip1 = data.at<float>(iy, ix + 1);
    const float &Ip2 = data.at<float>(iy, ix);
    const float &Ip3 = data.at<float>(iy + 1, ix);

    float Ipl = Ip2 * (1 - dy) + Ip3 * dy;
    float Ipr = Ip1 * (1 - dy) + Ip0 * dy;

    return Ipl * (1 - dx) + Ipr * dx;
}

/**
 * @brief 使用SSE指令集进行双线性插值的函数
 *
 * @param data  输入的数据矩阵，必须是32位浮点类型
 * @param u     包含4个插值点的x坐标的数组
 * @param v     包含4个插值点的y坐标的数组
 * @return std::vector<float> 包含4个插值结果的vector
 */
std::vector<float> BilinInterpSSE(const cv::Mat &data, float u[4], float v[4])
{
    assert(data.type() == CV_32F && "input data mast be CV_32F");

    std::vector<float> result(4);
    if (!__builtin_cpu_supports("sse"))
    {
        for (int i = 0; i < 4; ++i)
            result[i] = BilinInterp(data, u[i], v[i]);
        return result;
    }

    alignas(16) float Ip0[4], Ip1[4], Ip2[4], Ip3[4], dx[4], dy[4];
    for (int i = 0; i < 4; ++i)
    {
        int ix = static_cast<int>(u[i]);
        int iy = static_cast<int>(v[i]);

        if (ix < 0 || iy < 0 || ix >= data.cols - 1 || iy >= data.rows - 1)
            throw std::out_of_range("coordinate out of range");

        Ip0[i] = data.at<float>(iy + 1, ix + 1);
        Ip1[i] = data.at<float>(iy, ix + 1);
        Ip2[i] = data.at<float>(iy, ix);
        Ip3[i] = data.at<float>(iy + 1, ix);

        dx[i] = u[i] - ix;
        dy[i] = v[i] - iy;
    }

    __m128 Ip0_ = _mm_load_ps(Ip0);
    __m128 Ip1_ = _mm_load_ps(Ip1);
    __m128 Ip2_ = _mm_load_ps(Ip2);
    __m128 Ip3_ = _mm_load_ps(Ip3);
    __m128 dx_ = _mm_load_ps(dx);
    __m128 dy_ = _mm_load_ps(dy);
    __m128 one = _mm_set1_ps(1.0f);

    __m128 Ipl_ = _mm_add_ps(_mm_mul_ps(Ip2_, _mm_sub_ps(one, dy_)), _mm_mul_ps(Ip3_, dy_));
    __m128 Ipr_ = _mm_add_ps(_mm_mul_ps(Ip1_, _mm_sub_ps(one, dy_)), _mm_mul_ps(Ip0_, dy_));

    __m128 Ip = _mm_add_ps(_mm_mul_ps(Ipl_, _mm_sub_ps(one, dx_)), _mm_mul_ps(Ipr_, dx_));

    alignas(16) float temp[4];
    _mm_store_ps(temp, Ip);
    std::copy(temp, temp + 4, result.begin());
    return result;
}

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
cv::Mat BilinInterpCVMat(const cv::Mat &data, const cv::Mat &u, const cv::Mat &v)
{
    assert(data.type() == CV_32F && "input data mast be CV_32F");
    assert(u.type() == CV_32F && v.type() == CV_32F && "input u and v mast be CV_32F");
    assert(u.rows == v.rows && u.cols == v.cols && "input u and v must have same size");
    cv::Mat result(u.rows, u.cols, CV_32F);

    int allpixels = u.rows * u.cols;

    // 并行化执行函数
    auto block_function = [&](const tbb::blocked_range<int> &range)
    {
        int remains = (range.end() - range.begin()) % 4;
        for (int start = range.begin(); start < range.end() - 3; start += 4)
        {
            float ut[4], vt[4];
            int rowt[4], colt[4];

            for (int idx = start; idx < start + 4; ++idx)
            {
                int row = idx / u.cols;
                int col = idx % u.cols;
                ut[idx - start] = u.at<float>(row, col);
                vt[idx - start] = v.at<float>(row, col);
                rowt[idx - start] = row;
                colt[idx - start] = col;
            }
            auto res = BilinInterpSSE(data, ut, vt);
            for (int i = 0; i < 4; ++i)
                result.at<float>(rowt[i], colt[i]) = res[i];
        }

        for (int i = 1; i <= remains; ++i)
        {
            int idx = range.end() - i;
            int row = idx / u.cols;
            int col = idx % u.cols;

            result.at<float>(row, col) = BilinInterp(data, u.at<float>(row, col), v.at<float>(row, col));
        }
    };

    // 可以保证allpixels里面的元素被4整除以
    tbb::parallel_for(tbb::blocked_range<int>(0, allpixels, 4), block_function);

    return result;
}
} // namespace interp
