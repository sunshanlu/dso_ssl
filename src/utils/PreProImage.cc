#include <execution>

#include <Eigen/Core>

#include "utils/ParallelProcess.hpp"
#include "utils/PreProImage.hpp"

namespace prepro_image
{

/**
 * @brief 使用四合一均值滤波的缩小图像
 *
 * @param input_image 输入的待缩小的图像
 * @return cv::Mat 返回缩小后的图像
 */
cv::Mat MakePyrdOneLayer(const cv::Mat &input_image)
{
    if (input_image.cols % 2 != 0 || input_image.rows % 2 != 0)
        throw std::runtime_error("input image size must be even");

    if (input_image.empty())
        throw std::runtime_error("input image is empty");

    if (input_image.type() != CV_32F)
        throw std::runtime_error("input image type must be CV_32F");

    cv::Mat output_image(input_image.rows / 2, input_image.cols / 2, CV_32F, 0.f);

    std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>> pro_simd_positions;

    for (int row = 0; row < input_image.rows - 1; row += 2)
        for (int col = 0; col < input_image.cols - 1; col += 2)
            pro_simd_positions.emplace_back(col, row);

    auto simd_process = [&](const int &start)
    {
        alignas(16) float top_left[4], top_right[4], botton_left[4], bottom_right[4], result[4];
        for (int idx = start; idx < start + 4; ++idx)
        {
            const int &row = pro_simd_positions[idx][1];
            const int &col = pro_simd_positions[idx][0];

            top_left[idx - start] = input_image.at<float>(row, col);
            top_right[idx - start] = input_image.at<float>(row, col + 1);
            botton_left[idx - start] = input_image.at<float>(row + 1, col);
            bottom_right[idx - start] = input_image.at<float>(row + 1, col + 1);
        }

        auto top_left_simd = _mm_load_ps(top_left);
        auto top_right_simd = _mm_load_ps(top_right);
        auto bottom_left_simd = _mm_load_ps(botton_left);
        auto bottom_right_simd = _mm_load_ps(bottom_right);

        auto result_simd = _mm_add_ps(_mm_add_ps(top_left_simd, top_right_simd), _mm_add_ps(bottom_left_simd, bottom_right_simd));
        result_simd = _mm_mul_ps(_mm_set1_ps(0.25), result_simd);
        _mm_store_ps(result, result_simd);

        for (int idx = 0; idx < 4; ++idx)
        {
            const int &row = pro_simd_positions[idx + start][1];
            const int &col = pro_simd_positions[idx + start][0];

            output_image.at<float>(row / 2, col / 2) = result[idx];
        }
    };

    auto single_process = [&](const int &idx)
    {
        const int &row = pro_simd_positions[idx][1];
        const int &col = pro_simd_positions[idx][0];

        const float &top_left = input_image.at<float>(row, col);
        const float &top_right = input_image.at<float>(row, col + 1);
        const float &bottom_left = input_image.at<float>(row + 1, col);
        const float &bottom_right = input_image.at<float>(row + 1, col + 1);

        output_image.at<float>(row / 2, col / 2) = (top_right + top_left + bottom_right + bottom_left) * 0.25;
    };

    parallel::ParallelWrapper(0, pro_simd_positions.size(), 4, simd_process, single_process);

    return output_image;
}

/**
 * @brief 计算图像梯度
 *
 * @param input_image 输入的待计算梯度的图像
 * @param grad_x 返回的x方向梯度
 * @param grad_y 返回的y方向梯度
 *
 * @note 处理了1px的像素边缘问题
 */
void MakeGradOneLayer(const cv::Mat &input_image, cv::Mat &grad_x, cv::Mat &grad_y)
{
    if (input_image.empty())
        throw std::runtime_error("input image is empty");

    if (input_image.type() != CV_32F)
        throw std::runtime_error("input image type must be CV_32F");

    grad_x = cv::Mat(input_image.rows, input_image.cols, CV_32F, 0.f);
    grad_y = cv::Mat(input_image.rows, input_image.cols, CV_32F, 0.f);

    std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>> pro_positions;
    for (int row = 1; row < input_image.rows - 1; ++row)
        for (int col = 1; col < input_image.cols - 1; ++col)
            pro_positions.emplace_back(col, row);

    auto half_simd = _mm_set1_ps(0.5);
    auto simd_process = [&](const int &start)
    {
        alignas(16) float left[4], right[4], top[4], bottom[4];
        alignas(16) float grad_x_data[4], grad_y_data[4];
        for (int idx = start; idx < start + 4; ++idx)
        {
            left[idx - start] = input_image.at<float>(pro_positions[idx][1], pro_positions[idx][0] - 1);
            right[idx - start] = input_image.at<float>(pro_positions[idx][1], pro_positions[idx][0] + 1);
            top[idx - start] = input_image.at<float>(pro_positions[idx][1] - 1, pro_positions[idx][0]);
            bottom[idx - start] = input_image.at<float>(pro_positions[idx][1] + 1, pro_positions[idx][0]);
        }

        auto left_simd = _mm_load_ps(left);
        auto right_simd = _mm_load_ps(right);
        auto top_simd = _mm_load_ps(top);
        auto bottom_simd = _mm_load_ps(bottom);

        auto grad_x_simd = _mm_mul_ps(_mm_sub_ps(right_simd, left_simd), half_simd);
        auto grad_y_simd = _mm_mul_ps(_mm_sub_ps(bottom_simd, top_simd), half_simd);

        _mm_store_ps(grad_x_data, grad_x_simd);
        _mm_store_ps(grad_y_data, grad_y_simd);

        for (int idx = 0; idx < 4; ++idx){
            grad_x.at<float>(pro_positions[idx + start][1], pro_positions[idx + start][0]) = grad_x_data[idx];
            grad_y.at<float>(pro_positions[idx + start][1], pro_positions[idx + start][0]) = grad_y_data[idx];
        }
    };

    auto single_process = [&](const int &idx)
    {
        const auto &position = pro_positions[idx];

        const float &left = input_image.at<float>(position[1], position[0] - 1);
        const float &right = input_image.at<float>(position[1], position[0] + 1);
        const float &top = input_image.at<float>(position[1] - 1, position[0]);
        const float &bottom = input_image.at<float>(position[1] + 1, position[0]);

        grad_x.at<float>(position[1], position[0]) = (right - left) * 0.5;
        grad_y.at<float>(position[1], position[0]) = (bottom - top) * 0.5;
    };

    parallel::ParallelWrapper(0, pro_positions.size(), 4, simd_process, single_process);

    // 处理边缘1px的位置问题
    grad_x.row(1).copyTo(grad_x.row(0));
    grad_y.row(1).copyTo(grad_y.row(0));

    grad_x.row(grad_x.rows - 2).copyTo(grad_x.row(grad_x.rows - 1));
    grad_y.row(grad_y.rows - 2).copyTo(grad_y.row(grad_y.rows - 1));

    grad_x.col(1).copyTo(grad_x.col(0));
    grad_y.col(1).copyTo(grad_y.col(0));

    grad_x.col(grad_x.cols - 2).copyTo(grad_x.col(grad_x.cols - 1));
    grad_y.col(grad_y.cols - 2).copyTo(grad_y.col(grad_y.cols - 1));
}

} // namespace prepro_image