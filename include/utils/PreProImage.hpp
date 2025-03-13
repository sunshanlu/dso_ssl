#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace prepro_image
{
/**
 * @brief 使用四合一均值滤波的缩小图像
 *
 * @param input_image 输入的待缩小的图像
 * @return cv::Mat 返回缩小后的图像
 */
cv::Mat MakePyrdOneLayer(const cv::Mat &input_image);

/**
 * @brief 计算图像梯度
 *
 * @param input_image 输入的待计算梯度的图像
 * @param grad_x 返回的x方向梯度
 * @param grad_y 返回的y方向梯度
 */
void MakeGradOneLayer(const cv::Mat &input_image, cv::Mat &grad_x, cv::Mat &grad_y);

/**
 * @brief 对图像进行分块处理，使用中心分块的方案
 *
 * @tparam F 函数模板类型
 * @param input_image       输入的待分块图像
 * @param patch_function    输入的每块图像需要的处理函数
 * @param patch_size        分块的大小
 * @param row_start         开始处理的行数
 * @param col_start         开始处理的列数
 */
template <typename F>
void SegmentImage(const cv::Mat &input_image, F &&patch_function, const int &patch_size, const int &row_start, const int &col_start)
{
    std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>> pro_positons;
    for (int row = row_start; row < input_image.rows - patch_size + 1; row += patch_size)
        for (int col = col_start; col < input_image.cols - patch_size + 1; col += patch_size)
            pro_positons.emplace_back(col, row);

    std::for_each(std::execution::par_unseq, pro_positons.begin(), pro_positons.end(), [&](const Eigen::Vector2i &position) { patch_function(position); });
}
} // namespace prepro_image