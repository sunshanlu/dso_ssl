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
} // namespace prepro_image