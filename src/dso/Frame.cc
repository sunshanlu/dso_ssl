#include "utils/Interpolate.hpp"
#include "utils/ParallelProcess.hpp"
#include "utils/PreProImage.hpp"

#include "dso/Frame.hpp"

namespace dso_ssl
{

/**
 * @brief 构建图像金字塔，包括图像和梯度构建
 *
 * @param first_layer_image 输入的原图像
 *
 * @see prepro_image::MakeGradOneLayer() --> 构建图像梯度
 * @see prepro_image::MakePyrdOneLayer() --> 构建缩放2倍的图像
 */
void Frame::MakePyrdImages(const cv::Mat &first_layer_image)
{
  if (first_layer_image.empty())
    throw std::runtime_error("The input image is empty.");

  int downsample_scale = std::pow(2, (options_->pyra_levels_ - 1));
  if (first_layer_image.rows % downsample_scale != 0 || first_layer_image.cols % downsample_scale != 0)
    throw std::runtime_error("The input image size is not divisible by the downsample scale.");

  pyrd_image_and_grads_.clear();
  pyrd_image_and_grads_.resize(options_->pyra_levels_);

  cv::Mat layer_image = first_layer_image;
  cv::Mat first_layer_gradx, first_layer_grady;

  prepro_image::MakeGradOneLayer(layer_image, first_layer_gradx, first_layer_grady);
  cv::merge(std::vector({first_layer_image, first_layer_gradx, first_layer_grady}), pyrd_image_and_grads_[0]);

  for (int level = 1; level < options_->pyra_levels_; ++level)
  {
    layer_image = prepro_image::MakePyrdOneLayer(layer_image);

    cv::Mat layer_gradx, layer_grady;
    prepro_image::MakeGradOneLayer(layer_image, layer_gradx, layer_grady);

    cv::merge(std::vector({layer_image, layer_gradx, layer_grady}), pyrd_image_and_grads_[level]);
  }

  image_and_grad_ = pyrd_image_and_grads_[0];
}

/**
 * @brief Construct a new Frame:: Frame object
 *
 * @param config                        输入的Frame配置信息
 * @param first_layer_image             输入的图像金字塔第0层图像
 * @param only_pixel_undistorted_image  需要为点选做准备
 * @param timestamp                     获取图像数据的时间戳
 * @param exposure_time                 曝光时间
 */
Frame::Frame(const Options::SharedPtr &config, const cv::Mat &first_layer_image,
             const cv::Mat &only_pixel_undistorted_image, double timestamp, float exposure_time)
    : options_(config)
{
  frame_kernel_ = std::make_shared<FrameKernel>(timestamp, exposure_time);
  MakePyrdImages(first_layer_image);
  MakeSqureGrad(only_pixel_undistorted_image);
}

/**
 * @brief 计算第0层图像梯度平方和，方便后续的像素点选择操作
 *
 * 当use_origin_grad为true时，要求输入的only_pixel_undistorted_image不能为空
 * 当use_origin_grad为false时，输入的only_pixel_undistorted_image可以为空
 *
 * @param only_pixel_undistorted_image   仅像素去畸变图像，当点选部分要求使用原像素梯度时使用
 */
void Frame::MakeSqureGrad(const cv::Mat &only_pixel_undistorted_image)
{
  int allpixels = image_and_grad_.rows * image_and_grad_.cols;
  squre_grad_ = cv::Mat::zeros(image_and_grad_.rows, image_and_grad_.cols, CV_32F);

  cv::Mat grad_x, grad_y;

  if (options_->use_origin_grad_)
  {
    if (only_pixel_undistorted_image.empty())
      throw std::runtime_error("only_pixel_undistorted_image is empty");

    prepro_image::MakeGradOneLayer(only_pixel_undistorted_image, grad_x, grad_y);
  }
  else
  {
    std::vector<cv::Mat> image_and_grad;
    cv::split(image_and_grad_, image_and_grad);
    grad_x = image_and_grad[1];
    grad_y = image_and_grad[2];
  }

  auto process_simd = [&](const int &start)
  {
    alignas(16) float grad_x_data[4], grad_y_data[4], result[4];
    float rows[4], cols[4];

    for (int idx = start; idx < start + 4; ++idx)
    {
      int row = idx / image_and_grad_.cols;
      int col = idx % image_and_grad_.cols;

      grad_x_data[idx - start] = grad_x.at<float>(row, col);
      grad_y_data[idx - start] = grad_y.at<float>(row, col);

      rows[idx - start] = row;
      cols[idx - start] = col;
    }

    auto grad_x_simd = _mm_load_ps(grad_x_data);
    auto grad_y_simd = _mm_load_ps(grad_y_data);

    auto result_simd = _mm_add_ps(_mm_mul_ps(grad_x_simd, grad_x_simd), _mm_mul_ps(grad_y_simd, grad_y_simd));
    _mm_store_ps(result, result_simd);

    for (int idx = 0; idx < 4; ++idx)
      squre_grad_.at<float>(rows[idx], cols[idx]) = result[idx];
  };

  auto process_single = [&](const int idx)
  {
    int row = idx / image_and_grad_.cols;
    int col = idx % image_and_grad_.cols;

    const auto &gradx_value = grad_x.at<float>(row, col);
    const auto &grady_value = grad_y.at<float>(row, col);
    squre_grad_.at<float>(row, col) = gradx_value * gradx_value + grady_value * grady_value;
  };

  parallel::ParallelWrapper(0, allpixels, 4, process_simd, process_single);
}

// 静态成员变量的初始化位置
std::size_t FrameKernel::next_id_ = 0;

} // namespace dso_ssl
