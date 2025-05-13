#pragma once

#include <memory>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include "dso/PhotoAffine.hpp"
#include "dso/Undistorter.hpp"

namespace dso_ssl
{

/// 帧核心
struct FrameKernel
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using SharedPtr = std::shared_ptr<FrameKernel>;

  FrameKernel(double timestamp, float exposure_time)
      : timestamp_(std::move(timestamp))
      , exposure_time_(std::move(exposure_time))
  {
    id_ = next_id_++;
  }

  static std::size_t next_id_;

  std::size_t id_;
  Sophus::SE3f T_cw_;     ///< 当前帧的位姿
  PhotoAffine affine_ab_; ///< 当前帧的光度仿射参数
  double timestamp_;      ///< 帧的时间戳
  float exposure_time_;   ///< 帧的曝光时间
};

/// 定义DSO中的普通帧
class Frame
{
public:
  using SharedPtr = std::shared_ptr<Frame>;

  friend void ShowPyraidImagesAndGrads(Frame::SharedPtr frameptr); // 展示图像金字塔，供测试使用

  struct Options
  {
    using SharedPtr = std::shared_ptr<Options>;

    Options(const std::string &file_path)
    {
      auto info = YAML::LoadFile(file_path);
      pyra_levels_ = info["PyraidLevelsUsed"].as<int>();
      use_origin_grad_ = info["UseOriginGrad"].as<bool>();
    }

    int pyra_levels_;      ///< 图像金字塔层数
    bool use_origin_grad_; ///< 梯度平方和是否使用原图的梯度
  };

  Frame(const Options::SharedPtr &config, const cv::Mat &first_layer_image, const cv::Mat &only_pixel_undistorted_image, double timestamp,
        float exposure_time);

  Frame(Frame &&other) noexcept
  {
    options_ = std::move(other.options_);
    frame_kernel_ = std::move(other.frame_kernel_);
    pyrd_image_and_grads_ = std::move(other.pyrd_image_and_grads_);
    image_and_grad_ = std::move(other.image_and_grad_);
    squre_grad_ = std::move(other.squre_grad_);

    other.options_ = nullptr;
    other.frame_kernel_ = nullptr;
    other.pyrd_image_and_grads_.clear();
    other.image_and_grad_ = cv::Mat();
    other.squre_grad_ = cv::Mat();
  }

  /**
   * @brief 设置帧的状态，包括位姿和仿射参数
   *
   * @note 需要设置成绝对量而不是相对量，因此这里有aji到aj和bji到bj的转换 @see PhotoAffine::SetValFromRelative
   *
   * @param Tcw 输入的估计的位姿Tcw
   * @param acw 输入的估计的乘法仿射参数acw
   * @param bcw 输入的估计的加法仿射参数bcw
   * @param ti  输入的i帧曝光时间
   * @param ai  输入的i帧绝对仿射参数ai
   * @param bi  输入的i帧绝对仿射参数bi
   */
  void SetEstimate(const Sophus::SE3f &Tcw, const float &acw, const float &bcw, const float &ti = 1, const float &ai = 0, const float &bi = 0)
  {
    std::lock_guard<std::mutex> lock(status_mutex_);
    frame_kernel_->T_cw_ = Tcw;
    // frame_kernel_->affine_ab_.a_ = acw;
    // frame_kernel_->affine_ab_.b_ = bcw;

    float tj = GetExposureTime();
    if (tj < 0)
      tj = 1;

    frame_kernel_->affine_ab_.SetValFromRelative(acw, bcw, ti, tj, ai, bi);
  }

  virtual ~Frame() = default;

  const std::vector<cv::Mat> &GetPyrdImageAndGrads() const { return pyrd_image_and_grads_; }

  const cv::Mat &GetSqureGrad() const { return squre_grad_; }

  const float &GetExposureTime() const { return frame_kernel_->exposure_time_; }

  const double &GetTimestamp() const { return frame_kernel_->timestamp_; }

  const std::size_t &GetIdx() const { return frame_kernel_->id_; }

  Sophus::SE3f GetTcw()
  {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return frame_kernel_->T_cw_;
  }

  /**
   * 获取帧状态，加状态锁了
   *
   * @param Tcw 输出的Tcw
   * @param acw 输出的acw
   * @param bcw 输出的bcw
   */
  void GetFrameStatus(Sophus::SE3f &Tcw, float &acw, float &bcw)
  {
    std::lock_guard<std::mutex> lock(status_mutex_);
    Tcw = frame_kernel_->T_cw_;
    acw = frame_kernel_->affine_ab_.a_;
    bcw = frame_kernel_->affine_ab_.b_;
  }


protected:
  /// 构造图像金字塔 --> 4合1 + 均值滤波
  void MakePyrdImages(const cv::Mat &first_layer_image);

  /// 计算第0层梯度平方和，用于后续第0层的点选操作
  void MakeSqureGrad(const cv::Mat &only_pixel_undistorted_image);

  Options::SharedPtr options_;                ///< Frame的配置信息
  FrameKernel::SharedPtr frame_kernel_;       ///< 保存的帧核心参数信息
  std::vector<cv::Mat> pyrd_image_and_grads_; ///< 维护的图像金字塔上的图像和梯度信息
  cv::Mat image_and_grad_;                    ///< 金字塔第0层图像和梯度信息
  cv::Mat squre_grad_;                        ///< 梯度平方和
  std::mutex status_mutex_;                   ///< 状态互斥量
};

} // namespace dso_ssl
