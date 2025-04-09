#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>

#include "dso/Frame.hpp"
#include "dso/Keyframe.hpp"

namespace dso_ssl
{

struct TrackIdpethPoint
{
  using SharedPtr = std::shared_ptr<TrackIdpethPoint>;
  using Vec2f = Eigen::Vector2f;

  Vec2f ref_pixel_point_; ///< 参考帧的像素坐标
  float ref_idepth_;      ///< 参考帧的逆深度点
};

class Tracker
{

public:
  using SharedPtr = std::shared_ptr<Tracker>;
  using SE3f = Sophus::SE3f;
  using Mat3f = Eigen::Matrix3f;
  using Vec3f = Eigen::Vector3f;
  using Vec2i = Eigen::Vector2i;
  using Vec2f = Eigen::Vector2f;
  using TrackerIdepthPoints = std::vector<std::vector<TrackIdpethPoint::SharedPtr>>;

  struct Options
  {
    using SharedPtr = std::shared_ptr<Options>;

    Options(const std::string &filepath);

    int pyra_levels_; ///< 使用的图像金字塔层数
  };

  Tracker(Options::SharedPtr options)
      : options_(std::move(options))
      , curr_frame_(nullptr)
      , ref_keyframe_(nullptr)
      , track_idepth_points_(options_->pyra_levels_)
      , track_fx_(options_->pyra_levels_)
      , track_fy_(options_->pyra_levels_)
      , track_cx_(options_->pyra_levels_)
      , track_cy_(options_->pyra_levels_)
      , is_notified_(true)
  {
  }

  /**
   * @brief 当后端优化完成后，更新Tracker跟踪器信息，以匹配最新优化状态，后端线程调用
   *
   * 1. 更新参考关键帧、滑动窗口状态等信息
   * 2. 更新相机的内参矩阵
   *
   * @param sliding_window 输入的滑动窗口，用于向最新关键帧投影
   * @param fx             后端优化后的fx
   * @param fy             后端优化后的fy
   * @param cx             后端优化后的cx
   * @param cy             后端优化后的cy
   */
  void UpdateTracker(const std::vector<KeyFrame::SharedPtr> &sliding_window, const float &fx, const float &fy,
                     const float &cx, const float &cy);

  /**
   * @brief 构建参考点，将构造的参考点放到 track_idepth_points中
   *
   * 1. 将滑动窗口中的关键帧逆深度点，投影到最新的关键帧上，得到第0层逆深度点分布状态
   *  1.1 投影过程需要考虑四舍五入，因此存在多个投影点对应一个投影点的情况，使用高斯归一化积
   *  1.2 逆深度点，向上投影，得到金字塔层级上的逆深度点分布情况
   * 2. Tracker中弃用pattern,使用逆深度点膨胀的方式实现跟踪点的扩充
   *  2.1 0层和1层使用斜侧膨胀方式，对于多个逆深度情况，使用高斯归一化积
   *  2.2 2层及以上使用上下左右膨胀方式，对于多个逆深度情况，使用高斯归一化积
   *
   * @param sliding_window       输入的滑动窗口，用于向最新关键帧投影
   */
  void BuildTrackerPoints(const std::vector<KeyFrame::SharedPtr> &sliding_window);

  /**
   * @brief 更新Tracker跟踪器的内参矩阵
   *
   * @param fx 输入的后端更新后的fx
   * @param fy 输入的后端更新后的fy
   * @param cx 输入的后端更新后的cx
   * @param cy 输入的后端更新后的cy
   */
  void UpdateCalib(const float &fx, const float &fy, const float &cx, const float &cy);

  /**
   * @brief 将滑动窗口中的关键帧向最新关键帧进行投影
   *
   * 1. 要求关键帧的逆深度点状态里面没有外点
   * 2. 使用的逆深度点时、Tcw等滑动窗口状态时，不能在逆深度更新过程中
   * 3. 由于使用四舍五入，因此存在多对一的情况，使用高斯归一化积
   *
   * @param sliding_window          输入的滑动窗口，用于逆深度点投影
   * @param ref_idepth_sum          输出的逆深度点高斯归一化积的分布情况，第0层
   * @param ref_hessian_sum         输出的逆深度点hessian的分布情况，第0层
   */
  void ProjectWindow2Ref(const std::vector<KeyFrame::SharedPtr> &sliding_window, cv::Mat &ref_idepth_sum,
                         cv::Mat &ref_hessian_sum);

  /**
   * @brief 基于当前层的idpeth_sum和hessian_sum信息，向上一层进行投影
   *
   * @param curr_idepth_sum   输入的当前金字塔idepth_sum
   * @param curr_hessian_sum  输入的当前金字塔hessian_sum
   * @param prev_idepth_sum   输出的上一层金字塔idepth_sum
   * @param prev_hessian_sum  输出的上一层金字塔hessian_sum
   */
  void PropagateUp(const cv::Mat &curr_idepth_sum, const cv::Mat &curr_hessian_sum, cv::Mat &prev_idepth_sum,
                   cv::Mat &prev_hessian_sum);

  /**
   * @brief 逆深度点的膨胀，用于在track中取代pattern
   *
   *  1. 0层和1层使用斜侧膨胀方式，对于多个逆深度情况，使用高斯归一化积
   *  2. 2层及以上使用上下左右膨胀方式，对于多个逆深度情况，使用高斯归一化积
   *
   * @param level             输入金字塔的层数
   * @param input_idepth_sum  输入输出的金字塔level层上的逆深度高斯归一化积
   * @param input_hessian_sum 输入输出的金字塔level层上的hessian和
   */
  void IdpethExpansion(const int &level, cv::Mat &input_idepth_sum, cv::Mat &input_hessian_sum);

private:
  Options::SharedPtr options_;              ///< 跟踪器的配置参数
  Frame::SharedPtr curr_frame_;             ///< 当前待跟踪帧
  KeyFrame::SharedPtr ref_keyframe_;        ///< 跟踪器参考关键帧
  TrackerIdepthPoints track_idepth_points_; ///< 参考关键帧的逆深度点，要求初始化时定长度

  std::vector<float> track_fx_; ///< 图像金字塔的fx，可更新
  std::vector<float> track_fy_; ///< 图像金字塔的fy，可更新
  std::vector<float> track_cx_; ///< 图像金字塔的cx，可更新
  std::vector<float> track_cy_; ///< 图像金字塔的cy，可更新

  std::mutex tracker_mutex_;                 ///< 跟踪器互斥量
  std::condition_variable tracker_cond_var_; ///< 跟踪器条件变量
  bool is_notified_;                         ///< 是否执行了唤醒操作，防止虚假唤醒
};

} // namespace dso_ssl
