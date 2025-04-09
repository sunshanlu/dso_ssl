#include "dso/Keyframe.hpp"
#include "dso/Initializer2.hpp"

namespace dso_ssl
{

/**
 * 初始化过程中，将普通帧转换为关键帧，要求frame的状态的正确的 frame_kernel
 *
 * condition1: 初始化器参考帧，除了初始化关键帧的状态信息外，还可以初始化关键帧的逆深度点
 * condition2: 初始化器最后一个跟踪参考帧，仅初始化关键帧的状态信息
 *
 * @param frame       输入的普通帧，使用移动构造，frame没有意义了
 * @param init_points 输入的初始化器的优化点，是经过初始化器筛选的点
 */
KeyFrame::KeyFrame(const Frame::SharedPtr &frame, const std::vector<InitPointPtr> &init_points)
    : Frame(std::move(*frame))
{
  // 这里工作量不大，不需要使用多线程
  for (const auto &init_point: init_points)
  {
    if (!init_point->is_good_)
      throw std::invalid_argument("init_points have not good point");

    auto mp = std::make_shared<MapPoint>(init_point->idepth_, init_point->host_pixel_positon_, init_point->hessian_);
    map_points_.push_back(mp);
  }
}

} // namespace dso_ssl
