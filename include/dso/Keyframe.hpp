#pragma once

#include <memory>

#include "dso/Frame.hpp"
#include "dso/ImmaturePoint.hpp"
#include "dso/MapPoint.hpp"

namespace dso_ssl
{

struct InitIdepthPoint;

class KeyFrame : public Frame
{
public:
  using SharedPtr = std::shared_ptr<KeyFrame>;
  using InitPointPtr = std::shared_ptr<InitIdepthPoint>;

  /**
   * 初始化过程中，将普通帧转换为关键帧
   *
   * condition1: 初始化器参考帧，除了初始化关键帧的状态信息外，还可以初始化关键帧的逆深度点
   * condition2: 初始化器最后一个跟踪参考帧，仅初始化关键帧的状态信息
   *
   * @param frame       输入的普通帧，使用移动构造，frame没有意义了
   * @param init_points 输入的初始化器的优化点
   */
  KeyFrame(const Frame::SharedPtr &frame, const std::vector<InitPointPtr> &init_points);

  /// 设置关键帧的未成熟点
  void SetImmaturePoints(std::vector<ImmaturePoint::SharedPtr> immature_points)
  {
    immature_points_ = std::move(immature_points);
  }

  /// 获取关键帧的逆深度地图点
  const std::vector<MapPoint::SharedPtr> &GetMapPoints() const { return map_points_; }

private:
  std::vector<MapPoint::SharedPtr> map_points_;           ///< 关键帧的逆深度点
  std::vector<ImmaturePoint::SharedPtr> immature_points_; ///< 关键帧的未成熟点
  std::vector<MapPoint::SharedPtr> outlier_points_;       ///< 关键帧的逆深度点被判断为外点
  std::vector<MapPoint::SharedPtr> marginal_points_;      ///< 关键帧的逆深度点被判断为边缘化点
};

} // namespace dso_ssl
