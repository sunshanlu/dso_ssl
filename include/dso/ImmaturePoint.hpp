#pragma once

#include <Eigen/Core>
#include <memory>

namespace dso_ssl
{
class ImmaturePoint
{
public:
  using SharedPtr = std::shared_ptr<ImmaturePoint>;
  using Vec2f = Eigen::Vector2f;

  enum class PointType
  {
    FirstLayer,  ///< 第一层提取到的像素点
    SecondLayer, ///< 第二层提取到的像素点
    ThirdLayer   ///< 第三层提取到的像素点
  };

  ImmaturePoint(const Vec2f &host_pixel_position, const int &layer_type)
      : idepth_min_(0.f)
      , idepth_max_(NAN)
      , host_pixel_position_(host_pixel_position)
  {
    switch (layer_type)
    {
      case 0:
        type_ = PointType::FirstLayer;
        break;
      case 1:
        type_ = PointType::SecondLayer;
        break;
      case 2:
        type_ = PointType::ThirdLayer;
        break;
      default:
        throw std::runtime_error("invalid PointType in ImmaturePoint");
    }
  }

private:
  float idepth_min_, idepth_max_; ///< 逆深度极值范围
  Vec2f host_pixel_position_;     ///< 主帧对应的像素位置
  PointType type_;                ///< 像素点提取的金字塔层级
};
} // namespace dso_ssl
