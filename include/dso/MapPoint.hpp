#pragma once
#include <memory>

#include <Eigen/Core>

namespace dso_ssl
{
class MapPoint
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using SharedPtr = std::shared_ptr<MapPoint>;
  using Vec2f = Eigen::Vector2f;

  MapPoint(const float &idepth, const Vec2f &host_pixel_position_, const float &hessian)
      : idepth_(idepth)
      , hessian_(hessian)
      , host_pixel_position_(host_pixel_position_)
  {
  }

  const Vec2f &GetHostPixel() const { return host_pixel_position_; }

  const float &GetIdepth() const { return idepth_; }

  const float &GetHessian() const { return hessian_; }

private:
  float idepth_;              ///< 逆深度
  float hessian_;             ///< 信息矩阵
  Vec2f host_pixel_position_; ///< 主帧像素坐标
};
} // namespace dso_ssl
