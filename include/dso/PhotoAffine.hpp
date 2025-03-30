#pragma once

namespace dso_ssl
{
/// 光度仿射参数
struct PhotoAffine
{

  PhotoAffine()
      : a_(0.0)
      , b_(0.0)
  {
  }

  float a_; ///< 光度仿射参数A
  float b_; ///< 光度仿射参数B
};

} // namespace dso_ssl
