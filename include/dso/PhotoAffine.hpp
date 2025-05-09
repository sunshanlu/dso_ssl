#pragma once
#include <Eigen/Core>
#include <cmath>

namespace dso_ssl
{
/// 光度仿射参数
struct PhotoAffine
{
  using Vec2f = Eigen::Vector2f;

  PhotoAffine()
      : a_(0.0)
      , b_(0.0)
  {
  }

  /**
   * 根据aji、bji、ti、tj和ai、bi计算当前光度仿射参数aj和bj
   *
   * @param aji   输入的相对仿射参数aji
   * @param bji   输入的相对仿射参数bji
   * @param ti    输入的i帧曝光时间
   * @param tj    输入的j帧曝光时间
   * @param ai    输入的i帧绝对仿射参数ai
   * @param bi    输入的i帧绝对仿射参数bi
   */
  void SetValFromRelative(const float &aji, const float &bji, const float &ti = 1, const float &tj = 1,
                          const float &ai = 0, const float &bi = 0)
  {
    b_ = bji + std::exp(aji) * bi;
    a_ = std::log(tj * std::exp(ai) / (ti * std::exp(aji)));
  }

  /**
   * 根据ai、aj、bi、bj和两帧的曝光时间计算光度仿射参数aji和bji
   *
   * @param ai  输入的ai
   * @param aj  输入的aj
   * @param bi  输入的bi
   * @param bj  输入的bj
   * @param ti  输入的ti
   * @param tj  输入的tj
   * @return Vec2f 相对光度仿射参数aji和bji
   */
  static Vec2f GetRelative(const float &ai, const float &aj, const float &bi, const float &bj, const float &ti = 1,
                           const float &tj = 1)
  {
    float aji = std::exp(ai) * tj / (std::exp(aj) * ti);
    float bji = bj - aji * bi;
    return Vec2f(aji, bji);
  }

  float a_; ///< 光度仿射参数A
  float b_; ///< 光度仿射参数B
};

} // namespace dso_ssl
