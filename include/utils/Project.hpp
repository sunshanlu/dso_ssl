#pragma once
#include <immintrin.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace project
{
/**
 * @brief 投影像素点到归一化坐标系
 *
 * @tparam T        模板类型，float or double
 * @param pixel     待投影像素点
 * @param fx_inv    fx的倒数
 * @param fy_inv    fy的倒数
 * @param cx_inv    cx / fx
 * @param cy_inv    cy / fy
 * @return Eigen::Matrix<T, 3, 1> 输出的归一化坐标系下的点
 *
 * @note 注意cx_inv和cy_inv的具体含义，并不是cx和cy的逆
 */
template <typename T>
Eigen::Vector3<T> Pixel2Norm(const Eigen::Vector2<T> &pixel, const T &fx_inv, const T &fy_inv, const T &cx_inv, const T &cy_inv)
{
  Eigen::Vector3<T> norm;
  norm << 1, 1, 1;

  norm[0] = pixel[0] * fx_inv - cx_inv;
  norm[1] = pixel[1] * fy_inv - cy_inv;

  return norm;
}

/**
 * @brief 投影点到归一化坐标系点
 *
 * @tparam T        输入的模板类型
 * @param point     待投影点
 * @return Eigen::Matrix<T, 3, 1> 输出的归一化坐标系下的点
 */
template <typename T>
Eigen::Vector3<T> Point2Norm(const Eigen::Matrix<T, 3, 1> &point)
{
  return point / point[2];
}

/**
 * @brief 投影归一化坐标系点到像素点
 *
 * @tparam T    模板类型，float or double
 * @param norm  输入的归一化坐标系下的点
 * @param fx
 * @param fy
 * @param cx
 * @param cy
 * @return Eigen::Matrix<T, 2, 1> 输出的像素坐标系下的点
 */
template <typename T>
Eigen::Vector2<T> Norm2Pixel(const Eigen::Matrix<T, 3, 1> &norm, const T &fx, const T &fy, const T &cx, const T &cy)
{
  Eigen::Vector2<T> pixel;

  pixel[0] = norm[0] * fx + cx;
  pixel[1] = norm[1] * fy + cy;

  return pixel;
}

/**
 * @brief 将pi投影到pj上，并且拿到中间投影过程的点
 *
 * @param Tji       输入的ji之间的相对位姿
 * @param pixel_in  输入的pi的像素坐标系
 * @param dpi       输入的pi的逆深度
 * @param fx        输入的fx
 * @param fy        输入的fy
 * @param cx        输入的cx
 * @param cy        输入的cy
 * @param p         输出的中间变量
 * @param pixel_out 输出的pj的像素坐标系
 */
template <typename T>
void Pixel2Pixel(const Sophus::SE3<T> &Tji, const Eigen::Vector2<T> &pixel_in, const T &dpi, const T &fx, const T &fy, const T &cx, const T &cy,
                 Eigen::Vector3<T> &p, Eigen::Vector2<T> &pixel_out)
{
  if (fx == 0 || fy == 0)
    throw std::runtime_error("fx == 0 || fy == 0");

  T fx_inv = 1.0 / fx;
  T fy_inv = 1.0 / fy;
  T cx_inv = cx * fx_inv;
  T cy_inv = cy * fy_inv;

  Eigen::Vector3<T> pi_norm = Pixel2Norm(pixel_in, fx_inv, fy_inv, cx_inv, cy_inv);
  p = Tji.so3() * pi_norm + Tji.translation() * dpi;

  pixel_out = Norm2Pixel(Point2Norm(p), fx, fy, cx, cy);
}

/**
 * @brief 给定归一化坐标系和逆深度，求解点Pi到临时变量点Pj'
 *
 * @tparam T        模板类型
 * @param Tji       输入的ji之间的相对位姿
 * @param pnorm_i   归一化坐标系下的点Pi
 * @param dpi       输入的pi的逆深度
 * @return Eigen::Vector3<T>  输出的临时变量点Pj'
 */
template <typename T>
Eigen::Vector3<T> Point2PointTemp(const Sophus::SE3<T> &Tji, const Eigen::Vector3<T> &pnorm_i, const T &dpi)
{
  return Tji.so3() * pnorm_i + Tji.translation() * dpi;
}

using Vector3fArray = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;
using Vector3dArray = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
using Vector2fArray = std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>;
using Vector2dArray = std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;

// 输入输出为真实的Eigen矩阵
Vector3fArray Pixel2NormSSE(const Vector3fArray &pixels, const float &fx_inv, const float &fy_inv, const float &cx_inv, const float &cy_inv);
Vector2fArray Norm2PixelSSE(const Vector3fArray &norms, const float &fx, const float &fy, const float &cx, const float &cy);
Vector3fArray Point2NormSSE(const Vector3fArray &points);
Vector3fArray Point2PointTempSSE(const Vector3fArray &pnorm_i, const Sophus::SE3f &Tji, const float &dpi);

// 输入输出为SSE表示，用来耦合
void Pixel2NormSSEInternal(const __m128 &u_sse, const __m128 &v_sse, const __m128 &fx_inv_sse, const __m128 &fy_inv_sse, const __m128 &cx_inv_sse,
                           const __m128 &cy_inv_sse, __m128 &x_norm_sse, __m128 &y_norm_sse);
void Norm2PixelSSEInternal(const __m128 &x_norm_sse, const __m128 &y_norm_sse, const __m128 &fx_sse, const __m128 &fy_sse, const __m128 &cx_sse,
                           const __m128 &cy_sse, __m128 &result_u_sse, __m128 &result_v_sse);
void Point2NormSSEInternal(const __m128 &x_data_sse, const __m128 &y_data_sse, const __m128 &z_data_sse, __m128 &x_norm_sse, __m128 &y_norm_sse);
void Point2PointTempSSEInternal(const Sophus::SE3f &Tji, const __m128 &x_norm_i_sse, const __m128 &y_norm_i_sse, const float &dpi, __m128 &temp_data_x,
                                __m128 &temp_data_y, __m128 &temp_data_z);

// clang-format off
void Pixel2PixelSSE(const Sophus::SE3f &Tji, const Vector2fArray &pixel_in, const float &dpi, const float &fx, 
                    const float &fy, const float &cx, const float &cy, Vector3fArray &p, Vector2fArray &pixel_out);
// clang-format on

} // namespace project