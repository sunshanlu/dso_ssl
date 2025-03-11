#pragma once

#include <Eigen/Core>

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
Eigen::Matrix<T, 3, 1> ProjectPixel2Norm(const Eigen::Matrix<T, 2, 1> &pixel, const T &fx_inv, const T &fy_inv, const T &cx_inv, const T &cy_inv)
{
    Eigen::Matrix<T, 3, 1> norm;
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
Eigen::Matrix<T, 3, 1> ProjectPoint2Norm(const Eigen::Matrix<T, 3, 1> &point)
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
Eigen::Matrix<T, 2, 1> ProjectNorm2Pixel(const Eigen::Matrix<T, 3, 1> &norm, const T &fx, const T &fy, const T &cx, const T &cy)
{
    Eigen::Matrix<T, 2, 1> pixel;

    pixel[0] = norm[0] * fx + cx;
    pixel[1] = norm[1] * fy + cy;
}

} // namespace project