#include <cmath>

#include "utils/PixelUndistorted.hpp"

namespace undistort
{

/**
 * @brief FOV 畸变模型的公式描述
 *
 * @param distorted_params  输入的畸变参数
 * @param undistorted_point 输入的归一化坐标系未畸变坐标点
 * @return Eigen::Vector2f  输出的畸变归一化坐标系坐标点
 */
Eigen::Vector2f FOV(const FOVParams &distorted_params, const Eigen::Vector2f &undistorted_point)
{
    assert(distorted_params.omega != 0.0f);
    Eigen::Vector2f distorted_point;

    const float &x = undistorted_point[0];
    const float &y = undistorted_point[1];

    float r = sqrt(x * x + y * y);
    float rd = 1.0f / distorted_params.omega * std::atan(2.0f * r * std::tan(distorted_params.omega / 2.0f));

    float scale = rd / (r + 1e-5);

    distorted_point[0] = x * scale;
    distorted_point[1] = y * scale;
    return distorted_point;
}

/**
 * @brief KannalaBrandt 畸变模型的公式描述
 *
 * @param distorted_params  输入的畸变参数
 * @param undistorted_point 输入的归一化坐标系未畸变坐标点
 * @return Eigen::Vector2f  输出的畸变归一化坐标系坐标点
 */
Eigen::Vector2f KB(const KBParams &distorted_params, const Eigen::Vector2f &undistorted_point)
{
    const float &x = undistorted_point[0];
    const float &y = undistorted_point[1];

    float theta = std::atan(std::sqrt(x * x + y * y));
    float theta2 = theta * theta;
    float theta4 = theta2 * theta2;
    float theta6 = theta4 * theta2;

    float theta_d = theta * (1 + distorted_params.k1 * theta2 + distorted_params.k2 * theta4 + distorted_params.k3 * theta6);
    float scale = theta_d / (theta + 1e-5);

    return Eigen::Vector2f(x * scale, y * scale);
}

/**
 * @brief RadTan 3参数畸变模型的公式描述
 *
 * @tparam
 * @param distorted_params  畸变参数（3）
 * @param undistorted_point 输入的无畸变归一化坐标系坐标点
 * @return Eigen::Vector2f  输出的畸变归一化坐标系坐标点
 */
template <>
Eigen::Vector2f RadTan<3>(const RadTanParams<3> &distorted_params, const Eigen::Vector2f &undistorted_point)
{
    Eigen::Vector2f distorted_point;

    const float &x = undistorted_point[0];
    const float &y = undistorted_point[1];
    float r2 = x * x + y * y;

    float scale = (1 + distorted_params.k1 * r2);
    float xy = x * y;

    distorted_point[0] = x * scale + 2 * distorted_params.p1 * xy + distorted_params.p2 * (r2 + 2 * x * x);
    distorted_point[1] = y * scale + 2 * distorted_params.p2 * xy + distorted_params.p1 * (r2 + 2 * y * y);
    return distorted_point;
}

/**
 * @brief RadTan 5参数畸变模型的公式描述
 *
 * @tparam
 * @param distorted_params  畸变参数（5）
 * @param undistorted_point 输入的无畸变归一化坐标系坐标点
 * @return Eigen::Vector2f  输出的畸变归一化坐标系坐标点
 */
template <>
Eigen::Vector2f RadTan<5>(const RadTanParams<5> &distorted_params, const Eigen::Vector2f &undistorted_point)
{
    Eigen::Vector2f distorted_point;

    const float &x = undistorted_point[0];
    const float &y = undistorted_point[1];
    float r2 = x * x + y * y;
    float r4 = r2 * r2;
    float r6 = r4 * r2;

    float scale = (1 + distorted_params.k1 * r2 + distorted_params.k2 * r4 + distorted_params.k3 * r6);
    float xy = x * y;

    distorted_point[0] = x * scale + 2 * distorted_params.p1 * xy + distorted_params.p2 * (r2 + 2 * x * x);
    distorted_point[1] = y * scale + 2 * distorted_params.p2 * xy + distorted_params.p1 * (r2 + 2 * y * y);
    return distorted_point;
}

} // namespace undistort
