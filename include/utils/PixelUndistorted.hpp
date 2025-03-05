#pragma once
#include <memory>

#include <Eigen/Core>

namespace undistort
{

struct DistortedParams
{
    using SharedPtr = std::shared_ptr<DistortedParams>;

    virtual void LoadParams(const std::vector<float> &params) = 0;

    virtual ~DistortedParams() = default;
};

/// 定义畸变类型
enum class DistortionT
{
    Pinhole,
    RadTan3,
    RadTan5,
    FOV,
    KB
};

/// @brief Pinhole 无畸变模型
struct PinholeParams : public DistortedParams
{
    using SharedPtr = std::shared_ptr<PinholeParams>;

    void LoadParams(const std::vector<float> &params) override
    {
        assert(!params.empty() && "Pinhole params size must be 0");
        return;
    }
};

template <int N>
struct RadTanParams;

/// @brief RadTan 3参数畸变模型
template <>
struct RadTanParams<3> : public DistortedParams
{
    using SharedPtr = std::shared_ptr<RadTanParams<3>>;

    void LoadParams(const std::vector<float> &params) override
    {
        assert(params.size() >= 3 && "RadTan3 params size must be 3");

        k1 = params[0];
        p1 = params[1];
        p2 = params[2];
    }

    float k1;
    float p1, p2;
};

/// @brief RadTan 5参数畸变模型
template <>
struct RadTanParams<5> : public DistortedParams
{
    using SharedPtr = std::shared_ptr<RadTanParams<5>>;

    void LoadParams(const std::vector<float> &params) override
    {
        assert(params.size() >= 5 && "RadTan5 params size must be 5");

        k1 = params[0];
        k2 = params[1];
        k3 = params[2];
        p1 = params[3];
        p2 = params[4];
    }

    float k1, k2, k3;
    float p1, p2;
};

/// @brief FOV 畸变模型参数
struct FOVParams : public DistortedParams
{
    using SharedPtr = std::shared_ptr<FOVParams>;

    void LoadParams(const std::vector<float> &params) override
    {
        assert(params.size() >= 1 && "FOV params size must be 1");

        omega = params[0];
    }

    float omega;
};

/// @brief KannalaBrandt 畸变模型参数
struct KBParams : public DistortedParams
{
    using SharedPtr = std::shared_ptr<KBParams>;

    void LoadParams(const std::vector<float> &params) override
    {
        assert(params.size() >= 4 && "EQUI params size must be 4");

        k1 = params[0];
        k2 = params[1];
        k3 = params[2];
        k4 = params[3];
    }

    float k1, k2;
    float k3, k4;
};

/// @brief RadTan 畸变模型
template <int N>
Eigen::Vector2f RadTan(const RadTanParams<N> &distorted_params, const Eigen::Vector2f &undistorted_point);

/// RadTan 3参数畸变模型的公式描述
template <>
Eigen::Vector2f RadTan<3>(const RadTanParams<3> &distorted_params, const Eigen::Vector2f &undistorted_point);

/// RadTan 5参数畸变模型的公式描述
template <>
Eigen::Vector2f RadTan<5>(const RadTanParams<5> &distorted_params, const Eigen::Vector2f &undistorted_point);

/// FOV 畸变模型
Eigen::Vector2f FOV(const FOVParams &distorted_params, const Eigen::Vector2f &undistorted_point);

/// KannalaBrandt 畸变模型
Eigen::Vector2f KB(const KBParams &distorted_params, const Eigen::Vector2f &undistorted_point);

} // namespace undistort
