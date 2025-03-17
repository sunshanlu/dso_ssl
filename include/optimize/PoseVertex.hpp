#pragma once

#include <g2o/core/base_vertex.h>
#include <memory>
#include <sophus/se3.hpp>

namespace dso_ssl
{

/// 左扰动位姿顶点
class LeftSE3Vertex : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Vector6d = Eigen::Matrix<double, 6, 1>;
    using SharedPtr = std::shared_ptr<LeftSE3Vertex>;
    using SE3d = Sophus::SE3d;

    LeftSE3Vertex(SE3d init_estimate) { _estimate = std::move(init_estimate); }

    /// 定义位姿部分的更新
    void oplusImpl(const double *update) override
    {
        Vector6d update_e;
        update_e << update[0], update[1], update[2], update[3], update[4], update[5];

        /// 左扰动
        _estimate = SE3d::exp(update_e) * _estimate;
    }

    void setToOriginImpl() override { _estimate = SE3d(); }
};

/// 右扰动位姿顶点
class RightSE3Vertex : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Vector6d = Eigen::Matrix<double, 6, 1>;
    using SharedPtr = std::shared_ptr<RightSE3Vertex>;
    using SE3d = Sophus::SE3d;

    RightSE3Vertex(SE3d init_estimate) { _estimate = std::move(init_estimate); }

    /// 定义位姿部分的更新
    void oplusImpl(const double *update) override
    {
        Vector6d update_e;
        update_e << update[0], update[1], update[2], update[3], update[4], update[5];

        /// 右扰动
        _estimate = _estimate * SE3d::exp(update_e);
    }

    void setToOriginImpl() override { _estimate = SE3d(); }
};

} // namespace dso_ssl