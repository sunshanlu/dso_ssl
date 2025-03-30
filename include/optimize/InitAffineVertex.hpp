#pragma once

#include <g2o/core/base_vertex.h>

namespace dso_ssl
{

/// 初始化器使用的仿射参数
class InitAffineVertex : public g2o::BaseVertex<2, Eigen::Vector2d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Vector2d = Eigen::Matrix<double, 2, 1>;

    InitAffineVertex() = default;

    void oplusImpl(const double *update) override
    {
        _estimate[0] += update[0];
        _estimate[1] += update[1];
    }

    void setToOriginImpl() override { _estimate = Vector2d::Zero(); }

    bool read(std::istream &is) override { return false; }

    bool write(std::ostream &os) const override { return false; }
};

} // namespace dso_ssl