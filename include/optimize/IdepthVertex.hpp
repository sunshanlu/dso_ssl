#pragma once
#include <g2o/core/base_vertex.h>

namespace dso_ssl
{

/// 逆深度顶点
class IdepthVertex : public g2o::BaseVertex<1, double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Vector2f = Eigen::Vector2f;

    IdepthVertex(double init_estimate, Vector2f pixel_position)
        : pixel_position_(std::move(pixel_position))
    {
        _estimate = std::move(init_estimate);
    }

    void oplusImpl(const double *update) override { _estimate += update[0]; }

    void setToOriginImpl() override { _estimate = 0; }

    Vector2f pixel_position_; ///< host帧像素位置
};

} // namespace dso_ssl
