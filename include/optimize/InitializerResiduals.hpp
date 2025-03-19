#pragma once

#include <immintrin.h>

#include <Eigen/Core>
#include <g2o/core/base_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <opencv2/opencv.hpp>

#include "dso/Pattern.hpp"
#include "optimize/IdepthVertex.hpp"
#include "optimize/PhotoAffineVertex.hpp"
#include "optimize/PoseVertex.hpp"
#include "utils/Interpolate.hpp"
#include "utils/ParallelProcess.hpp"
#include "utils/Project.hpp"

namespace dso_ssl
{
/// 初始化部分需要构建的残差
class InitializerPointResidual : public g2o::BaseMultiEdge<Eigen::Dynamic, Eigen::VectorXd>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Vector2f = Eigen::Vector2f;
    using Vector3f = Eigen::Vector3f;
    using Vector2fArray = std::vector<Vector2f, Eigen::aligned_allocator<Vector2f>>;
    using Vector3fArray = std::vector<Vector3f, Eigen::aligned_allocator<Vector3f>>;

    InitializerPointResidual() { resize(3); }

    static float ComputeError(const float &Ii, const float &Ij, const float &aji, const float &bji) { return Ii - aji * Ij - bji; };

    static std::vector<float> ComputeErrorSSE(const std::vector<float> &Ii_data, const std::vector<float> &Ij_data, const __m128 &aji_sse,
                                              const __m128 &bji_sse);

    /**
     * @brief 计算初始化残差，定义多顶点的顺序
     *
     * _vertices[0] 相对位姿
     * _vertices[1] 逆深度
     * _vertices[2] 相对初始化仿射参数 aji和bji
     */
    void computeError() override;

    /// 计算残差相对的雅可比矩阵
    void linearizeOplus() override;

private:
    LeftSE3Vertex *relative_pose_;      ///< 相对位姿
    IdepthVertex *idepth_;              ///< 逆深度
    InitAffineVertex *affine_;          ///< 相对初始化仿射参数
    Pattern::SharedPtr pattern_;        ///< pattern类型
    const cv::Mat *image_i_, *image_j_; ///< 相邻两帧图像
    const cv::Mat *grads_x_, *grads_y_; ///< j帧的图像梯度
    float fx_, fy_, cx_, cy_;           ///< 相机内参

    // 需要暂存的变量，用于计算雅可比矩阵
    Vector3fArray temp_j_;        ///< Pj'的中间变量
    std::vector<float> grads_xj_; ///< pj 的x方向梯度
    std::vector<float> grads_yj_; ///< pj 的y方向梯度
    std::vector<float> Ii_;       ///< pi 的灰度值
    double exp_aji_;              ///< aji 的指数
};

/// 逆深度正则化残差，注意需要输入信息矩阵
class IdepthNormResidual : public g2o::BaseUnaryEdge<1, double, IdepthVertex>
{
public:
    IdepthNormResidual() = default;

    void computeError() override
    {
        idepth_vertex_ = dynamic_cast<IdepthVertex *>(_vertices[0]);
        _error[0] = idepth_vertex_->estimate() - _measurement;
    }

    void linearizeOplus() override { _jacobianOplusXi[0] = 1.0; }

private:
    IdepthVertex *idepth_vertex_;
};

/// tji的正则化残差，注意需要输入信息矩阵
class TjiResidual : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, LeftSE3Vertex>
{
public:
    TjiResidual() = default;

    void computeError() override
    {
        auto se3_vertex = dynamic_cast<LeftSE3Vertex *>(_vertices[0]);

        _error = se3_vertex->estimate().translation();
    }

    void linearizeOplus() override
    {
        _jacobianOplusXi.setZero();

        _jacobianOplusXi.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    }
};

} // namespace dso_ssl
