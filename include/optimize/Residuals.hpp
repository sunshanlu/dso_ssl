#pragma once

#include <immintrin.h>

#include <Eigen/Core>
#include <g2o/core/base_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <opencv2/opencv.hpp>

#include "dso/Pattern.hpp"
#include "optimize/IdepthVertex.hpp"
#include "optimize/PhotoAffineVertex.hpp"
#include "optimize/PoseVertex.hpp"
#include "utils/Interpolate.hpp"

namespace dso_ssl
{
/// 初始化部分需要构建的残差
class InitiallizerResidual : public g2o::BaseMultiEdge<8, Eigen::Matrix<double, 8, 1>>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Vector8d = Eigen::Matrix<double, 8, 1>;

    InitiallizerResidual() { resize(2); }

    /**
     * @brief 计算初始化残差，定义多顶点的顺序
     *
     * _vertices[0] 相对位姿
     * _vertices[1] 逆深度
     * _vertices[2] 相对初始化仿射参数 aji和bji
     */
    void computeError() override
    {
        relative_pose_ = dynamic_cast<LeftSE3Vertex *>(_vertices[0]);
        idepth_ = dynamic_cast<IdepthVertex *>(_vertices[1]);
        affine_ = dynamic_cast<InitAffineVertex *>(_vertices[2]);

        const auto &aji = affine_->estimate()[0];
        const auto &bji = affine_->estimate()[1];

        auto pattern_positions = pattern_->GetPattern();

        for (int start = 0; start < pattern_positions.size() - 3; start += 4)
        {
            alignas(16) float Ii[4], Ij[4];
            float ui[4], vi[4], uj[4], vj[4];
            auto aji_sse = _mm_set1_ps(aji);
            auto bji_sse = _mm_set1_ps(bji);

            // 投影过程，使用SSE加速，并且保存中间变量
            

            interp::BilinInterpSSE(*image_i_, ui, vi);
            interp::BilinInterpSSE(*image_j_, uj, vj);
        }
    }

    /// 计算残差相对的雅可比矩阵
    void linearizeOplus() override;

private:
    LeftSE3Vertex *relative_pose_;      ///< 相对位姿
    IdepthVertex *idepth_;              ///< 逆深度
    InitAffineVertex *affine_;          ///< 相对初始化仿射参数
    Pattern::SharedPtr pattern_;        ///< pattern类型
    const cv::Mat *image_i_, *image_j_; ///< 相邻两帧图像
    const cv::Mat *grads_i_, *grads_j_; ///< 两帧相邻图像的梯度
};
} // namespace dso_ssl
