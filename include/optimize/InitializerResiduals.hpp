#pragma once

#include <immintrin.h>

#include <Eigen/Core>
#include <g2o/core/base_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <opencv2/opencv.hpp>

#include "dso/Pattern.hpp"
#include "optimize/IdepthVertex.hpp"
#include "optimize/InitAffineVertex.hpp"
#include "optimize/PoseVertex.hpp"
#include "utils/Interpolate.hpp"
#include "utils/ParallelProcess.hpp"
#include "utils/Project.hpp"

namespace dso_ssl
{
/// 初始化部分需要构建的残差
class InitializerPhotoResidual : public g2o::BaseMultiEdge<Eigen::Dynamic, Eigen::VectorXd>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum class Status
  {
    OOB,     ///< 投影失败
    OUTLIER, ///< 优化失败
    OK       ///< 优化成功
  };

  using Vector2f = Eigen::Vector2f;
  using Vector3f = Eigen::Vector3f;
  using Vector2fArray = std::vector<Vector2f, Eigen::aligned_allocator<Vector2f>>;
  using Vector3fArray = std::vector<Vector3f, Eigen::aligned_allocator<Vector3f>>;

  InitializerPhotoResidual(Pattern::SharedPtr pattern, cv::Mat *image_i, float outlier_threshold, float fx, float fy, float cx, float cy)
      : pattern_(std::move(pattern))
      , image_i_(std::move(image_i))
      , image_j_(nullptr)
      , grads_x_(nullptr)
      , grads_y_(nullptr)
      , fx_(std::move(fx))
      , fy_(std::move(fy))
      , cx_(std::move(cx))
      , cy_(std::move(cy))
      , is_outlier_(false)
      , compute_times_(0)
      , res_status_(Status::OK)
      , outlier_threshold_(std::move(outlier_threshold))
  {
    resize(3);
    int pattern_size = pattern_->GetPattern().size();
    setDimension(pattern_size);
    _information.setIdentity();
  }

  void UpdateParams(cv::Mat *image_j, cv::Mat *grads_x, cv::Mat *grads_y)
  {
    image_j_ = std::move(image_j);
    grads_x_ = std::move(grads_x);
    grads_y_ = std::move(grads_y);
  }

  /// 重置计算次数，在优化完一轮后调用，当lm中的尝试被接受时，ResetComputeTimes设置为1即可，可防止冗余调用
  void ResetComputeTimes(int compute_times) { compute_times_ = std::move(compute_times); }

  /// 获取逆深度的 jacobian 矩阵
  Eigen::MatrixXd GetIdepthJacobian() const
  {
    // bug 使用多线程策略的时，会出现得到的jacobian为0的情况，这太离奇了
    return _jacobianOplus[1];
  }

  const Status &GetResStatus() const { return res_status_; }

  static float ComputeError(const float &Ii, const float &Ij, const float &aji, const float &bji) { return Ii - aji * Ij - bji; };

  static std::vector<float> ComputeErrorSSE(const std::vector<float> &Ii_data, const std::vector<float> &Ij_data, const __m128 &aji_sse, const __m128 &bji_sse);

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

  bool read(std::istream &is) override { return false; }

  bool write(std::ostream &os) const override { return false; }

  // 判断残差是否大于阈值或者被遮挡
  bool outlier() const { return is_outlier_; }

private:
  Pattern::SharedPtr pattern_;        ///< pattern类型
  const cv::Mat *image_i_, *image_j_; ///< 相邻两帧图像
  const cv::Mat *grads_x_, *grads_y_; ///< j帧的图像梯度
  float fx_, fy_, cx_, cy_;           ///< 相机内参
  bool is_outlier_;                   ///< 是否被判断为外点，大于阈值和被遮挡的情况
  int compute_times_;                 ///< 要求第二次计算时，不更新直接推出即可。
  Status res_status_;                 ///< 残差状态
  float outlier_threshold_;           ///< 外点梯度阈值

  // 需要暂存的变量，用于计算雅可比矩阵
  LeftSE3Vertex *relative_pose_; ///< 相对位姿
  IdepthVertex *idepth_;         ///< 逆深度
  InitAffineVertex *affine_;     ///< 相对初始化仿射参数
  Vector3fArray temp_j_;         ///< Pj'的中间变量
  std::vector<float> grads_xj_;  ///< pj 的x方向梯度
  std::vector<float> grads_yj_;  ///< pj 的y方向梯度
  std::vector<float> Ii_;        ///< pi 的灰度值
  double exp_aji_;               ///< aji 的指数
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

  bool read(std::istream &is) override { return false; }

  bool write(std::ostream &os) const override { return false; }

  void linearizeOplus() override { _jacobianOplusXi[0] = 1.0; }

private:
  IdepthVertex *idepth_vertex_;
};

class IdepthNormIRResidual : public g2o::BaseUnaryEdge<1, double, IdepthVertex>
{
public:
  IdepthNormIRResidual() = default;

  void computeError() override;

  void linearizeOplus() override { _jacobianOplusXi[0] = 1.0; }

  bool read(std::istream &is) override { return false; }

  bool write(std::ostream &os) const override { return false; }

private:
  IdepthVertex *idepth_vertex_;
};

/// tji的正则化残差，注意需要输入信息矩阵
class TjiNormResidual : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, LeftSE3Vertex>
{
public:
  TjiNormResidual() = default;

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

  bool read(std::istream &is) override { return false; }

  bool write(std::ostream &os) const override { return false; }
};

} // namespace dso_ssl
