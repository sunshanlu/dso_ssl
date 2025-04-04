#pragma once
#include <memory>

#include <Eigen/Core>
#include <g2o/core/block_solver.h>
#include <g2o/core/hyper_graph_action.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "dso/Frame.hpp"
#include "dso/Pattern.hpp"
#include "dso/PixelSelector.hpp"
#include "optimize/IdepthVertex.hpp"
#include "optimize/InitializerResiduals.hpp"

namespace dso_ssl
{

struct PixelPoint;

/// 金字塔层级图像内容维护
struct LayerFrame
{
  using SharedPtr = std::shared_ptr<LayerFrame>;
  using PixelPointPtr = std::shared_ptr<PixelPoint>;

  int nlevel_;                              ///< 金字塔层级
  cv::Mat layer_image_;                     ///< 金字塔层级图像
  cv::Mat layer_gradx_;                     ///< 金字塔层级图像的x方向梯度
  cv::Mat layer_grady_;                     ///< 金字塔层级图像的y方向梯度
  std::vector<PixelPointPtr> pixel_points_; ///< 金字塔层级中的像素点信息
};

/// 像素点，维护父节点和相邻节点的关系信息
struct PixelPoint
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Vector2f = Eigen::Vector2f;
  using SharedPtr = std::shared_ptr<PixelPoint>;

  /// 逆深度状态
  enum class Status
  {
    OOB,     ///< 超出图像边界点，未参与优化
    OUTLIER, ///< 没有超出图像边界，但是残差超出阈值
    OK       ///< 优化成功的点
  };

  PixelPoint(float &reg_weight)
      : idepth_(1.0)
      , idepth_avg_(1.0)
      , idepth_hessian_(0.0)
      , idepth_vertex_(nullptr)
      , layer_frame_(nullptr)
      , reg_weight_(reg_weight)
      , status_(Status::OK)
  {
  }

  /// 在优化过程中，考虑点的深度连通性，更新逆深度期望值
  void UpdateIR(const std::vector<InitializerPhotoResidual *> photo_residuals);

  /// 在优化过程后，上下投影过程中，更新逆深度期望值
  void UpdateIR(const std::vector<bool> &make_sense);

  /// 设置为原始内容
  void SetOrigin()
  {
    idepth_ = 1.0;
    idepth_avg_ = 1.0;
    idepth_hessian_ = 0.0;
  }

  int id_;                        ///< 当前点的id
  int nlevel_;                    ///< 所在的金字塔层级
  int parent_id_;                 ///< 上一层父节点id，可以为当点提供优化初值
  std::vector<int> children_ids_; ///< 下一层子节点id，可以使用高斯归一化更新当前逆深度
  std::vector<int> neighbor_ids_; ///< 邻居节点id，用于联系当前层相邻点关系
  Vector2f pixel_position_;       ///< 当前层的像素值

  float idepth_;         ///< 逆深度值
  float idepth_avg_;     ///< 考虑了相邻点关系的逆深度值
  float idepth_hessian_; ///< 逆深度的Hessian值

  IdepthVertex *idepth_vertex_; ///< 逆深度顶点信息
  LayerFrame *layer_frame_;     ///< 像素点所在的layer frame
  const float &reg_weight_;     ///< 逆深度中位数的置信度
  Status status_;               ///< 点的状态，代表逆深度的优化状态
};

class LayerOptimizer;

/// dso系统的初始化器
class Initializer
{
public:
  using PointT = pcl::PointXY;
  using CloudT = pcl::PointCloud<PointT>;
  using KdTree2d = pcl::KdTreeFLANN<PointT>;
  using SharedPtr = std::shared_ptr<Initializer>;
  using Vector2f = Eigen::Vector2f;
  using Vector2fArray = std::vector<Vector2f, Eigen::aligned_allocator<Vector2f>>;
  using SE3d = Sophus::SE3d;
  using SE3f = Sophus::SE3f;
  using LayerOptimizerPtr = std::shared_ptr<LayerOptimizer>;

  struct Options
  {
    using SharedPtr = std::shared_ptr<Options>;

    Options(const std::string &config_path);

    int pyra_levels_;                     ///< 使用的图像金字塔层级
    int neighbor_nums_;                   ///< 邻居点的数量
    std::vector<float> select_densities_; ///< 选择的点密度
    std::vector<int> max_iterations_;     ///< 不同层最大的优化次数
    float reg_weight_;                    ///< 逆深度中位数的置信度
    bool verbose_;                        ///< 优化过程中是否输出内容
    int continue_snap_times_;             ///< 要求连续几次优化满足tji要求
    float huber_threshold_;               ///< huber和函数阈值，对应的photo残差（单维度）
    float outlier_threshold_;             ///< 外点能量阈值
    float inlier_ratio_;                  ///< 优化完成后要求的内点比例
    float alpha_w_;                       ///< 平移不足时，正则化权重
    float alpha_;                         ///< 平移足够时，正则化权重
    float tji_threshold_;                 ///< tji的阈值信息
    float make_sense_ratio_;              ///< 优化比例，更新优化的比例占优化次数的比例
  };

  Initializer(Options::SharedPtr config, PixelSelector::SharedPtr pixel_selector, Pattern::SharedPtr pattern, float fx, float fy, float cx, float cy);

  const std::vector<LayerFrame::SharedPtr> &GetRefLayerInfo() const { return ref_layer_info_; }

  /**
   * @brief 向初始化器中添加帧，试图完成初始化过程
   *
   * 要求连续k帧，满足tji的阈值条件，才认为初始化成功
   *
   * @param frame_ptr 输入的帧
   * @return true     初始化成功
   * @return false    初始化失败
   */
  bool AddActivateFrame(Frame::SharedPtr frame_ptr);

  /**
   * @brief 根据输入的参考帧信息，构建初始化器的图像金字塔信息
   *
   * 1. 构建相邻层的父子点关系，以便使用高斯归一化积进行逆深度传播
   * 2. 构建同一层的相邻关系，以便考虑相邻点之间的逆深度关系
   *
   * @param frame_ptr 输入的参考帧
   */
  void SetReferenceFrame(Frame::SharedPtr frame_ptr);

  /**
   * @brief 构建同层金字塔提取点之间的相邻关系
   *
   * @param layer_frame       输入输出的金字塔层级信息
   * @param selected_points   输入的选择点信息
   * @param layer_kdtree      输出的金字塔层级的kdtree信息
   */
  void BuildNeighborRelation(LayerFrame::SharedPtr &layer_frame, const PixelSelector::Vector2iArray &selected_points, KdTree2d::Ptr &layer_kdtree);

  /**
   * @brief 构建相邻层金字塔提取点之间的父子关系
   *
   * @param cur_layer_frame       输入输出的当前层金字塔层级信息
   * @param last_layer_frame      输入输出的上一层金字塔层级信息
   * @param cur_kdtrees           输入的当前层金字塔层级的kdtree信息
   */
  void BuildParentChildRelation(LayerFrame::SharedPtr &cur_layer_frame, LayerFrame::SharedPtr &last_layer_frame, const KdTree2d::Ptr &cur_kdtrees);

  /**
   * @brief 针对输入的帧，构建初始化器约束，进行优化
   *
   * 1. 当位移tcr 小于阈值时，使用额外的逆深度1约束和位移tcr的模约束，这里可以保证tcr的鲁棒性
   * 2. 当位移tcr 大于阈值时，使用使用idepth_avg_对逆深度进行约束
   * 3. 注意，当tcr优化没有满足阈值条件时，仅保留优化的旋转要求
   */
  bool Optimize(Frame::SharedPtr frame_ptr, SE3f &Tji, float &aji, float &bji, bool &meet_tji);

  /**
   * @brief 将上层点投影到下层来
   *
   * 1. 可以初始化下层点的部分逆深度信息
   * 2. 只将ok部分的逆深度投影到下层去
   * 3. 值得注意的是，ok部分的逆深度存在未优化的部分
   * 4. 不改变下层的逆深度点的状态，每层点的状态仅通过残差部分的computeError来确定
   */
  void PropagateDown(const int &nlevel);

  /**
   * @brief 当优化完成后，使用下层点投影到上层点
   *
   * 1. 时机：某个普通帧，所有的金字塔层优化完成后
   * 2. 从上到下进行点的投影时，使用高斯归一化乘积进行操作
   * 3. 当子点被判断为make_sense的时，父点会被判断为make_sense
   *
   */
  void PropagateUp();

  /**
   * @brief 展示优化结果
   * 1. level，优化层级
   * 2. MeanResBefore，优化前的绝对平均残差
   * 3. MeanResAfter，优化后的绝对平均残差
   * 4. MakeSensePoints，这此优化中，调整的逆深度数量
   * 5. ActualIterations，实际花费的迭代次数
   * 6. MeetSnap，优化后，是否满足snap要求
   * 7. Duriation, 优化耗时
   */
  void OptimizationResultsShow(const int &make_sense_layer, std::ostream &stream = std::cout);

private:
  Options::SharedPtr config_;                        ///< 初始化器配置信息
  std::vector<LayerOptimizerPtr> layer_optimizers_; ///< 层级优化器
  Pattern::SharedPtr pattern_;                      ///< 初始化器构建残差时，使用的pattern
  std::vector<float> fx_, fy_, cx_, cy_;            ///< 层级金字塔相机内参

  bool initialized_;                                  ///< 初始化过程是否完成
  Frame::SharedPtr ref_frame_;                        ///< 初始化器中的参考帧
  std::vector<LayerFrame::SharedPtr> ref_layer_info_; ///< 参考帧的图像金字塔信息
  PixelSelector::SharedPtr pixel_selector_;           ///< 参考帧的像素选择器

  SE3f Tji_estimate_;  ///< 待估计的Tji
  float aji_estimate_; ///< 待估计的aji
  float bji_estimate_; ///< 待估计的bji
};

/// 层级优化器，维护优化的内容
class LayerOptimizer
{
public:
  using SharedPtr = std::shared_ptr<LayerOptimizer>;

  using Matrix1d = Eigen::Matrix<double, 1, 1>;
  using Matrix3d = Eigen::Matrix3d;
  using SE3d = Sophus::SE3d;
  using SE3f = Sophus::SE3f;
  using Vector2f = Eigen::Vector2f;
  using BlockSolver = g2o::BlockSolverX;
  using LinearSolver = g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType>;

  struct OptimizeResultDetails
  {
    using SharedPtr = std::shared_ptr<OptimizeResultDetails>;

    int level_;             ///< 金字塔层级
    float mean_res_before_; ///< 优化前的绝对平均残差
    float mean_res_after_;  ///< 优化后的绝对平均残差
    int make_sense_points_; ///< 在当前层优化中，调整了多少逆深度点
    int actual_iterations_; ///< 该层实际的优化次数
    bool meet_snap_;        ///< 该层优化是否满足snap的要求
    int duration_ms_;       ///< 该层优化所花费的时间，ms
  };

  /// 根据输入的ref layer frame信息，创建优化过程需要的所有vertex和edge
  LayerOptimizer(LayerFrame::SharedPtr ref_layer_frame, int nlevel, float fx, float fy, float cx, float cy, Pattern::SharedPtr pattern,
                 const Initializer::Options::SharedPtr config);

  /// 获取优化结果详情
  const OptimizeResultDetails &GetResultDetails() const { return result_details_; }

  /// 判断inlier的个数
  int GetInlierPhotoResidualNum() const
  {
    int ninliner = 0;
    for (const auto &res : photo_residuals_)
    {
      if (!res->outlier())
        ninliner++;
    }
    return ninliner;
  }

  /**
   * @brief 更新像素点状态
   *
   * 在更新像素点状态之前，需要计算一次当前状态下的残差，来获取最新状态下的残差状态
   *
   * @param pixel_points          输入输出的像素点
   * @param actual_iterations     输入的实际的优化的迭代次数
   * @return int                  输出的为内点的像素点个数
   */
  int UpdatePixelPointsStatus(std::vector<PixelPoint::SharedPtr> &pixel_points, const int &actual_iterations);

  /**
   * @brief 计算逆深度的 hessian 矩阵
   *
   * 计算逆深度的hessian信息，考虑到存在OK状态下的点，但没有hessian信息，只需要计算一次jacobian即可
   * 这是因为，可能存在某些点，仅仅做了computeError的计算过程后得到了OK的状态，但是没有执行优化过程，
   * 因此没有hessian信息，所以在获取jacobian之前，需要执行一次 linearizeOplus
   */
  void ComputeIdepthHessian(std::vector<PixelPoint::SharedPtr> &pixel_points, const int &actual_iterations);

  /**
   * @brief 针对不同的frame_ptr，更新顶点状态的估计值和残差的测量值
   *
   * @param frame_ptr     输入的普通帧
   * @param pixel_points  输入的像素点逆深度估计
   * @param Tji_new       输入的Tji估计
   * @param aji_new       输入的aji估计
   * @param bji_new       输入的bji估计
   */
  void UpdateEstimateAndMeasurement(Frame::SharedPtr frame_ptr, const std::vector<PixelPoint::SharedPtr> &pixel_points, const SE3f &Tji_new,
                                    const float &aji_new, const float &bji_new);

  /**
   * @brief 更新edge边对应的 level，控制优化图
   *
   * @param meet_tji 输入的是否满足tji条件
   */
  void UpdateEdgesLevel(const bool &meet_tji);

  /**
   * @brief 根据输入的新的frame帧，构建某一层的初始化优化问题。
   *
   * 1. 考虑meet_tji的情况，定义两种初始化的优化方法
   *  1.1 当tji不满足条件，使用tji正则化和逆深度1.0的正则化方法
   *  1.2 当tji满足条件时，仅使用逆深度期望的正则化方法
   *
   * 2. 逆深度期望的更新方式
   *  2.1 首先，逆深度期望与邻居逆深度中值和当前状态的逆深度有关系，使用regWeight解决
   *
   * @param frame_ptr         输入的新创建的普通帧
   * @param pixel_points      输入输出的像素点（更新逆深度信息）
   * @param Tji_new           输入输出的Tji
   * @param aji_new           输入输出的aji
   * @param bji_new           输入输出的bji
   * @param meet_tji          输入的是否满足tij要求
   * @param verbose           输入的是否输出优化的verbose信息
   *
   * @return bool 是否优化成功
   */
  bool Optimize(Frame::SharedPtr frame_ptr, std::vector<PixelPoint::SharedPtr> &pixel_points, SE3f &Tji_new, float &aji, float &bji, bool &meet_tji,
                bool verbose = false);

  /**
   * @brief 设置优化后的估计参数
   *
   * 该函数依托于pixel point 的逆深度hessian信息，当逆深度hessian信息存在时，才会将当前估计
   *
   * @param Tji           输出的Tji
   * @param aji           输出的aji
   * @param bji           输出的bji
   * @param pixel_points  输出的逆深度信息
   * @param meet_tji      是否符合tji要求
   */
  void SetEstimateParams(SE3f &Tji, float &aji, float &bji, std::vector<PixelPoint::SharedPtr> &pixel_points, const bool &meet_tji,
                         const int &actual_iterations);

  const int &GetLastActualIterations() const { return last_actual_iterations_; }

  const int &GetAccumulateIterations() const { return accumulate_iterations_; }

private:
  g2o::SparseOptimizer graph_optimizer_;           ///< g2o的图优化器，这里new出来的所有空间都由g2o去析构
  g2o::OptimizationAlgorithmLevenberg *algorithm_; ///< 定义lm优化算法，用来设置 init scale
  OptimizeResultDetails result_details_;           ///< 优化结果详情

  std::vector<g2o::OptimizableGraph::Vertex *> all_vertexes_; ///< 所有顶点信息，用于析构
  std::vector<g2o::OptimizableGraph::Edge *> all_edges_;      ///< 所有边信息，用于析构
  std::vector<IdepthNormResidual *> idepth_norm_one_edges_;   ///< 逆深度正则化1.0的边
  std::vector<IdepthNormIRResidual *> idepth_norm_ir_edges_;  ///< 逆深度正则化ir的边
  std::vector<InitializerPhotoResidual *> photo_residuals_;   ///< 光度残差
  TjiNormResidual *tji_edge_;                                 ///< tji正则化边

  LeftSE3Vertex *relative_pose_;                ///< 相对位姿顶点
  InitAffineVertex *relative_affine_;           ///< 相对仿射参数顶点
  std::vector<IdepthVertex *> idepth_vertexes_; ///< 逆深度顶点

  int nlevel_;                 ///< 图像金字塔层级
  float fx_, fy_, cx_, cy_;    ///< 相机内参
  Pattern::SharedPtr pattern_; ///< 初始化器构建残差时，使用的pattern
  float huber_threshold_;      ///< huber阈值
  float outlier_threshold_;    ///< 外点能量阈值
  float inlier_ratio_;         ///< 优化完成后要求的内点比例
  float alpha_w_;              ///< 不满足tji条件下的正则化权重--> information
  float alpha_;                ///< 满足tji条件下的正则化权重 --> information
  float tji_threshold_;        ///< tji阈值条件
  float make_sense_ratio_;     ///< 优化比例，更新优化的比例占优化次数的比例
  int max_iterations_;         ///< 最大迭代次数
  int accumulate_iterations_;  ///< 初始化过程累计的迭代次数
  int last_actual_iterations_; ///< 最后一次优化的迭代次数
};

} // namespace dso_ssl