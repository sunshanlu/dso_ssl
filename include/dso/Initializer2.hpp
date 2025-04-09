#pragma once

#include <memory>

#include <Eigen/Core>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "dso/Frame.hpp"
#include "dso/Keyframe.hpp"
#include "dso/Pattern.hpp"
#include "dso/PixelSelector.hpp"

namespace dso_ssl
{

class Visualizer;

/**
 * @brief 初始化中的逆深度点
 *
 * 1. 金字塔层级
 * 2. hessian 和 hessian_new
 * 3. idepth 和 idepth_new
 * 4. energy 和 energy_new
 * 5. iR
 * 6. b 和 b_new
 * 7. hessian_fd_ 和 hessian_fd_new_
 */
struct InitIdepthPoint
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using SharedPtr = std::shared_ptr<InitIdepthPoint>;
  using Vector8f = Eigen::Vector<float, 8>;
  using Vector8d = Eigen::Vector<double, 8>;
  using Vector2f = Eigen::Vector2f;

  InitIdepthPoint()
      : iR_(1.0)
      , b_(0.0)
      , only_photo_hessian_(0.0)
      , hessian_(0.0)
      , idepth_(1.0)
      , energy_(0.0)
      , is_update_(false)
  {
    ResetOptimizeInfo();
  }

  void ResetOptimizeInfo()
  {
    b_new_ = 0;
    only_photo_hessian_new_ = 0;
    hessian_new_ = 0;
    energy_new_ = 0;
    hessian_fd_new_.setZero();
    is_good_ = true;
    idepth_new_ = idepth_;
  }

  int nlevel_;                    ///< 金字塔层级
  int parent_id_;                 ///< 父点的id
  std::vector<int> neighbor_ids_; ///< 邻居点的id
  std::vector<int> children_ids_; ///< 子点id
  Vector2f host_pixel_positon_;   ///< 主帧对应的像素坐标

  float iR_;                             ///< 逆深度期望值
  float b_, b_new_;                      ///< 用于计算step
  float only_photo_hessian_;             ///< 优化完成后维护光度残差的hessian
  float only_photo_hessian_new_;         ///< 优化过程中仅维护光度残差的hessian
  float hessian_, hessian_new_;          ///< 维护逆深度点的hessian
  float idepth_, idepth_new_;            ///< 优化过程中维护的逆深度
  float energy_, energy_new_;            ///< 保存地图点的能量值，仅包含光度残差部分
  Vector8d hessian_fd_, hessian_fd_new_; ///< 用于计算schur
  bool is_update_;                       ///< 标记当前点是否更新了idepth_new....
  bool is_good_;                         ///< 在计算残差过程中，判断是否是good点
};

/// dso的初始化器
class Initializer2
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using VisualizerPtr = std::shared_ptr<Visualizer>;
  using SharedPtr = std::shared_ptr<Initializer2>;
  using PointT = pcl::PointXY;
  using CloudT = pcl::PointCloud<PointT>;
  using KdTree2d = pcl::KdTreeFLANN<PointT>;
  using SE3d = Sophus::SE3d;
  using SE3f = Sophus::SE3f;
  using Mat8d = Eigen::Matrix<double, 8, 8>;
  using Mat8f = Eigen::Matrix<float, 8, 8>;
  using Vec8f = Eigen::Vector<float, 8>;
  using Vec8d = Eigen::Vector<double, 8>;
  using Vec3f = Eigen::Vector3f;
  using Vec3d = Eigen::Vector3d;
  using Vec4f = Eigen::Vector4f;
  using Vec4d = Eigen::Vector4d;

  /// 初始化器的配置项
  struct Options
  {
    using SharedPtr = std::shared_ptr<Options>;

    Options(const std::string &filepath);

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
    int require_points_num_;              ///< 初始化过程中需要的初始化点的数量
    int immature_points_num_;             ///< 需要多少未成熟点
  };

  Initializer2(Options::SharedPtr options, PixelSelector::SharedPtr pixel_selector, Pattern::SharedPtr pattern,
               const float fx, const float fy, const float cx, const float cy)
      : options_(std::move(options))
      , pattern_(std::move(pattern))
      , pixel_selector_(std::move(pixel_selector))
      , init_idepth_points_(options_->pyra_levels_)
  {
    Tji_ = Tji_new_ = SE3f();
    aji_ = aji_new_ = 0;
    bji_ = bji_new_ = 0;

    for (int level = 0; level < options_->pyra_levels_; ++level)
    {
      float down_scale = std::pow(2, level);
      init_fx_.push_back(fx / down_scale);
      init_fy_.push_back(fy / down_scale);
      init_cx_.push_back(cx / down_scale);
      init_cy_.push_back(cy / down_scale);
    }
  }

  /**
   * @brief 当初始化完成后，获取由初始化器构建的关键帧信息
   *
   * @param keyframe0 输出的参考关键帧，带有逆深度信息
   * @param keyframe1 输出的最后跟踪的参考关键帧，带有未成熟点信息
   */
  void GetKeyFrames(KeyFrame::SharedPtr &keyframe0, KeyFrame::SharedPtr &keyframe1)
  {
    keyframe0 = keyframe0_;
    keyframe1 = keyframe1_;
  }

  /**
   * @brief 设置初始化器的优化参数
   *
   * 将优化过程中的参数和维护的参数都进行了设置
   *
   * @param Tji 输入的Tji优化初始值
   * @param aji 输入的aji优化初始值
   * @param bji 输入的bji优化初始值
   */
  void SetOptimizationParams(const SE3f &Tji, const float &aji, const float &bji)
  {
    Tji_ = Tji_new_ = Tji;
    aji_ = aji_new_ = aji;
    bji_ = bji_new_ = bji;
  }

  /**
   * @brief 初始化器的后处理过程
   *
   * 1. 根据初始化器参考帧，构建关键帧0
   * 2. 根据跟踪最后帧，构建关键帧1
   * 3. 根据初始化器的优化点，构建关键帧0的地图点（逆深度，host为关键帧0）
   * 4. 提取参考关键帧1的未成熟地图点，以便后续跟踪过程对未成熟点的更新
   */
  void PostProcess();

  /**
   * @brief 获取优化参数
   *
   * @param Tji 输出的Tji优化值
   * @param aji 输出的aji优化值
   * @param bji 输出的bji优化值
   */
  void GetOptimizationParams(SE3f &Tji, float &aji, float &bji)
  {
    Tji = Tji_;
    aji = aji_;
    bji = bji_;
  }

  /**
   * @brief 将正规方程得到的解进行应用到临时优化变量上
   *
   * 1. 更新Tji、aji、bji，直接使用inc即可
   * 2. 更新参考关键帧的逆深度信息
   *
   * @param lvl     输入的金字塔优化层级
   * @param inc     输入的正规方程得到的增量信息
   * @param lambda  输入的lambda值，考虑lm方法
   */
  void ApplyStep(const int &lvl, const Vec8f &inc, const float &lambda);

  /**
   * @brief 更新逆深度的期望值
   *
   * 1. 有options_->reg_weight_的比例相信周围逆深度组成的IR
   * 2. 有(1 - options_->reg_weight_)的比例相信当前点的逆深度
   *
   * @param idx         输入的待更新逆深度期望的索引位置
   * @param level       输入的待更新点所在的金字塔层级
   * @param make_sense  输入的与当前层所有像素点，是否有意义
   */
  void UpdateIR(const int &idx, const int &level, const std::vector<bool> &make_sense);

  /**
   * @brief 将上层点投影到下层来
   *
   * @param nlevel nlevel层投影到 nlevel - 1 层
   *
   * 1. 可以初始化下层点的部分逆深度信息
   * 2. 将被判断为是内点，且参与过优化的逆深度投影进行投影继承
   * 3. 只有当snap为true时，PropagateDown才有意义
   */
  void PropagateDown(const int &nlevel);

  /**
   * @brief 当优化完成后，使用下层点投影到上层点 nlevel -> nlevel + 1
   *
   * 1. 时机：某个普通帧，所有的金字塔层优化完成后
   * 2. 从上到下进行点的投影时，使用高斯归一化乘积进行操作
   * 3. 当子点被判断为make_sense的时，父点会被判断为make_sense
   * 4. 注意，需要在投影到上一层后，进行IR更新
   */
  void PropagateUp();

  /**
   * @brief 跟踪激活帧
   *
   * 1. 设置初始化器的参考关键帧（初始化逆深度信息，层级之间的相邻关系和同层之间的邻居关系）
   * 2. 使用图像金字塔完成由粗到精细的单目初始化
   *  2.1 根据投影模型，计算残差状态
   *  2.2 根据雅可比公式，计算正规方程的H和b矩阵
   *  2.3 使用LM方法，求解
   *
   * @param frame
   * @return true
   * @return false
   */
  bool TrackActivateFrame(Frame::SharedPtr frame);

  /**
   * @brief 计算能量、正规方程和schur边缘化后的正规方程
   *
   * 1. 对于待优化的变量 Tji、aji 和 bji，在遍历过程中，始终使用 H_d 和 b_d 进行维护
   * 2. 对于待优化的变量 dpi, 在遍历过程中，某个点结束后需维护 Hfd、Hdd和bd
   * 3. 针对某个点，计算完成后 根据snap条件，考虑逆深度的正则化信息，变更 Hdd 和 bd
   * 4. 然后，进行schur计算，用来维护 Hsc和bsc信息，考虑了逆深度的正则化信息
   * 5. 根据snap情况，维护tji的正则化信息，变更 H_d和b_d
   *
   * 对于优化过程中的大数吃小数问题，这里使用 double 进行正规方程的维护，然后求解完成后变更为float
   * 在计算jacobian和error过程中，不会对snap的情况进行变更，这里提倡即便snap发生了变更，这里仍然
   * 需要先进行原先snap的计算，然后判断是否accept,如果accept,那么就再次执行snap变更后的行为即可。
   *
   * @param level 输入的要优化的图像金字塔层级
   * @param H     输出的H矩阵
   * @param b     输出的b矩阵
   * @param Hsc   输出的schur的H矩阵
   * @param bsc   输出的schur的b矩阵
   * @param snap  输入的是否snap，tji达到要求
   * @return Initializer2::Vec4f  [energy_photo, energy_idepth_norm, energy_tji_norm, ngood]
   */
  Vec4f ComputeJacobianAndError(const int &level, Mat8f &H, Vec8f &b, Mat8f &Hsc, Vec8f &bsc, const bool &snap);

  /**
   * @brief 设置初始化器的参考帧
   *
   * 1. 根据Options的pyra_levels_和select_densities_，使用像素选择器构建参考帧的像素点
   * 2. 根据Options的neighbor_nums_，构建参考帧的邻居关系，使用KDTree进行选择
   * 3. 维护下一层点和上一层点之间的父子关系，使用KDTree进行选择
   *
   * @param frame 输入的普通帧，将其设置为参考帧
   */
  void SetReference(Frame::SharedPtr frame);

  /**
   * @brief 移除像素选择器中由于pattern的size导致的外点
   *
   * 当pattern不同时，会要求点距离图像边缘有一定的距离，这个距离要保证pi点完全不用考虑
   * 外点的情况，在投影过程中仅仅需要关注pj点即可。
   *
   * @param selected_points 输入初步选择的像素点，输出处理后的点
   * @param rows            图像的行数
   * @param cols            图像的列数
   */
  void RemoveOutlierSelected(PixelSelector::Vector2iArray &selected_points, const int &rows, const int &cols);

  /**
   * @brief 构建level层上的相邻点关系
   *
   * 1. 使用KdTree进行搜索，构建相邻点关系
   * 2. 使用c++17 标准的并行化策略，加速
   *
   * @param level   输入的要处理的金字塔层级
   * @param cloud   输入的要处理的点云
   * @param kdtree  输入的基于cloud构建的kdtree
   */
  void BuildNeighborAss(const int &level, const CloudT::ConstPtr &cloud, const KdTree2d::ConstPtr &kdtree);

  /// 供测试使用
  const Frame::SharedPtr &GetRefframe() const { return reference_frame_; }

  /// 供测试使用
  const std::vector<std::vector<InitIdepthPoint::SharedPtr>> &GetPoints() const { return init_idepth_points_; }

  /// 供测试使用
  void SetCurrentFrame(Frame::SharedPtr frame) { current_frame_ = std::move(frame); }

  /**
   * @brief 构建当前成和上一层之间的父子关系
   *
   * 1. 遍历当前层上的点，将其点投影到上一层上去
   * 2. 使用上一层的kdtree进行投影点最紧邻点搜索，获取parent的id信息
   * 3. 维护父子关系
   * 4. 这里不使用多线程，因为父点可能会产生竞争关系
   *
   * @param level       当前层级
   * @param prev_kdtree 上一层级的kdtree
   * @param curr_cloud  当前层级的kdtree
   */
  void BuildParentChildAss(const int &level, const KdTree2d::ConstPtr &prev_kdtree, const CloudT::ConstPtr &curr_cloud);

  /// 设置可视化器
  void SetVisualizer(VisualizerPtr visualizer);

  ~Initializer2() = default;

private:
  SE3f Tji_, Tji_new_;                  ///< 相对位姿变换
  float aji_, aji_new_, bji_, bji_new_; ///< aji和bji

  Frame::SharedPtr reference_frame_;        ///< 参考帧
  Frame::SharedPtr current_frame_;          ///< 当前帧
  Options::SharedPtr options_;              ///< 初始化器配置选项
  Pattern::SharedPtr pattern_;              ///< pattern点，共享逆深度
  PixelSelector::SharedPtr pixel_selector_; ///< 像素选择器

  std::vector<std::vector<InitIdepthPoint::SharedPtr>> init_idepth_points_; ///< 初始化过程中的逆深度点
  std::vector<float> init_fx_, init_fy_, init_cx_, init_cy_;                ///< 初始化过程中的相机内参

  VisualizerPtr visualizer_; ///< 可视化器，用于初始化器的可视化

  KeyFrame::SharedPtr keyframe0_; ///< 初始化完成后，由参考帧构建的关键帧
  KeyFrame::SharedPtr keyframe1_; ///< 初始化完成后，由最后一个跟踪帧构建的关键帧
};

} // namespace dso_ssl
