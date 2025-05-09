#include <algorithm>
#include <random>

#include "dso/Initializer2.hpp"
#include "dso/Visualizer.hpp"
#include "utils/Interpolate.hpp"
#include "utils/Project.hpp"

namespace dso_ssl
{

/**
 * @brief 设置初始化器的参考帧
 *
 * 1. 根据Options的pyra_levels_和select_densities_，使用像素选择器构建参考帧的像素点
 * 2. 根据Options的neighbor_nums_，构建参考帧的邻居关系，使用KDTree进行选择
 * 3. 维护下一层点和上一层点之间的父子关系，使用KDTree进行选择
 *
 * @param frame 输入的普通帧，将其设置为参考帧
 */
void Initializer2::SetReference(Frame::SharedPtr frame)
{
  reference_frame_ = std::move(frame);

  cv::Mat squre_grad = reference_frame_->GetSqureGrad();
  auto images_and_grads = reference_frame_->GetPyrdImageAndGrads();

  std::vector<int> selected_points_conf;
  PixelSelector::Vector2iArray selected_points;
  KdTree2d::Ptr curr_kdtree;
  CloudT::Ptr prev_cloud, curr_cloud;
  for (int level = 0; level < options_->pyra_levels_; ++level)
  {
    curr_kdtree = pcl::make_shared<KdTree2d>();
    curr_cloud = pcl::make_shared<CloudT>();

    cv::Mat image_and_grad = images_and_grads[level];
    int selected_num = options_->select_densities_[level] * image_and_grad.rows * image_and_grad.cols;

    // 使用像素选择器提取像素点
    if (level == 0)
      pixel_selector_->SelectFirstLayer(selected_num, squre_grad, selected_points, selected_points_conf);
    else
      pixel_selector_->SelectOtherLayer(selected_num, image_and_grad, selected_points);

    // 删除外点，考虑pattern的尺寸影响
    RemoveOutlierSelected(selected_points, image_and_grad.rows, image_and_grad.cols);

    // 构建点云、kdtree 并维护逆深度点信息
    for (int idx = 0; idx < selected_points.size(); ++idx)
    {
      PointT curr_point;
      curr_point.x = selected_points[idx][0];
      curr_point.y = selected_points[idx][1];
      curr_cloud->points.push_back(curr_point);

      auto init_idepth_point = std::make_shared<InitIdepthPoint>();
      init_idepth_point->nlevel_ = level;
      init_idepth_point->host_pixel_positon_ = selected_points[idx].cast<float>();
      init_idepth_points_[level].push_back(init_idepth_point);
    }

    curr_kdtree->setInputCloud(curr_cloud);

    BuildNeighborAss(level, curr_cloud, curr_kdtree);
    if (level > 0)
      BuildParentChildAss(level, curr_kdtree, prev_cloud);

    prev_cloud = curr_cloud;
  }
}

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
void Initializer2::RemoveOutlierSelected(PixelSelector::Vector2iArray &selected_points, const int &rows,
                                         const int &cols)
{
  // 由于pattern的作用，需要进一步处理selected_points里面不合法的像素点
  PixelSelector::Vector2iArray temp;
  auto half_pattern_size = pattern_->GetHalfPatternSize();
  for (const auto &point: selected_points)
  {
    if (point[0] > half_pattern_size && point[0] < cols - half_pattern_size && point[1] > half_pattern_size &&
        point[1] < rows - half_pattern_size)
      temp.push_back(point);
  }
  std::swap(selected_points, temp);
  temp.clear();
}

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
void Initializer2::BuildNeighborAss(const int &level, const CloudT::ConstPtr &cloud, const KdTree2d::ConstPtr &kdtree)
{
  auto neighbor_process = [&](const int &idx)
  {
    std::vector<float> dists;
    std::vector<int> neightbor_idx;
    if (kdtree->nearestKSearch(cloud->points[idx], options_->neighbor_nums_ + 1, neightbor_idx, dists) !=
        options_->neighbor_nums_ + 1)
      throw std::runtime_error("neighbor search error");

    for (int i = 1; i < options_->neighbor_nums_ + 1; ++i)
      init_idepth_points_[level][idx]->neighbor_ids_.push_back(neightbor_idx[i]);
  };

  std::vector<int> indices(cloud->size());
  std::iota(indices.begin(), indices.end(), 0);
  std::for_each(std::execution::par, indices.begin(), indices.end(), neighbor_process);
}

/**
 * @brief 构建当前成和上一层之间的父子关系
 *
 * 1. 遍历当前层上的点，将其点投影到上一层上去
 * 2. 使用上一层的kdtree进行投影点最紧邻点搜索，获取parent的id信息
 * 3. 维护父子关系
 * 4. 这里不使用多线程，因为父点可能会产生竞争关系
 *
 * @param level       当前层级
 * @param curr_kdtree 当前层级的kdtree，维护的是父点的kdtree
 * @param prev_cloud  上一层级的cloud，维护的是子点的点云信息
 */
void Initializer2::BuildParentChildAss(const int &level, const KdTree2d::ConstPtr &curr_kdtree,
                                       const CloudT::ConstPtr &prev_cloud)
{
  // level 层为父点，level - 1 层为子点
  auto parent_child_process = [&](const int &idx)
  {
    // 将 last_layer_frame 上的点，投影到 cur_layer_frame 中
    PointT parent_point;
    const auto &child_point = prev_cloud->points[idx];
    parent_point.x = child_point.x / 2;
    parent_point.y = child_point.y / 2;

    // 搜索最近邻居点
    std::vector<int> k_indices;
    std::vector<float> k_distances;
    if (curr_kdtree->nearestKSearch(parent_point, 1, k_indices, k_distances) != 1)
      throw std::runtime_error("kdtree search error");

    // 维护父子关系
    init_idepth_points_[level - 1][idx]->parent_id_ = k_indices[0];
    init_idepth_points_[level][k_indices[0]]->children_ids_.push_back(idx);
  };

  // 父点的children_ids的push_back函数会出现数据竞争问题
  std::vector<int> indices(prev_cloud->size());
  std::iota(indices.begin(), indices.end(), 0);
  std::for_each(indices.begin(), indices.end(), parent_child_process);
}

/**
 * @brief 跟踪新的帧，试图初始化整个DSO系统
 *
 * 1. 从粗到精的初始化的优化过程
 * 2. snap到底在哪里进行判断不会影响整个优化过程之间的accpet判断
 * 3. 在某个帧来到后，首先应该判断snap条件，然后根据snap条件进行状态重置
 * 4. 从上到下进行逆深度点的初始化
 * 5. 从下到上进行逆深度点的修正
 *
 * @note 当return true时，输入的frame帧还没有进行跟踪操作
 *
 * @param frame   输入的待跟踪的帧
 * @return true   输出初始化成功
 * @return false  输出初始化失败
 */
bool Initializer2::TrackActivateFrame(Frame::SharedPtr frame)
{
  static int continus_snap = 0;

  bool snap = Tji_.translation().norm() > options_->tji_threshold_;

  if (snap)
  {
    ++continus_snap;
    if (continus_snap >= options_->continue_snap_times_)
    {
      PostProcess();
      return true;
    }
  }
  else
    continus_snap = 0;

  current_frame_ = std::move(frame);
  Tji_new_ = Tji_;
  aji_new_ = aji_;
  bji_new_ = bji_;

  float ti = reference_frame_->GetExposureTime();
  float tj = current_frame_->GetExposureTime();
  if (ti > 0 && tj > 0)
    aji_ = aji_new_ = std::log(tj / ti);

  if (!snap)
  {
    Tji_new_.translation().setZero();
    for (int level = 0; level < options_->pyra_levels_ - 1; ++level)
    {
      auto level_points = init_idepth_points_[level];
      for (auto &point: level_points)
      {
        point->ResetOptimizeInfo();
        point->iR_ = 1.0f;
      }
    }
  }

  static float eps = 1e-4, min_lambda = 1e-4, max_lambda = 1e4;
  static int max_fails_times = 2;

  for (int level = options_->pyra_levels_ - 1; level >= 0; --level)
  {
    Mat8f H, Hsc;
    Vec8f b, bsc;
    Vec4f res_old = ComputeJacobianAndError(level, H, b, Hsc, bsc, snap);

    float lambda = 0.1;
    int fails_times = 0;
    for (int iteration = 0; iteration < options_->max_iterations_[level]; ++iteration)
    {
      // 保存之前的正规方程状态，以便回溯状态
      Mat8f H_copy = H, Hsc_copy = Hsc;
      Vec8f b_copy = b, bsc_copy = bsc;

      for (int i = 0; i < 8; ++i)
        H(i, i) *= (1 + lambda);

      bsc /= (1 + lambda);
      Hsc /= (1 + lambda);
      Mat8f Hl = H - Hsc;
      Vec8f bl = b - bsc;

      Vec8f inc = -Hl.ldlt().solve(bl);

      // 在应用增量到new之前，需要先拷贝new状态，方便backup
      ApplyStep(level, inc, lambda);

      Vec4f res_new = ComputeJacobianAndError(level, H, b, Hsc, bsc, snap);

      float mean_energy_old =
              res_old[0] / res_old[3] + res_old[1] / res_old[3] + res_old[2] / init_idepth_points_[level].size();
      float mean_energy_new =
              res_new[0] / res_new[3] + res_new[1] / res_new[3] + res_new[2] / init_idepth_points_[level].size();

      // 如果平均能量降低，则认为接受了当前优化
      bool accept = mean_energy_new < mean_energy_old;
      if (accept)
      {
        Tji_ = Tji_new_;
        aji_ = aji_new_;
        bji_ = bji_new_;

        for (auto &lvl_point: init_idepth_points_[level])
        {
          if (lvl_point->is_update_)
          {
            if (lvl_point->is_good_)
            {
              // 只有当接受，且被认定为good条件的点，才会被接受new
              lvl_point->b_ = lvl_point->b_new_;
              lvl_point->only_photo_hessian_ = lvl_point->only_photo_hessian_new_;
              lvl_point->hessian_ = lvl_point->hessian_new_;
              lvl_point->idepth_ = lvl_point->idepth_new_;
              lvl_point->energy_ = lvl_point->energy_new_;
              lvl_point->hessian_fd_ = lvl_point->hessian_fd_new_;
            }
            else
            {
              // 当new被认为是外点时，需要将这部分内容进行backup
              lvl_point->b_new_ = lvl_point->b_;
              lvl_point->only_photo_hessian_new_ = lvl_point->only_photo_hessian_;
              lvl_point->hessian_new_ = lvl_point->hessian_;
              lvl_point->idepth_new_ = lvl_point->idepth_;
              lvl_point->energy_new_ = lvl_point->energy_;
              lvl_point->hessian_fd_new_ = lvl_point->hessian_fd_;
            }
            lvl_point->is_update_ = false;
          }
        }

        // 调整lm方法的参数，lambda递减、fails_times置0
        lambda *= 0.5;
        fails_times = 0;
        if (lambda < min_lambda)
          lambda = min_lambda;

        std::swap(res_new, res_old);
      }
      else
      {
        // 如果不接受，我就将之前被更新的状态进行backup
        Tji_new_ = Tji_;
        aji_new_ = aji_;
        bji_new_ = bji_;
        for (auto &lvl_point: init_idepth_points_[level])
        {
          if (lvl_point->is_update_)
          {
            lvl_point->b_new_ = lvl_point->b_;
            lvl_point->only_photo_hessian_new_ = lvl_point->only_photo_hessian_;
            lvl_point->hessian_new_ = lvl_point->hessian_;
            lvl_point->idepth_new_ = lvl_point->idepth_;
            lvl_point->energy_new_ = lvl_point->energy_;
            lvl_point->hessian_fd_new_ = lvl_point->hessian_fd_;
            lvl_point->is_update_ = false;
            lvl_point->is_good_ = true;
          }
        }

        lambda *= 4;
        ++fails_times;
        if (lambda > max_lambda)
          lambda = max_lambda;

        // 这里将优化的正规方程backup
        H = H_copy;
        b = b_copy;
        Hsc = Hsc_copy;
        bsc = bsc_copy;
      }

      // 调试输出一些内容
      if (options_->verbose_)
      {
        if (!accept)
          std::cout << "level: " << level << "\titeration: " << iteration
                    << "\tsqrt(energy / nums): " << std::sqrt(res_old[0] / res_old[3]) << "->"
                    << std::sqrt(res_new[0] / res_new[3]) << "\tinlier num: " << static_cast<int>(res_old[3]) << "->"
                    << static_cast<int>(res_new[3]) << "\tReject" << std::endl;
        else
          std::cout << "level: " << level << "\titeration: " << iteration
                    << "\tsqrt(energy / nums): " << std::sqrt(res_new[0] / res_new[3]) << "->"
                    << std::sqrt(res_old[0] / res_old[3]) << "\tinlier num: " << static_cast<int>(res_new[3]) << "->"
                    << static_cast<int>(res_old[3]) << "\tAccept" << std::endl;
      }

      if (inc.norm() < eps || fails_times >= max_fails_times)
        break;
    }

    if (level > 0 && snap)
      PropagateDown(level);
  }

  if (snap)
    PropagateUp();

  if (visualizer_)
  {
    visualizer_->UpdateReferenceFrame(reference_frame_, init_idepth_points_[0]);
    visualizer_->UpdateTrackingFrame(current_frame_);
    visualizer_->UpdateInitializerCloud(init_idepth_points_[0], init_fx_[0], init_fy_[0], init_cx_[0], init_cy_[0]);
  }

  return false;
}

/// 设置可视化器
void Initializer2::SetVisualizer(VisualizerPtr visualizer) { visualizer_ = std::move(visualizer); }

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
Initializer2::Vec4f Initializer2::ComputeJacobianAndError(const int &level, Mat8f &H, Vec8f &b, Mat8f &Hsc, Vec8f &bsc,
                                                          const bool &snap)
{
  // 防止精度问题，使用double类型做累加
  Mat8d H_d, H_sc;
  Vec8d b_d, b_sc;

  H_d.setZero();
  H_sc.setZero();
  b_d.setZero();
  b_sc.setZero();

  double energy_photo = 0, energy_idepth_norm = 0, energy_tji_norm = 0;

  auto &level_idepth_points = init_idepth_points_[level];
  const auto &pattern = pattern_->GetPattern();

  const float &lfx = init_fx_[level], &lfy = init_fy_[level];
  const float &lcx = init_cx_[level], &lcy = init_cy_[level];
  float exp_aji = std::exp(aji_new_);

  Eigen::Matrix3f Ki;
  Ki << 1.0 / lfx, 0, -lcx / lfx, 0, 1.0 / lfy, -lcy / lfy, 0, 0, 1;

  Eigen::Matrix3f RKi = Tji_new_.rotationMatrix() * Ki;
  Eigen::Vector3f tji = Tji_new_.translation();

  cv::Mat ref_image_and_grad = reference_frame_->GetPyrdImageAndGrads()[level];
  cv::Mat cur_image_and_grad = current_frame_->GetPyrdImageAndGrads()[level];

  int n_good = 0;
  for (int idx = 0; idx < level_idepth_points.size(); ++idx)
  {
    // 将pi点投影到pj点上，然后根据aji、bji内容构建残差
    const auto &point_ref = level_idepth_points[idx];
    Eigen::Vector3f tjidpi = tji * point_ref->idepth_new_;
    point_ref->energy_new_ = 0;
    point_ref->is_good_ = true;
    point_ref->b_new_ = 0;
    point_ref->hessian_new_ = 0;
    point_ref->hessian_fd_new_.setZero();
    point_ref->only_photo_hessian_new_ = 0;

    for (int pidx = 0; pidx < pattern.size(); ++pidx)
    {
      Eigen::Vector2f point_i = point_ref->host_pixel_positon_ + pattern[pidx];
      Eigen::Vector3f pj_temp = RKi * Eigen::Vector3f(point_i[0], point_i[1], 1) + tjidpi;

      float idepth_pj = point_ref->idepth_new_ / pj_temp[2];
      float uj = pj_temp[0] / pj_temp[2], vj = pj_temp[1] / pj_temp[2];
      float kuj = lfx * uj + lcx, kvj = lfy * vj + lcy;

      if (kuj < 1 || kuj > ref_image_and_grad.cols - 2 || kvj < 1 || kvj > ref_image_and_grad.rows - 2)
      {
        // 当投影的pj点逃出边界时，认为该点不可用，能量继续保持
        point_ref->is_good_ = false;
        point_ref->energy_new_ = point_ref->energy_;
        break;
      }

      const float &Ii = ref_image_and_grad.at<cv::Vec3f>(point_i[1], point_i[0])[0];
      Eigen::Vector3f Ij_and_grad = interp::BilinInterp3(cur_image_and_grad, kuj, kvj);

      // 计算残差，并统计能量值
      float residual = Ij_and_grad[0] - exp(aji_new_) * Ii - bji_new_;
      float hw = std::abs(residual) <= options_->huber_threshold_ ? 1 : options_->huber_threshold_ / std::abs(residual);
      float hw_sqrt = std::sqrt(hw);
      point_ref->energy_new_ += hw * residual * residual * (2 - hw);

      // 计算jacobian矩阵
      Vec8d Jacobian;
      float dxfxhw_sqrt = Ij_and_grad[1] * lfx * hw_sqrt;
      float dyfyhw_sqrt = Ij_and_grad[2] * lfy * hw_sqrt;

      Jacobian[0] = dxfxhw_sqrt * idepth_pj;
      Jacobian[1] = dyfyhw_sqrt * idepth_pj;
      Jacobian[2] = -(dxfxhw_sqrt * uj * idepth_pj + dyfyhw_sqrt * vj * idepth_pj);
      Jacobian[3] = -dxfxhw_sqrt * uj * vj - dyfyhw_sqrt * (1 + vj * vj);
      Jacobian[4] = dxfxhw_sqrt * (1 + uj * uj) + dyfyhw_sqrt * uj * vj;
      Jacobian[5] = -dxfxhw_sqrt * vj + dyfyhw_sqrt * uj;
      Jacobian[6] = -hw_sqrt * exp_aji * Ii;
      Jacobian[7] = -hw_sqrt;
      float drk_ddpi =
              dxfxhw_sqrt * (tji[0] - uj * tji[2]) / pj_temp[2] + dyfyhw_sqrt * (tji[1] - vj * tji[2]) / pj_temp[2];
      float rk = residual * hw_sqrt;

      point_ref->only_photo_hessian_new_ += drk_ddpi * drk_ddpi;
      point_ref->hessian_fd_new_ += Jacobian * drk_ddpi;
      point_ref->b_new_ += drk_ddpi * rk;
      H_d += Jacobian * Jacobian.transpose();
      b_d += Jacobian * rk;
    }

    if (!point_ref->is_good_ || point_ref->energy_new_ > options_->outlier_threshold_ || point_ref->idepth_new_ < 1e-3)
    {
      point_ref->is_good_ = false;
      point_ref->energy_new_ = point_ref->energy_;
      continue;
    }

    energy_photo += point_ref->energy_new_;

    point_ref->hessian_new_ = point_ref->only_photo_hessian_new_;

    if (!snap)
    {
      float point_norm_rk = (point_ref->idepth_new_ - 1.0);
      point_ref->hessian_new_ += options_->alpha_w_;
      point_ref->b_new_ += options_->alpha_w_ * point_norm_rk;
      energy_idepth_norm += options_->alpha_w_ * (point_norm_rk * point_norm_rk);
    }
    else
    {
      float point_norm_rk = (point_ref->idepth_new_ - point_ref->iR_);
      point_ref->hessian_new_ += options_->alpha_;
      point_ref->b_new_ += options_->alpha_ * point_norm_rk;
      energy_idepth_norm += options_->alpha_ * (point_norm_rk * point_norm_rk);
    }

    // 计算schur
    float Hdd_inv = 1.0 / point_ref->hessian_new_;
    H_sc += point_ref->hessian_fd_new_ * point_ref->hessian_fd_new_.transpose() * Hdd_inv;
    b_sc += point_ref->hessian_fd_new_ * point_ref->b_new_ * Hdd_inv;

    ++n_good;
  }

  // 判断snap，然后根据snap的结果维护正规方程
  if (!snap)
  {
    float information = level_idepth_points.size() * options_->alpha_w_;
    H_d.block<3, 3>(0, 0) += Eigen::Matrix3d::Identity() * information;
    b_d.block<3, 1>(0, 0) += tji.cast<double>() * information;
  }
  energy_tji_norm += options_->alpha_w_ * tji.squaredNorm();

  H = H_d.cast<float>();
  b = b_d.cast<float>();
  Hsc = H_sc.cast<float>();
  bsc = b_sc.cast<float>();

  Vec4f result;
  result << energy_photo, energy_idepth_norm, energy_tji_norm, static_cast<float>(n_good);
  return result;
}

/**
 * @brief 初始化器的 Options 的yaml文件配置
 *
 * @param filepath 输入的yaml配置文件
 */
Initializer2::Options::Options(const std::string &filepath)
{
  if (!std::filesystem::exists(filepath))
    throw std::runtime_error("配置文件不存在");

  auto info = YAML::LoadFile(filepath);
  pyra_levels_ = info["PyraidLevelsUsed"].as<int>();
  neighbor_nums_ = info["NeighborNum"].as<int>();
  select_densities_ = info["SelectDensities"].as<std::vector<float>>();
  max_iterations_ = info["MaxIterations"].as<std::vector<int>>();
  reg_weight_ = info["RegWeight"].as<float>();
  verbose_ = info["Verbose"].as<bool>();
  continue_snap_times_ = info["ContinusSnapTimes"].as<int>();
  huber_threshold_ = info["HuberThreshold"].as<float>();
  outlier_threshold_ = info["OutlierThreshold"].as<float>();
  inlier_ratio_ = info["InlierRatio"].as<float>();
  alpha_w_ = info["AlphaW"].as<float>();
  alpha_ = info["Alpha"].as<float>();
  tji_threshold_ = info["TransjiThreshold"].as<float>();
  make_sense_ratio_ = info["MakeSenseRatio"].as<float>();
  require_points_num_ = info["RequirePointsNum"].as<int>();
  immature_points_num_ = info["ImmaturePointsNum"].as<int>();
}

/**
 * @brief 当优化完成后，使用下层点投影到上层点 nlevel -> nlevel + 1
 *
 * 1. 时机：某个普通帧，所有的金字塔层优化完成后
 * 2. 从上到下进行点的投影时，使用高斯归一化乘积进行操作
 * 3. 当子点被判断为make_sense的时，父点会被判断为make_sense
 * 4. 注意，需要在投影到上一层后，进行IR更新
 */
void Initializer2::PropagateUp()
{
  auto pixel_points_curr = init_idepth_points_[0];
  std::vector<InitIdepthPoint::SharedPtr> pixel_points_prev;

  std::vector<bool> make_sense_curr(pixel_points_curr.size(), false); // 当前层的逆深度是否make sense
  std::vector<bool> make_sense_prev;                                  // 上一层的逆深度是否make sense

  std::vector<float> idepth_with_hessian;
  std::vector<float> idepth_hessian;
  std::vector<int> sense_child_nums;

  int nlevel = 0;

  // 使用第0层的优化状态点，初始化make_sense_curr
  auto pixel_points_layer0_process = [&](const int &idx)
  {
    if (!pixel_points_curr[idx]->is_good_)
      return;

    make_sense_curr[idx] = true;
    assert(pixel_points_curr[idx]->only_photo_hessian_new_ != 0);
  };

  // 计算 idepth_with_hessian idepth_hessian make_sense_prev
  auto compute_idepth_and_hessian_process = [&](const int &idx)
  {
    const auto &parent_point = pixel_points_prev[idx];
    for (const int &child_idx: parent_point->children_ids_)
    {
      if (!make_sense_curr[child_idx])
        continue;

      ++sense_child_nums[idx];
      idepth_hessian[idx] += pixel_points_curr[child_idx]->only_photo_hessian_new_;
      idepth_with_hessian[idx] +=
              pixel_points_curr[child_idx]->idepth_new_ * pixel_points_curr[child_idx]->only_photo_hessian_new_;
    }

    if (sense_child_nums[idx])
      make_sense_prev[idx] = true;
  };

  // 使用 idepth_with_hessian idepth_hessian make_sense_prev 设置 parent 的 idepth 和 hessian
  auto set_parent_idepth_process = [&](const int &idx)
  {
    if (!make_sense_prev[idx])
      return;

    auto &parent_point = pixel_points_prev[idx];
    auto this_idepth_with_hessian = idepth_with_hessian[idx];
    auto this_idepth_hessian = idepth_hessian[idx];

    // 由于向上投影时，是否make_sense仅考虑下层是否make_sense, 避免出现hessian为0,这里造一个hessian出来
    parent_point->idepth_new_ = this_idepth_with_hessian / (this_idepth_hessian + 1e-8);
    if (!parent_point->only_photo_hessian_new_)
      parent_point->only_photo_hessian_new_ = this_idepth_hessian / sense_child_nums[idx];

    assert(this_idepth_hessian != 0);
  };

  // 上一层点的ir处理
  auto update_ir_process = [&](const int &idx)
  {
    if (make_sense_prev[idx])
      UpdateIR(idx, nlevel + 1, make_sense_prev);
  };

  std::vector<int> indices_curr(pixel_points_curr.size(), 0);
  std::vector<int> indices_prev;
  std::iota(indices_curr.begin(), indices_curr.end(), 0);
  std::for_each(std::execution::par, indices_curr.begin(), indices_curr.end(), pixel_points_layer0_process);

  for (; nlevel < options_->pyra_levels_ - 1; ++nlevel)
  {
    pixel_points_prev = init_idepth_points_[nlevel + 1];

    make_sense_prev = std::vector<bool>(pixel_points_prev.size(), false);
    idepth_with_hessian = std::vector<float>(pixel_points_prev.size(), 0);
    idepth_hessian = std::vector<float>(pixel_points_prev.size(), 0);
    sense_child_nums = std::vector<int>(pixel_points_prev.size(), 0);

    indices_prev = std::vector<int>(pixel_points_prev.size(), 0);
    std::iota(indices_prev.begin(), indices_prev.end(), 0);

    std::for_each(indices_prev.begin(), indices_prev.end(), compute_idepth_and_hessian_process);
    std::for_each(indices_prev.begin(), indices_prev.end(), set_parent_idepth_process);
    std::for_each(indices_prev.begin(), indices_prev.end(), update_ir_process);

    std::swap(pixel_points_curr, pixel_points_prev);
    std::swap(make_sense_curr, make_sense_prev);
    std::swap(indices_prev, indices_curr);
  }
}

/**
 * @brief 将上层点投影到下层来
 *
 * @param nlevel nlevel层投影到 nlevel - 1 层
 *
 * 1. 可以初始化下层点的部分逆深度信息
 * 2. 将被判断为是内点，且参与过优化的逆深度投影进行投影继承
 * 3. 只有当snap为true时，PropagateDown才有意义
 */
void Initializer2::PropagateDown(const int &nlevel)
{
  if (nlevel < 1)
    throw std::runtime_error("nlevel error");

  auto parent_pixel_points = init_idepth_points_[nlevel];
  auto children_pixel_points = init_idepth_points_[nlevel - 1];

  std::vector<bool> make_sense(children_pixel_points.size(), false);
  std::vector<int> indices(children_pixel_points.size(), 0);
  std::iota(indices.begin(), indices.end(), 0);

  // 父点向子点投影过程
  auto child_point_process = [&](const int &idx)
  {
    // 继承状态信息
    auto &child_point = children_pixel_points[idx];
    auto &parent_point = parent_pixel_points[child_point->parent_id_];

    // 认为在优化最后被判断为内点
    if (!parent_point->is_good_)
      return;

    // 继承逆深度信息，更相信自己一点
    child_point->idepth_new_ = 2 * child_point->only_photo_hessian_new_ * child_point->idepth_new_ +
                               parent_point->only_photo_hessian_new_ * parent_point->idepth_new_;
    float new_hessian = 2 * child_point->only_photo_hessian_new_ + parent_point->only_photo_hessian_new_;
    child_point->idepth_new_ /= (new_hessian + 1e-8);
    make_sense[idx] = true;

    assert(new_hessian != 0);
  };

  // 当前层点的ir处理
  auto update_ir_process = [&](const int &idx)
  {
    if (make_sense[idx])
      UpdateIR(idx, nlevel - 1, make_sense);
  };

  // 这时仅仅使用parent 的内容，而不是修改，因此不存在数据竞争
  std::for_each(std::execution::par, indices.begin(), indices.end(), child_point_process);
  std::for_each(std::execution::par, indices.begin(), indices.end(), update_ir_process);
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
void Initializer2::ApplyStep(const int &lvl, const Vec8f &inc, const float &lambda)
{
  // 状态保存，以便backup
  Tji_ = Tji_new_;
  aji_ = aji_new_;
  bji_ = bji_new_;

  // Tji、aji、bji的更新
  Tji_new_ = Sophus::SE3f::exp(inc.head<6>()) * Tji_new_;
  aji_new_ += inc[6];
  bji_new_ += inc[7];

  // idepth的更新
  auto &lvl_idepth_points = init_idepth_points_[lvl];
  for (auto &lvl_point: lvl_idepth_points)
  {
    if (!lvl_point->is_good_)
      continue;

    // 在applyStep时，做状态copy，以便backup
    lvl_point->b_ = lvl_point->b_new_;
    lvl_point->only_photo_hessian_ = lvl_point->only_photo_hessian_new_;
    lvl_point->hessian_ = lvl_point->hessian_new_;
    lvl_point->idepth_ = lvl_point->idepth_new_;
    lvl_point->energy_ = lvl_point->energy_new_;
    lvl_point->hessian_fd_ = lvl_point->hessian_fd_new_;

    float bd = lvl_point->b_new_;
    float Hdd = lvl_point->hessian_new_;

    bd += lvl_point->hessian_fd_new_.cast<float>().dot(inc);
    float delta_idepth = -bd / ((1 + lambda) * Hdd);
    lvl_point->idepth_new_ += delta_idepth;
    lvl_point->is_update_ = true;
  }
}

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
void Initializer2::UpdateIR(const int &idx, const int &level, const std::vector<bool> &make_sense)
{
  std::vector<float> iRnn;
  auto &level_points = init_idepth_points_[level];
  auto &level_point = level_points[idx];

  for (const auto &inner_idx: level_point->neighbor_ids_)
  {
    if (!make_sense[inner_idx])
      continue;

    iRnn.push_back(level_points[inner_idx]->idepth_new_);
  }

  if (!iRnn.empty())
  {
    std::nth_element(iRnn.begin(), iRnn.begin() + iRnn.size() / 2, iRnn.end());
    level_point->iR_ =
            (1 - options_->reg_weight_) * level_point->idepth_new_ + options_->reg_weight_ * iRnn[iRnn.size() / 2];
    return;
  }

  level_point->iR_ = level_point->idepth_new_;
}

/**
 * @brief 初始化器的后处理过程
 *
 * 1. 根据初始化器参考帧，构建关键帧0
 * 2. 根据跟踪最后帧，构建关键帧1
 * 3. 根据初始化器的优化点，构建关键帧0的地图点（逆深度，host为关键帧0）
 * 4. 提取参考关键帧1的未成熟地图点，以便后续跟踪过程对未成熟点的更新
 */
void Initializer2::PostProcess()
{
  std::vector<InitIdepthPoint::SharedPtr> init_points = init_idepth_points_[0];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(init_points.begin(), init_points.end(), gen);

  std::vector<InitIdepthPoint::SharedPtr> keyframe_points;
  for (int idx = 0; idx < init_points.size(); ++idx)
  {
    if (!init_points[idx]->is_good_)
      continue;

    keyframe_points.push_back(init_points[idx]);
    if (keyframe_points.size() == options_->require_points_num_)
      break;
  }

  if (keyframe_points.size() != options_->require_points_num_)
    std::cerr << "Post Process have not enough points" << std::endl;

  // 认定当前参考帧是世界坐标系
  float exposure_ti = reference_frame_->GetExposureTime();
  if (exposure_ti < 0)
    exposure_ti = 1;

  reference_frame_->SetEstimate(SE3f(), 0.f, 0.f, exposure_ti, 0.f, 0.f);
  current_frame_->SetEstimate(Tji_, aji_, bji_, exposure_ti, 0, 0);

  keyframe0_ = std::make_shared<KeyFrame>(reference_frame_, keyframe_points);
  keyframe1_ = std::make_shared<KeyFrame>(current_frame_, std::vector<InitIdepthPoint::SharedPtr>());

  PixelSelector::Vector2iArray selected_points;
  std::vector<int> selected_points_conf;
  pixel_selector_->SelectFirstLayer(options_->immature_points_num_, keyframe1_->GetSqureGrad(), selected_points,
                                    selected_points_conf);

  std::vector<ImmaturePoint::SharedPtr> immature_points;
  for (int idx = 0; idx < selected_points.size(); ++idx)
  {
    auto immature_ptr = std::make_shared<ImmaturePoint>(selected_points[idx].cast<float>(), selected_points_conf[idx]);
    immature_points.push_back(immature_ptr);
  }

  keyframe1_->SetImmaturePoints(immature_points);
}

} // namespace dso_ssl
