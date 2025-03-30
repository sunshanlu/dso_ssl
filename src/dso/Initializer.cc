#include <execution>
#include <filesystem>
#include <mutex>
#include <tabulate/table.hpp>

#include "dso/Initializer.hpp"

namespace dso_ssl
{

/**
 * @brief 根据输入的参考帧信息，构建初始化器的图像金字塔信息
 *
 * 1. 构建相邻层的父子点关系，以便使用高斯归一化积进行逆深度传播
 * 2. 构建同一层的相邻关系，以便考虑相邻点之间的逆深度关系
 *
 * @param frame_ptr 输入的参考帧
 */
void Initializer::SetReferenceFrame(Frame::SharedPtr frame_ptr)
{
  if (ref_frame_)
    return;

  if (!frame_ptr)
    throw std::runtime_error("SetReferenceFrame: frame_ptr is nullptr");

  if (!pattern_)
    throw std::runtime_error("SetReferenceFrame: pattern_ is nullptr");

  if (!pixel_selector_)
    throw std::runtime_error("SetReferenceFrame: pixel_selector_ is nullptr");

  auto image_and_grads = frame_ptr->GetPyrdImageAndGrads();
  auto squre_grad = frame_ptr->GetSqureGrad();

  ref_layer_info_.clear();
  ref_layer_info_.resize(config_->pyra_levels_);
  std::vector<pcl::KdTreeFLANN<PointT>::Ptr> pyra_kdtrees(config_->pyra_levels_, nullptr);
  for (int nlevel = 0; nlevel < config_->pyra_levels_; ++nlevel)
  {
    // 维护layer_frame的层级信息
    LayerFrame::SharedPtr layer_frame = std::make_shared<LayerFrame>();
    layer_frame->nlevel_ = nlevel;

    // 维护layer_frame的图像和梯度信息
    std::vector<cv::Mat> image_and_grads_vec;
    cv::split(image_and_grads[nlevel], image_and_grads_vec);
    layer_frame->layer_image_ = image_and_grads_vec[0];
    layer_frame->layer_gradx_ = image_and_grads_vec[1];
    layer_frame->layer_grady_ = image_and_grads_vec[2];

    // 像素选择器提取像素位置
    int select_num = config_->select_densities_[nlevel] * image_and_grads[nlevel].rows * image_and_grads[nlevel].cols;
    PixelSelector::Vector2iArray selected_points;
    if (nlevel == 0)
    {
      std::vector<int> selected_points_conf;
      pixel_selector_->SelectFirstLayer(select_num, squre_grad, selected_points, selected_points_conf);
    }
    else
      pixel_selector_->SelectOtherLayer(select_num, image_and_grads[nlevel], selected_points);

    // 由于pattern的作用，需要进一步处理selected_points里面不合法的像素点
    decltype(selected_points) temp;
    auto half_pattern_size = pattern_->GetHalfPatternSize();
    for (const auto &point : selected_points)
    {
      if (point[0] > half_pattern_size && point[0] < image_and_grads[nlevel].cols - 4 && point[1] > half_pattern_size &&
          point[1] < image_and_grads[nlevel].rows - 4)
        temp.push_back(point);
    }
    std::swap(selected_points, temp);
    temp.clear();

    // 维护金字塔层级中的像素点信息，相邻关系和父子关系
    BuildNeighborRelation(layer_frame, selected_points, pyra_kdtrees[nlevel]);
    if (nlevel > 0)
      BuildParentChildRelation(layer_frame, ref_layer_info_[nlevel - 1], pyra_kdtrees[nlevel]);

    ref_layer_info_[nlevel] = layer_frame;
  }

  ref_frame_ = frame_ptr;
}

/**
 * @brief 构建同层金字塔提取点之间的相邻关系
 *
 * 0. 维护LayerFrame::pixel_points_
 * 1. 构建pcl的KD-Tree结构，维护当前点的信息
 * 2. 然后遍历点，试图找到相邻点，并维护相邻关系
 *
 * @param layer_kdtree      输出的金字塔层级的kdtree信息
 * @param layer_frame       输入输出的金字塔层级信息
 * @param selected_points   输入的选择点信息
 */
void Initializer::BuildNeighborRelation(LayerFrame::SharedPtr &layer_frame, const PixelSelector::Vector2iArray &selected_points, KdTree2d::Ptr &layer_kdtree)
{
  CloudT::Ptr layer_cloud = pcl::make_shared<CloudT>();

  layer_cloud->resize(selected_points.size());
  layer_frame->pixel_points_.resize(selected_points.size(), nullptr);
  std::vector<int> indices(selected_points.size());
  std::iota(indices.begin(), indices.end(), 0);

  // 构建点云layer_cloud，并填充layer_frame的像素信息
  auto point_process = [&](const int &idx)
  {
    PointT p;
    p.x = static_cast<float>(selected_points[idx][0]);
    p.y = static_cast<float>(selected_points[idx][1]);
    layer_cloud->points[idx] = p;

    auto pixel_point = std::make_shared<PixelPoint>(config_->reg_weight_);
    pixel_point->id_ = idx;
    pixel_point->nlevel_ = layer_frame->nlevel_;
    pixel_point->pixel_position_ = selected_points[idx].cast<float>();
    pixel_point->layer_frame_ = layer_frame.get();
    layer_frame->pixel_points_[idx] = pixel_point;
  };

  // 构建和当前层config_->neighbor_nums_个最近临点之间的关系
  auto neighbor_process = [&](const int &idx)
  {
    std::vector<float> dists;
    std::vector<int> neightbor_idx;
    if (layer_kdtree->nearestKSearch(layer_cloud->points[idx], config_->neighbor_nums_ + 1, neightbor_idx, dists) != config_->neighbor_nums_ + 1)
      throw std::runtime_error("neighbor search error");

    for (int i = 1; i < config_->neighbor_nums_ + 1; ++i)
      layer_frame->pixel_points_[idx]->neighbor_ids_.push_back(neightbor_idx[i]);
  };

  // 并行处理构建点云，并维护kdtree信息
  std::for_each(std::execution::par, indices.begin(), indices.end(), point_process);
  layer_kdtree = pcl::make_shared<KdTree2d>();
  layer_kdtree->setInputCloud(layer_cloud);

  // 寻找邻居点
  std::for_each(std::execution::par, indices.begin(), indices.end(), neighbor_process);
}

/**
 * @brief 构建相邻层金字塔提取点之间的父子关系
 *
 * 1. 将 last_layer_frame 上的点，投影到 cur_layer_frame 中
 * 2. 根据 cur_kdtrees 和 投影点 在cur_layer_frame找到最近点
 * 3. 维护父子关系
 *
 * @param cur_layer_frame       输入输出的当前层金字塔层级信息
 * @param last_layer_frame      输入输出的上一层金字塔层级信息
 * @param cur_kdtrees           输入的当前层金字塔层级的kdtree信息
 */
void Initializer::BuildParentChildRelation(LayerFrame::SharedPtr &cur_layer_frame, LayerFrame::SharedPtr &last_layer_frame, const KdTree2d::Ptr &cur_kdtrees)
{
  int last_points_num = last_layer_frame->pixel_points_.size();
  int curr_points_num = cur_layer_frame->pixel_points_.size();

  std::vector<int> indices(last_points_num, 0);
  std::iota(indices.begin(), indices.end(), 0);
  std::vector<std::mutex> cur_mutexes(curr_points_num);

  auto parent_child_process = [&](const int &idx)
  {
    // 将 last_layer_frame 上的点，投影到 cur_layer_frame 中
    PointT pc;
    auto &pl = last_layer_frame->pixel_points_[idx];
    pc.x = pl->pixel_position_[0] / 2 - 0.5;
    pc.y = pl->pixel_position_[1] / 2 - 0.5;

    // 搜索最近邻居点
    std::vector<int> k_indices;
    std::vector<float> k_distances;
    if (cur_kdtrees->nearestKSearch(pc, 1, k_indices, k_distances) != 1)
      throw std::runtime_error("kdtree search error");

    // 维护父子关系
    pl->parent_id_ = k_indices[0];
    {
      std::lock_guard<std::mutex> lock(cur_mutexes[k_indices[0]]);
      cur_layer_frame->pixel_points_[k_indices[0]]->children_ids_.push_back(idx);
    }
  };

  std::for_each(std::execution::par, indices.begin(), indices.end(), parent_child_process);
}

/**
 * @brief 初始化器配置构造
 *
 * @param config_path 输入的初始化器配置文件路径
 */
Initializer::Config::Config(const std::string &config_path)
{
  if (!std::filesystem::exists(config_path))
    throw std::runtime_error("配置文件不存在");

  auto info = YAML::LoadFile(config_path);
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
}

/**
 * @brief 根据参考关键帧，构造优化需要的信息，预先分配空间，可重复使用
 *
 * 1. 相对位姿顶点
 * 2. 相对仿射顶点
 * 3. 逆深度顶点
 * 4. 光度残差边
 * 5. 逆深度惩罚项为1.0的边
 * 6. 逆深度惩罚项为IR的边
 * 7. tji残差边
 *
 * @param ref_layer_frame 输入的参考帧上金字塔层上的信息
 */
LayerOptimizer::LayerOptimizer(LayerFrame::SharedPtr ref_layer_frame, int nlevel, float fx, float fy, float cx, float cy, Pattern::SharedPtr pattern,
                               const Initializer::Config::SharedPtr config)
    : nlevel_(std::move(nlevel))
    , fx_(std::move(fx))
    , fy_(std::move(fy))
    , cx_(std::move(cx))
    , cy_(std::move(cy))
    , pattern_(std::move(pattern))
    , huber_threshold_(config->huber_threshold_)
    , outlier_threshold_(config->outlier_threshold_)
    , inlier_ratio_(config->inlier_ratio_)
    , alpha_w_(config->alpha_w_)
    , alpha_(config->alpha_)
    , tji_threshold_(config->tji_threshold_)
    , make_sense_ratio_(config->make_sense_ratio_)
    , max_iterations_(config->max_iterations_[nlevel_])
    , accumulate_iterations_(0)
    , last_actual_iterations_(0)
{
  auto block_solver = std::make_unique<BlockSolver>(std::make_unique<LinearSolver>());
  algorithm_ = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
  graph_optimizer_.setAlgorithm(algorithm_);
  graph_optimizer_.setVerbose(false);

  // 相对位姿顶点
  relative_pose_ = new LeftSE3Vertex();
  relative_pose_->setId(all_vertexes_.size());
  all_vertexes_.push_back(relative_pose_);

  // 相对仿射参数
  relative_affine_ = new InitAffineVertex();
  relative_affine_->setId(all_vertexes_.size());
  all_vertexes_.push_back(relative_affine_);

  // 逆深度顶点 和 边信息
  idepth_vertexes_ = decltype(idepth_vertexes_)(ref_layer_frame->pixel_points_.size(), nullptr);
  photo_residuals_ = decltype(photo_residuals_)(ref_layer_frame->pixel_points_.size(), nullptr);

  // g2o中使用的huber与正常的huber不太一样，需要重新计算
  float huber_threshold = std::sqrt(pattern_->GetPattern().size() * huber_threshold_ * huber_threshold_);
  for (int i = 0; i < ref_layer_frame->pixel_points_.size(); ++i)
  {
    // 逆深度顶点
    auto pixel_point = ref_layer_frame->pixel_points_[i];
    auto idepth_vertex = new IdepthVertex(pixel_point);
    idepth_vertex->setId(all_vertexes_.size());
    idepth_vertex->setMarginalized(true);
    idepth_vertexes_[i] = idepth_vertex;
    all_vertexes_.push_back(idepth_vertex);

    // photo光度误差 对应的 残差
    auto photo_residual = new InitializerPhotoResidual(pattern_, &ref_layer_frame->layer_image_, outlier_threshold_, fx_, fy_, cx_, cy_);
    auto robust_kernel = new g2o::RobustKernelHuber();
    robust_kernel->setDelta(huber_threshold);
    photo_residual->setId(all_edges_.size());
    photo_residual->setVertex(0, relative_pose_);
    photo_residual->setVertex(1, idepth_vertex);
    photo_residual->setVertex(2, relative_affine_);
    photo_residual->setRobustKernel(robust_kernel);
    photo_residuals_[i] = photo_residual;
    all_edges_.push_back(photo_residual);

    // 逆深度为1.0惩罚项残差
    auto idepth_norm_res = new IdepthNormResidual();
    idepth_norm_res->setId(all_edges_.size());
    idepth_norm_res->setVertex(0, idepth_vertex);
    idepth_norm_res->setMeasurement(1.0);
    all_edges_.push_back(idepth_norm_res);
    idepth_norm_one_edges_.push_back(idepth_norm_res);

    // 逆深度为IR的惩罚项残差
    auto idepth_norm_ir_res = new IdepthNormIRResidual();
    idepth_norm_ir_res->setId(all_edges_.size());
    idepth_norm_ir_res->setVertex(0, idepth_vertex);
    all_edges_.push_back(idepth_norm_ir_res);
    idepth_norm_ir_edges_.push_back(idepth_norm_ir_res);
  }

  // tji 惩罚项
  tji_edge_ = new TjiNormResidual();
  tji_edge_->setId(all_edges_.size());
  tji_edge_->setVertex(0, relative_pose_);
  all_edges_.push_back(tji_edge_);

  // 向 graph_optimizer中添加所有边和顶点
  for (auto &vertex : all_vertexes_)
    graph_optimizer_.addVertex(vertex);
  for (auto &edge : all_edges_)
    graph_optimizer_.addEdge(edge);
}

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
bool LayerOptimizer::Optimize(Frame::SharedPtr frame_ptr, std::vector<PixelPoint::SharedPtr> &pixel_points, SE3f &Tji_new, float &aji_new, float &bji_new,
                              bool &meet_tji, bool verbose)
{
  // 根据输入的普通帧，更新顶点的估计初值和边的观测内容
  UpdateEstimateAndMeasurement(frame_ptr, pixel_points, Tji_new, aji_new, bji_new);
  UpdateEdgesLevel(meet_tji);

  // 仅改变残差状态，不进行error计算，可充分利用lm方法的最后一次尝试computeError
  auto photo_rest_process_not_error = [&](const int &idx)
  {
    if (photo_residuals_[idx]->outlier())
    {
      photo_residuals_[idx]->setLevel(1);
      idepth_norm_ir_edges_[idx]->setLevel(1);
      idepth_norm_one_edges_[idx]->setLevel(1);
    }
    else
    {
      photo_residuals_[idx]->setLevel(0);
      meet_tji ? idepth_norm_ir_edges_[idx]->setLevel(0) : idepth_norm_one_edges_[idx]->setLevel(0);
    }
  };

  // 将残差的计算次数重置为1，也就是说下次计算computeError时直接跳过
  auto reset_compute_times_to1 = [&](const int &idx) { photo_residuals_[idx]->ResetComputeTimes(1); };
  auto reset_compute_times_to0 = [&](const int &idx) { photo_residuals_[idx]->ResetComputeTimes(0); };

  std::vector<int> indices(photo_residuals_.size());
  std::iota(indices.begin(), indices.end(), 0);

  int inlier_residuals_before = 0;
  float system_energy_before = 0.0;
  std::for_each(std::execution::par, indices.begin(), indices.end(), reset_compute_times_to0);
  for (auto photo_residual : photo_residuals_)
  {
    photo_residual->computeError();

    // 当verbose 为true 时，统计优化前的平均残差值，仅 photo res
    if (verbose && !photo_residual->outlier())
    {
      Eigen::Vector3d rho;
      photo_residual->robustKernel()->robustify(photo_residual->chi2(), rho);
      system_energy_before += rho[0];
      ++inlier_residuals_before;
    }
  }

  std::for_each(std::execution::par, indices.begin(), indices.end(), photo_rest_process_not_error);

  int actual_iterations = 0;
  auto start_time = std::chrono::high_resolution_clock::now();
  for (int iteration = 0; iteration < max_iterations_; ++iteration)
  {
    // g2o的lm方法会在内部做10次调整lambda的尝试
    graph_optimizer_.initializeOptimization(0);
    graph_optimizer_.optimize(1);
    relative_pose_->ComputeAccept();
    ++actual_iterations;

    // 如果10次尝试都失败，则认定优化已经收敛
    if (!relative_pose_->Accept())
      break;

    // 产生 meet_tji 由 false 到 true 的切换，将edge的level进行切换
    if (relative_pose_->estimate().translation().norm() > tji_threshold_ && !meet_tji)
    {
      meet_tji = true;
      UpdateEdgesLevel(meet_tji);
    }

    // 结合lm方法最后一次判断的computeError，重置computeError的计算次数为1, 充分利用lm方法的computeError
    if (!relative_pose_->Accept())
      std::for_each(std::execution::par, indices.begin(), indices.end(),
                    [&](const int &idx)
                    {
                      // 光度残差被判断为外点，则将有关这个逆深度的所有边都视为1 level
                      photo_residuals_[idx]->computeError();
                    });

    std::for_each(std::execution::par, indices.begin(), indices.end(), photo_rest_process_not_error);
    std::for_each(std::execution::par, indices.begin(), indices.end(), reset_compute_times_to1);

    // 当满足tji条件时，要求那些在最新状态下被判断为inlier,且逆深度产生更新的点进行IR更新
    std::for_each(std::execution::par, idepth_vertexes_.begin(), idepth_vertexes_.end(), [&](IdepthVertex *vertex) { vertex->ComputeAccept(); });
    if (meet_tji)
    {
      for (int idx = 0; idx < photo_residuals_.size(); ++idx)
      {
        // 如果vertex_接受了上述条件，那么这时IR需要被更新，只判断Accept即可，代表了这次优化是否被接受
        bool last_optimized = idepth_vertexes_[idx]->Accept();

        if (photo_residuals_[idx]->outlier() || !last_optimized)
          continue;

        pixel_points[idx]->UpdateIR(photo_residuals_);
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  // condition 1 iteration not accept
  if (!relative_pose_->Accept())
    std::for_each(std::execution::par, indices.begin(), indices.end(),
                  [&](const int &idx)
                  {
                    // 光度残差被判断为外点，则将有关这个逆深度的所有边都视为1 level
                    photo_residuals_[idx]->computeError();
                  });

  // condition 2 iteration max，lm中的computeError有意义
  int nok = UpdatePixelPointsStatus(pixel_points, actual_iterations);
  if (static_cast<float>(nok) / pixel_points.size() < inlier_ratio_)
    return false;

  // meet_tji 由 true -> false 只能在优化完成后进行判断
  if (relative_pose_->estimate().translation().norm() < tji_threshold_)
    meet_tji = false;

  if (meet_tji)
    ComputeIdepthHessian(pixel_points, actual_iterations);
  SetEstimateParams(Tji_new, aji_new, bji_new, pixel_points, meet_tji, actual_iterations);

  // 定义优化输出的内容 level MeanResBefore MeanResAfter MakeSensePoints ActualIterations MeetSnap
  if (verbose)
  {
    float system_energy = 0;
    int inliner_residuals = 0;
    for (auto photo_residual : photo_residuals_)
    {
      Eigen::Vector3d rho;

      if (photo_residual->outlier())
        continue;

      photo_residual->robustKernel()->robustify(photo_residual->chi2(), rho);
      system_energy += static_cast<float>(rho[0]);
      ++inliner_residuals;
    }

    result_details_.level_ = nlevel_;
    result_details_.mean_res_before_ = system_energy_before == 0 ? 0 : std::sqrt(system_energy_before / inlier_residuals_before);
    result_details_.mean_res_after_ = system_energy == 0 ? 0 : std::sqrt(system_energy / inliner_residuals);
    result_details_.make_sense_points_ = nok;
    result_details_.actual_iterations_ = actual_iterations;
    result_details_.meet_snap_ = meet_tji;
    result_details_.duration_ms_ = static_cast<int>(duration.count());
  }

  return true;
}

/**
 * @brief 设置优化后的估计参数
 *
 * 该函数依托于pixel point 的逆深度hessian信息，当逆深度hessian信息存在时，才会将当前估计
 *
 * @param Tji               输出的Tji
 * @param aji               输出的aji
 * @param bji               输出的bji
 * @param pixel_points      输出的逆深度信息
 * @param meet_tji          是否符合tji要求
 * @param actual_iteratios  实际迭代次数
 */
void LayerOptimizer::SetEstimateParams(SE3f &Tji, float &aji, float &bji, std::vector<PixelPoint::SharedPtr> &pixel_points, const bool &meet_tji,
                                       const int &actual_iteratios)
{
  accumulate_iterations_ += actual_iteratios;
  last_actual_iterations_ = actual_iteratios;
  Tji = relative_pose_->estimate().cast<float>();

  if (meet_tji)
  {
    aji = relative_affine_->estimate()[0];
    bji = relative_affine_->estimate()[1];

    for (int idx = 0; idx < pixel_points.size(); ++idx)
    {
      auto &pixel_point = pixel_points[idx];
      bool meet_optimized = pixel_point->idepth_vertex_->AcceptTimes() / (float)actual_iteratios > make_sense_ratio_;
      if (meet_optimized && pixel_point->status_ == PixelPoint::Status::OK)
      {
        pixel_point->idepth_ = idepth_vertexes_[idx]->estimate();
      }
    }
    return;
  }

  Tji.translation().setZero();
  for (int idx = 0; idx < pixel_points.size(); ++idx)
  {
    auto &pixel_point = pixel_points[idx];
    pixel_point->SetOrigin();
  }
}

/**
 * @brief 在优化被接受后，考虑点的深度连通性，更新逆深度期望值
 *
 * 需要超参数reg_weight_的作用，reg_weight_用来表示周围逆深度的置信度
 * 在优化过程中，估计值还没有放到pixel point 里面时，更新逆深度期望值
 *
 */
void PixelPoint::UpdateIR(const std::vector<InitializerPhotoResidual *> photo_residuals)
{
  auto &pixel_points = layer_frame_->pixel_points_;
  float idepth_estimate = idepth_vertex_->estimate();
  std::vector<float> iRnn;
  for (const auto &idx : neighbor_ids_)
  {
    // 在判断接受优化后，内点和最新接受优化的点被判断为make sense点
    bool last_optimized = pixel_points[idx]->idepth_vertex_->Accept();
    if (photo_residuals[idx]->outlier() || !last_optimized)
      continue;

    const auto &idepth = layer_frame_->pixel_points_[idx]->idepth_vertex_->estimate();
    iRnn.push_back(idepth);
  }

  // 将指定位置放入某个元素，使该位置前的元素都小于它，该位置后的元素都大于它
  if (!iRnn.empty())
  {
    std::nth_element(iRnn.begin(), iRnn.begin() + iRnn.size() / 2, iRnn.end());
    idepth_avg_ = (1 - reg_weight_) * idepth_estimate + reg_weight_ * iRnn[iRnn.size() / 2];
    return;
  }

  idepth_avg_ = idepth_estimate;
}

/**
 * @brief 在优化过程后，上下投影过程中，更新逆深度期望值
 *
 * @param make_sense 与this 当前层的所有pixel point 是否有意义
 *
 * make_sense 代表着其为内点的条件下，其还参与过优化
 *
 * 1. 在PropagateDown 或者 PropagateUp 两个函数中使用
 * 2. 在优化过程完成后使用，非优化过程中
 */
void PixelPoint::UpdateIR(const std::vector<bool> &make_sense)
{
  std::vector<float> iRnn;
  for (const auto &idx : neighbor_ids_)
  {
    if (!make_sense[idx])
      continue;

    iRnn.push_back(layer_frame_->pixel_points_[idx]->idepth_);
  }

  if (!iRnn.empty())
  {
    std::nth_element(iRnn.begin(), iRnn.begin() + iRnn.size() / 2, iRnn.end());
    idepth_avg_ = (1 - reg_weight_) * idepth_ + reg_weight_ * iRnn[iRnn.size() / 2];
    return;
  }

  idepth_avg_ = idepth_;
}

/**
 * @brief 针对不同的frame_ptr，更新顶点状态的估计值和残差的测量值
 *
 * @param frame_ptr     输入的普通帧
 * @param pixel_points  输入的像素点逆深度估计
 * @param Tji_new       输入的Tji估计
 * @param aji_new       输入的aji估计
 * @param bji_new       输入的bji估计
 */
void LayerOptimizer::UpdateEstimateAndMeasurement(Frame::SharedPtr frame_ptr, const std::vector<PixelPoint::SharedPtr> &pixel_points, const SE3f &Tji_new,
                                                  const float &aji_new, const float &bji_new)
{
  // 拿出需要使用的变量
  static std::vector<cv::Mat> image_and_grads_j;
  image_and_grads_j.clear();

  cv::Mat frame_image_and_grads = frame_ptr->GetPyrdImageAndGrads()[nlevel_];
  cv::split(frame_image_and_grads, image_and_grads_j);
  cv::Mat &image_j = image_and_grads_j[0];
  cv::Mat &grads_x_j = image_and_grads_j[1];
  cv::Mat &grads_y_j = image_and_grads_j[2];

  // 重新设置lm 的 lambda 初始化为 0
  algorithm_->setUserLambdaInit(0);

  // 更新顶点部分的观测
  relative_pose_->setEstimate(Tji_new.cast<double>());
  relative_affine_->setEstimate(Vector2f(aji_new, bji_new).cast<double>());
  for (int idx = 0; idx < pixel_points.size(); ++idx)
  {
    idepth_vertexes_[idx]->accept_ = false;
    idepth_vertexes_[idx]->accepted_times_ = 0;
    idepth_vertexes_[idx]->setEstimate(pixel_points[idx]->idepth_);
  }

  // 更新光度误差边
  for (auto &photo_res : photo_residuals_)
    photo_res->UpdateParams(&image_j, &grads_x_j, &grads_y_j);

  // 更新逆深度惩罚边
  for (int idx = 0; idx < pixel_points.size(); ++idx)
  {
    idepth_norm_one_edges_[idx]->setInformation(alpha_w_ * Matrix1d::Ones());
    idepth_norm_ir_edges_[idx]->setInformation(alpha_ * Matrix1d::Ones());
  }

  // 更新tji惩罚边
  tji_edge_->setInformation(pixel_points.size() * alpha_w_ * Matrix3d::Ones());
}

/**
 * @brief 更新edge边对应的 level，控制优化图
 *
 * @param meet_tji 输入的是否满足tji条件
 */
void LayerOptimizer::UpdateEdgesLevel(const bool &meet_tji)
{
  if (meet_tji)
  {
    tji_edge_->setLevel(1);
    for (auto &edge : idepth_norm_ir_edges_)
      edge->setLevel(0);
    for (auto &edge : idepth_norm_one_edges_)
      edge->setLevel(1);

    return;
  }

  tji_edge_->setLevel(0);
  for (auto &edge : idepth_norm_one_edges_)
    edge->setLevel(0);
  for (auto &edge : idepth_norm_ir_edges_)
    edge->setLevel(1);
}

/**
 * @brief 计算逆深度的 hessian 矩阵
 *
 * 计算逆深度的hessian信息，考虑到存在OK状态下的点，但没有hessian信息，只需要计算一次jacobian即可
 * 这是因为，可能存在某些点，仅仅做了computeError的计算过程后得到了OK的状态，但是没有执行优化过程，
 * 因此没有hessian信息，所以在获取jacobian之前，需要执行一次 linearizeOplus
 */
void LayerOptimizer::ComputeIdepthHessian(std::vector<PixelPoint::SharedPtr> &pixel_points, const int &actual_iterations)
{
  auto idepth_hessian_process = [&](const int &idx)
  {
    auto &photo_edge = photo_residuals_[idx];

    // 下面对是inlier 并且 在当前优化过程中，参生优化的逆深度点求解hessian
    auto idepth_vertex = pixel_points[idx]->idepth_vertex_;
    bool meet_optimized = idepth_vertex->AcceptTimes() / (float)actual_iterations > make_sense_ratio_;
    if (pixel_points[idx]->status_ != PixelPoint::Status::OK || !meet_optimized)
      return;

    // 针对huber核函数部分，g2o里面使用的 整体近似 的情况，这里针对残差单独处理
    photo_edge->linearizeOplus();
    assert(photo_edge->GetResStatus() == InitializerPhotoResidual::Status::OK);
    Eigen::VectorXd residual = photo_edge->error();
    Eigen::MatrixXd huber_weight;
    Eigen::MatrixXd photo_idepth_jacobian = photo_edge->GetIdepthJacobian();
    huber_weight.resize(residual.rows(), residual.rows());
    huber_weight.setIdentity();
    for (int idx = 0; idx < residual.rows(); ++idx)
    {
      if (residual[idx] > huber_threshold_)
        huber_weight(idx, idx) = huber_threshold_ / std::abs(residual[idx]);
    }

    auto hessian_matrix = photo_idepth_jacobian.transpose() * huber_weight * photo_idepth_jacobian;
    if (hessian_matrix.rows() != 1 || hessian_matrix.cols() != 1)
      throw std::runtime_error("hessian matrix size error");

    pixel_points[idx]->idepth_hessian_ = hessian_matrix(0, 0);

    if (hessian_matrix(0, 0) == 0 && relative_pose_->estimate().translation().norm() != 0)
      throw std::runtime_error("hessian matrix is zero");
  };

  std::vector<int> indices(pixel_points.size());
  std::iota(indices.begin(), indices.end(), 0);

  // bug 这里使用并行化策略，理论上来讲不会出现数据竞争的情况，但是在tji != 0 的条件下，逆深度hessian总是会出现为0的情况！！
  // std::for_each(std::execution::par, indices.begin(), indices.end(), idepth_hessian_process);
  std::for_each(indices.begin(), indices.end(), idepth_hessian_process);
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
int LayerOptimizer::UpdatePixelPointsStatus(std::vector<PixelPoint::SharedPtr> &pixel_points, const int &actual_iterations)
{
  // 优化完成后，根据残差状态，统计逆深度点信息，使用原子量保证线程安全
  std::atomic_int nok = 0;
  auto process = [&](const int &idx)
  // for (int idx = 0; idx < photo_residuals_.size(); ++idx)
  {
    const auto &photo_residual = photo_residuals_[idx];
    const auto &res_status = photo_residual->GetResStatus();
    switch (res_status)
    {
    case InitializerPhotoResidual::Status::OOB:
      pixel_points[idx]->status_ = PixelPoint::Status::OOB;
      break;
    case InitializerPhotoResidual::Status::OUTLIER:
      pixel_points[idx]->status_ = PixelPoint::Status::OUTLIER;
      break;

    case InitializerPhotoResidual::Status::OK:
      pixel_points[idx]->status_ = PixelPoint::Status::OK;
      if (pixel_points[idx]->idepth_vertex_->AcceptTimes() / (float)actual_iterations > make_sense_ratio_)
        nok.fetch_add(1, std::memory_order_relaxed);

      break;

    default:
      throw std::runtime_error("unsupported InitializerPhotoResidual::Status");
      break;
    }
  };

  std::vector<int> indices(pixel_points.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::for_each(std::execution::par, indices.begin(), indices.end(), process);

  return nok.load();
}

/**
 * @brief 针对输入的帧，构建初始化器约束，进行优化
 *
 * 1. 当位移tcr 小于阈值时，使用额外的逆深度1约束和位移tcr的模约束，这里可以保证tcr的鲁棒性
 * 2. 当位移tcr 大于阈值时，使用使用idepth_avg_对逆深度进行约束
 * 3. 注意，当tcr优化没有满足阈值条件时，仅保留优化的旋转要求
 */
bool Initializer::Optimize(Frame::SharedPtr frame_ptr, SE3f &Tji, float &aji, float &bji, bool &meet_tji)
{
  // 从顶层向低层优化，coarse -> fine
  bool ok = false;
  int nlayer = config_->pyra_levels_ - 1;
  for (; nlayer >= 0; --nlayer)
  {
    auto &layer_frame = ref_layer_info_[nlayer];
    auto &pixel_points = layer_frame->pixel_points_;
    const int &max_iterations = config_->max_iterations_[nlayer];
    const bool &verbose = config_->verbose_;
    ok = layer_optimizers_[nlayer]->Optimize(frame_ptr, pixel_points, Tji, aji, bji, meet_tji, verbose);

    if (!ok)
      break;

    // 将优化成功(这次优化有意义的点)的上层点投影到下层去
    // 同时要求meet_tji为true，否则hessian不会被计算
    if (nlayer >= 1 && meet_tji)
      PropagateDown(nlayer);
  }

  // 当ok被提前退出时，需要将没有优化的位置标红
  if (config_->verbose_)
  {
    int make_sense_layer = ok ? 0 : nlayer + 1;
    OptimizationResultsShow(make_sense_layer);
  }

  // 金字塔优化完成后，从低层到顶层 逆深度投影上去
  // 同时要求meet_tji为true,否则heesian为0,没有计算意义
  if (ok && meet_tji)
  {
    PropagateUp();
    return true;
  }

  return false;
}

/**
 * @brief 展示优化结果
 *
 * @param make_sense_layer  有意义的优化的层级
 *
 * 1. level，优化层级
 * 2. MeanResBefore，优化前的绝对平均残差
 * 3. MeanResAfter，优化后的绝对平均残差
 * 4. MakeSensePoints，这此优化中，调整的逆深度数量
 * 5. ActualIterations，实际花费的迭代次数
 * 6. MeetSnap，优化后，是否满足snap要求
 * 7. Duriation, 优化耗时
 */
void Initializer::OptimizationResultsShow(const int &make_sense_layer, std::ostream &stream)
{
  using namespace tabulate;

  tabulate::Table optimization_result_details;
  optimization_result_details.format()
      .border_color(Color::magenta)
      .font_color(Color::green)
      .font_align(FontAlign::center)
      .font_style({FontStyle::bold, FontStyle::italic});

  optimization_result_details.add_row({"Level", "Mean Res Before", "Mean Res After", "Make Sense Number", "Actual Iterations", "Meet Snap", "Duration / ms"});
  optimization_result_details[0].format().font_color(Color::yellow);

  int row_idx = 1;

  for (int idx = config_->pyra_levels_ - 1; idx >= 0; --idx)
  {
    bool make_sense = idx >= make_sense_layer;

    auto layer_optimizer = layer_optimizers_[idx];
    auto layer_result_details = layer_optimizer->GetResultDetails();

    std::string level = std::to_string(idx);
    std::string mean_res_before = make_sense ? std::to_string(layer_result_details.mean_res_before_) : "N/A";
    std::string mean_res_after = make_sense ? std::to_string(layer_result_details.mean_res_after_) : "N/A";
    std::string make_sense_number = make_sense ? std::to_string(layer_result_details.make_sense_points_) : "N/A";
    std::string actual_iterations = make_sense ? std::to_string(layer_result_details.actual_iterations_) : "N/A";
    std::string meet_snap = make_sense ? (layer_result_details.meet_snap_ ? "Yes" : "No") : "N/A";
    std::string duration_time = make_sense ? std::to_string(layer_result_details.duration_ms_) : "N/A";

    optimization_result_details.add_row({level, mean_res_before, mean_res_after, make_sense_number, actual_iterations, meet_snap, duration_time});
    if (!make_sense)
      optimization_result_details[row_idx].format().background_color(Color::red).font_color(Color::white);

    ++row_idx;
  }

  stream << optimization_result_details << "\n" << std::endl;
}

/**
 * @brief 将上层点投影到下层来
 *
 * @param nlevel nlevel层投影到nlevel - 1层
 *
 * 1. 可以初始化下层点的部分逆深度信息
 * 2. 将被判断为是内点，且参与过优化的逆深度投影进行投影继承
 * 3. 只有当meet_tji为true时，PropagateDown才有意义
 */
void Initializer::PropagateDown(const int &nlevel)
{
  if (nlevel < 1)
    throw std::runtime_error("nlevel error");

  auto &curr_layer_frame = ref_layer_info_[nlevel];
  auto &last_layer_frame = ref_layer_info_[nlevel - 1];
  auto &parent_pixel_points = curr_layer_frame->pixel_points_;
  auto &children_pixel_points = last_layer_frame->pixel_points_;

  std::vector<bool> make_sense(children_pixel_points.size(), false);
  std::vector<int> indices(children_pixel_points.size(), 0);
  std::iota(indices.begin(), indices.end(), 0);

  // 父点向子点投影过程
  auto child_point_process = [&](const int &idx)
  {
    // 继承状态信息
    auto &child_point = children_pixel_points[idx];
    auto &parent_point = parent_pixel_points[child_point->parent_id_];

    // 认为在优化最后被判断为内点，且真实参与了优化内容
    const int &actual_iterations = layer_optimizers_[nlevel]->GetLastActualIterations();
    bool meet_optimized = parent_point->idepth_vertex_->AcceptTimes() / (float)actual_iterations > config_->make_sense_ratio_;
    if (parent_point->status_ != PixelPoint::Status::OK || !meet_optimized)
      return;

    // 继承逆深度信息，信息矩阵不改变，更相信自己一点
    child_point->idepth_ = 2 * child_point->idepth_hessian_ * child_point->idepth_ + parent_point->idepth_hessian_ * parent_point->idepth_;
    float new_hessian = 2 * child_point->idepth_hessian_ + parent_point->idepth_hessian_;
    child_point->idepth_ /= (new_hessian + 1e-8);
    make_sense[idx] = true;

    assert(new_hessian != 0);
  };

  // 当前层点的ir处理
  auto update_ir_process = [&](const int &idx)
  {
    auto &child_point = children_pixel_points[idx];
    if (make_sense[idx])
      child_point->UpdateIR(make_sense);
  };

  // 这时仅仅使用parent 的内容，而不是修改，因此不存在数据竞争
  std::for_each(std::execution::par, indices.begin(), indices.end(), child_point_process);
  std::for_each(std::execution::par, indices.begin(), indices.end(), update_ir_process);
}

/**
 * @brief 当优化完成后，使用下层点投影到上层点
 *
 * 1. 时机：某个普通帧，所有的金字塔层优化完成后
 * 2. 从上到下进行点的投影时，使用高斯归一化乘积进行操作
 * 3. 当子点被判断为make_sense的时，父点会被判断为make_sense
 * 4. 注意，需要在投影到上一层后，进行IR更新
 */
void Initializer::PropagateUp()
{
  LayerFrame::SharedPtr layer_frame_curr = ref_layer_info_[0];
  LayerFrame::SharedPtr layer_frame_prev;
  std::vector<PixelPoint::SharedPtr> pixel_points_curr = layer_frame_curr->pixel_points_;
  std::vector<PixelPoint::SharedPtr> pixel_points_prev;

  std::vector<bool> make_sense_curr(pixel_points_curr.size(), false); // 当前层的逆深度是否make sense
  std::vector<bool> make_sense_prev;                                  // 上一层的逆深度是否make sense

  std::vector<float> idepth_with_hessian;
  std::vector<float> idepth_hessian;
  std::vector<int> sense_child_nums;

  int nlevel = 0;
  // 使用第0层的优化状态点，初始化make_sense_curr
  auto pixel_points_layer0_process = [&](const int &idx)
  {
    auto &pixel_point = pixel_points_curr[idx];
    const int &actual_iterations = layer_optimizers_[nlevel]->GetLastActualIterations();
    bool meet_optimized = pixel_point->idepth_vertex_->AcceptTimes() / (float)actual_iterations > config_->make_sense_ratio_;

    if (pixel_point->status_ != PixelPoint::Status::OK || !meet_optimized)
      return;

    make_sense_curr[idx] = true;
  };

  // 计算 idepth_with_hessian idepth_hessian make_sense_prev
  auto compute_idepth_and_hessian_process = [&](const int &idx)
  {
    const auto &parent_point = pixel_points_prev[idx];
    for (const int &child_idx : parent_point->children_ids_)
    {
      if (!make_sense_curr[child_idx])
        continue;

      ++sense_child_nums[idx];
      idepth_hessian[idx] += pixel_points_curr[child_idx]->idepth_hessian_;
      idepth_with_hessian[idx] += pixel_points_curr[child_idx]->idepth_ * pixel_points_curr[child_idx]->idepth_hessian_;
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

    // 由于向上投影时，是否make_sense仅考虑下层是否make_sense,因此会出现hessian为0的情况（outlier->makesense）
    parent_point->idepth_ = this_idepth_with_hessian / (this_idepth_hessian + 1e-8);

    if (parent_point->idepth_hessian_ == 0)
      parent_point->idepth_hessian_ = this_idepth_hessian / sense_child_nums[idx];

    assert(this_idepth_hessian != 0);
    assert(parent_point->idepth_hessian_ != 0);
  };

  // 上一层点的ir处理
  auto update_ir_process = [&](const int &idx)
  {
    auto &parent_point = pixel_points_prev[idx];
    if (make_sense_prev[idx])
      parent_point->UpdateIR(make_sense_prev);
  };

  std::vector<int> indices_curr(pixel_points_curr.size(), 0);
  std::vector<int> indices_prev;
  std::iota(indices_curr.begin(), indices_curr.end(), 0);
  std::for_each(std::execution::par, indices_curr.begin(), indices_curr.end(), pixel_points_layer0_process);

  for (; nlevel < config_->pyra_levels_ - 1; ++nlevel)
  {
    layer_frame_prev = ref_layer_info_[nlevel + 1];
    pixel_points_prev = layer_frame_prev->pixel_points_;

    make_sense_prev = std::vector<bool>(pixel_points_prev.size(), false);
    idepth_with_hessian = std::vector<float>(pixel_points_prev.size(), 0);
    idepth_hessian = std::vector<float>(pixel_points_prev.size(), 0);
    sense_child_nums = std::vector<int>(pixel_points_prev.size(), 0);

    indices_prev = std::vector<int>(pixel_points_prev.size(), 0);
    std::iota(indices_prev.begin(), indices_prev.end(), 0);

    // std::for_each(std::execution::par, indices_prev.begin(), indices_prev.end(), compute_idepth_and_hessian_process);
    // std::for_each(std::execution::par, indices_prev.begin(), indices_prev.end(), set_parent_idepth_process);
    // std::for_each(std::execution::par, indices_prev.begin(), indices_prev.end(), update_ir_process);

    std::for_each(indices_prev.begin(), indices_prev.end(), compute_idepth_and_hessian_process);
    std::for_each(indices_prev.begin(), indices_prev.end(), set_parent_idepth_process);
    std::for_each(indices_prev.begin(), indices_prev.end(), update_ir_process);

    std::swap(layer_frame_curr, layer_frame_prev);
    std::swap(pixel_points_curr, pixel_points_prev);
    std::swap(make_sense_curr, make_sense_prev);
    std::swap(indices_prev, indices_curr);
  }
}

Initializer::Initializer(Config::SharedPtr config, PixelSelector::SharedPtr pixel_selector, Pattern::SharedPtr pattern, float fx, float fy, float cx, float cy)
    : config_(std::move(config))
    , pattern_(std::move(pattern))
    , initialized_(false)
    , ref_frame_(nullptr)
    , pixel_selector_(std::move(pixel_selector))
    , aji_estimate_(0.f)
    , bji_estimate_(0.f)
{
  for (int nlevel = 0; nlevel < config_->pyra_levels_; ++nlevel)
  {
    fx_.push_back(fx / std::pow(2, nlevel));
    fy_.push_back(fy / std::pow(2, nlevel));
    cx_.push_back(cx / std::pow(2, nlevel));
    cy_.push_back(cy / std::pow(2, nlevel));
  }
}

/**
 * @brief 向初始化器中添加帧，试图完成初始化过程
 *
 * 要求连续k帧，满足tji的阈值条件，才认为初始化成功
 *
 * @param frame_ptr 输入的帧
 * @return true     初始化成功
 * @return false    初始化失败
 */
bool Initializer::AddActivateFrame(Frame::SharedPtr frame_ptr)
{
  // ref frame 配置和层优化器配置
  if (!ref_frame_)
  {
    SetReferenceFrame(frame_ptr);

    layer_optimizers_.resize(config_->pyra_levels_, nullptr);
    for (int nlevel = 0; nlevel < config_->pyra_levels_; ++nlevel)
    {
      const float &lfx = fx_[nlevel];
      const float &lfy = fy_[nlevel];
      const float &lcx = cx_[nlevel];
      const float &lcy = cy_[nlevel];
      layer_optimizers_[nlevel] = std::make_shared<LayerOptimizer>(ref_layer_info_[nlevel], nlevel, lfx, lfy, lcx, lcy, pattern_, config_);
    }

    return false;
  }

  if (initialized_)
    return true;

  static int continus_snap = 0;

  // 当曝光时间存在时，倾向去重新构造aji和bji,而不是使用优化后的
  float tj = frame_ptr->GetExposureTime();
  float ti = ref_frame_->GetExposureTime();
  if (tj > 0 && ti > 0)
  {
    aji_estimate_ = std::log(tj / ti);
    bji_estimate_ = 0;
  }

  bool meet_tji = Tji_estimate_.translation().norm() > config_->tji_threshold_;
  bool ok = Optimize(frame_ptr, Tji_estimate_, aji_estimate_, bji_estimate_, meet_tji);
  if (ok && meet_tji)
    ++continus_snap;
  else
    continus_snap = 0;

  if (continus_snap >= config_->continue_snap_times_)
  {
    initialized_ = true;
    return true;
  }

  return false;
}

} // namespace dso_ssl
