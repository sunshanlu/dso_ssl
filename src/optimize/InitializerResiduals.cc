#include "optimize/InitializerResiduals.hpp"
#include "dso/Initializer.hpp"

namespace dso_ssl
{
/**
 * @brief 在给定 Ii, Ij, exp_aji和bji的条件下，计算残差值，使用SSE加速
 *
 *
 * @param Ii_data 输入的Ii(pi)帧的灰度值
 * @param Ij_data 输入的Ij(pj)帧的灰度值
 * @param aji_sse 输入的exp_aji
 * @param bji_sse 输入的bji
 * @return std::vector<float> 输出的残差值
 */
std::vector<float> InitializerPhotoResidual::ComputeErrorSSE(const std::vector<float> &Ii_data, const std::vector<float> &Ij_data, const __m128 &aji_sse,
                                                             const __m128 &bji_sse)
{
  if (Ii_data.size() != 4 || Ij_data.size() != 4)
    throw std::runtime_error("InitiallizerResidual::ComputeErrorSSE: Ii_data or Ij_data size is not 4");
  alignas(16) float Ii[4], Ij[4], error_data[4];

  for (int idx = 0; idx < 4; ++idx)
  {
    Ii[idx] = Ii_data[idx];
    Ij[idx] = Ij_data[idx];
  }

  __m128 Ii_sse = _mm_load_ps(Ii);
  __m128 Ij_sse = _mm_load_ps(Ij);

  __m128 error_sse = _mm_sub_ps(_mm_sub_ps(Ij_sse, _mm_mul_ps(aji_sse, Ii_sse)), bji_sse);
  _mm_store_ps(error_data, error_sse);

  std::vector<float> error_result(4);
  for (int idx = 0; idx < 4; ++idx)
    error_result[idx] = error_data[idx];

  return error_result;
}

/**
 * @brief 计算初始化残差，定义多顶点的顺序
 *
 * _vertices[0] 相对位姿
 * _vertices[1] 逆深度
 * _vertices[2] 相对初始化仿射参数 aji和bji
 *
 * 考虑和lm方法配置一起使用，手动调用第一次，当lm调用第二次的时，不用进行任何操作
 * 但是需要注意，优化一轮lm后，需要手动的重置这个计算次数
 */
void InitializerPhotoResidual::computeError()
{
  // 当计算第二次时，代表lm方法第一次调用computeError，直接跳过，防止冗余调用
  ++compute_times_;

  if (compute_times_ == 2)
    return;

  Eigen::VectorXd error_temp(_error);
  is_outlier_ = false;
  temp_j_.clear();
  grads_xj_.clear();
  grads_yj_.clear();
  Ii_.clear();

  relative_pose_ = dynamic_cast<LeftSE3Vertex *>(_vertices[0]);
  idepth_ = dynamic_cast<IdepthVertex *>(_vertices[1]);
  affine_ = dynamic_cast<InitAffineVertex *>(_vertices[2]);

  const auto &affine_value = affine_->estimate();

  const auto exp_aji = exp(affine_value[0]);
  exp_aji_ = exp_aji;
  const auto &bji = affine_value[1];
  __m128 exp_aji_sse = _mm_set1_ps(exp_aji);
  __m128 bji_sse = _mm_set1_ps(bji);

  auto pattern_positions = pattern_->GetPattern();
  Sophus::SE3f Tji = relative_pose_->estimate().cast<float>();
  float idepth_value = static_cast<float>(idepth_->estimate());

  std::vector<bool> is_outlier(pattern_->GetPattern().size(), false);

  auto simd_process = [&](const int &start)
  {
    // 投影过程，使用SSE加速，并且保存中间变量
    Vector2fArray points_i(4), points_j;
    for (int idx = start; idx < start + 4; ++idx)
      points_i[idx - start] = pattern_positions[idx] + idepth_->pixel_position_;

    Vector3fArray points_temp_j(4);

    project::Pixel2PixelSSE(Tji, points_i, idepth_value, fx_, fy_, cx_, cy_, points_temp_j, points_j);

    // 对pj进行检查，pj点是否能投影到j帧上
    bool have_outlier = false;
    for (int idx = 0; idx < 4; ++idx)
    {
      is_outlier[start + idx] = points_j[idx][0] < 0 || points_j[idx][0] > image_j_->cols - 1 || points_j[idx][1] < 0 || points_j[idx][1] > image_j_->rows - 1;
      if (is_outlier[start + idx])
        have_outlier = true;
    }

    if (have_outlier)
      return;

    float uj[4] = {points_j[0][0], points_j[1][0], points_j[2][0], points_j[3][0]};
    float vj[4] = {points_j[0][1], points_j[1][1], points_j[2][1], points_j[3][1]};
    float ui[4] = {points_i[0][0], points_i[1][0], points_i[2][0], points_i[3][0]};
    float vi[4] = {points_i[0][1], points_i[1][1], points_i[2][1], points_i[3][1]};

    // 使用双线性差值，计算pi和pj点的灰度值，并计算残差项
    // auto Ii_result = interp::BilinInterpSSE(*image_i_, ui, vi);
    // i 帧并不需要双线性差值
    std::vector<float> Ii_result(4, 0);
    for (int idx = 0; idx < 4; ++idx)
    {
      int ui_int = static_cast<int>(ui[idx]);
      int vi_int = static_cast<int>(vi[idx]);
      Ii_result[idx] = image_i_->at<float>(vi_int, ui_int);
    }

    auto Ij_result = interp::BilinInterpSSE(*image_j_, uj, vj);
    auto grad_xj = interp::BilinInterpSSE(*grads_x_, uj, vj);
    auto grad_yj = interp::BilinInterpSSE(*grads_y_, uj, vj);
    auto error_result = ComputeErrorSSE(Ii_result, Ij_result, exp_aji_sse, bji_sse);

    for (int idx = 0; idx < 4; ++idx)
    {
      temp_j_[start + idx] = points_temp_j[idx];
      _error[start + idx] = error_result[idx];
      grads_xj_[start + idx] = grad_xj[idx];
      grads_yj_[start + idx] = grad_yj[idx];
      Ii_[start + idx] = Ii_result[idx];
    }
  };

  auto single_process = [&](const int &idx)
  {
    Vector2f point_i = pattern_positions[idx] + idepth_->pixel_position_, point_j;

    // pi -> pj 的投影过程
    project::Pixel2Pixel(Tji, point_i, idepth_value, fx_, fy_, cx_, cy_, temp_j_[idx], point_j);

    if (point_j[0] < 0 || point_j[0] > image_j_->cols - 1 || point_j[1] < 0 || point_j[1] > image_j_->rows - 1)
    {
      is_outlier[idx] = true;
      return;
    }

    // 计算残差项，float -> int 可以相互转换
    float Ii = image_i_->at<float>(point_i[1], point_i[1]);
    float Ij = interp::BilinInterp(*image_j_, point_j[0], point_j[1]);
    float grad_xj = interp::BilinInterp(*grads_x_, point_j[0], point_j[1]);
    float grad_yj = interp::BilinInterp(*grads_y_, point_j[0], point_j[1]);
    _error[idx] = ComputeError(Ii, Ij, exp_aji, bji);
    grads_xj_[idx] = grad_xj;
    grads_yj_[idx] = grad_yj;
    Ii_[idx] = Ii;
  };

  // 在实际执行之前，改变存储项的长度
  temp_j_.resize(pattern_positions.size());
  grads_xj_.resize(pattern_positions.size());
  grads_yj_.resize(pattern_positions.size());
  Ii_.resize(pattern_positions.size());
  
  // todo 为了方便调试，先注释掉多线程实现
  // parallel::ParallelWrapper(0, pattern_positions.size(), 4, simd_process, single_process);
  int start = 0;
  for (; start < pattern_positions.size() - 3; start += 4)
    simd_process(start);

  for (int idx = start; idx < pattern_positions.size(); ++idx)
    single_process(idx);

  Eigen::Vector3d rho;
  double energy = chi2();
  if (_robustKernel)
  {
    _robustKernel->robustify(energy, rho);
    energy = rho[0];
  }

  // 残差大于阈值外点，错的比较离谱
  if (energy > outlier_threshold_)
  {
    is_outlier_ = true;
    res_status_ = Status::OUTLIER;
    return;
  }

  // 因为是并行执行，因此，无法直接判断外点然后停止，被遮挡的外点
  for (int idx = 0; idx < pattern_positions.size(); ++idx)
  {

    if (is_outlier[idx])
    {
      is_outlier_ = true;
      _error = error_temp;
      res_status_ = Status::OOB;
      return;
    }
  }

  res_status_ = Status::OK;
}

/**
 * @brief 计算残差相对的雅可比矩阵
 *
 * 在计算雅可比过程中，需要提前存储一些变量来降低计算量，降低时间复杂度
 *  1. Pj' 计算pj像素位置的中间变量
 *  2. grads_xj_ pj 像素位置的x方向的梯度
 *  3. grads_yj_ pj 像素位置的y方向的梯度
 *  4. Ii_ 存储pi像素位置的灰度值
 *  5. exp_aji_ aji的指数值
 */
void InitializerPhotoResidual::linearizeOplus()
{
  auto pattern_positions = pattern_->GetPattern();
  auto tji = relative_pose_->estimate().translation();

  // drk / dTji
  _jacobianOplus[0].setZero();

  // drk / didepth
  _jacobianOplus[1].setZero();

  // drk / affine_ji
  _jacobianOplus[2].setZero();

  // 当在computeError中判断为外点时，这里雅可比矩阵设置为0,不会提供任何约束
  if (is_outlier_)
  {
    assert(_level == 1);
    return;
  }

  std::vector<double> maxstep(pattern_positions.size(), 0);

  auto single_process = [&](const int &idx)
  {
    const double temp_pjx = static_cast<double>(temp_j_[idx][0]);
    const double temp_pjy = static_cast<double>(temp_j_[idx][1]);
    const double temp_pjz = static_cast<double>(temp_j_[idx][2]);

    const double dpipz = idepth_->estimate() / (temp_pjz + 1e-8);
    const double pxpz = temp_pjx / (temp_pjz + 1e-8);
    const double pypz = temp_pjy / (temp_pjz + 1e-8);

    const double dxfx = fx_ * grads_xj_[idx];
    const double dyfy = fy_ * grads_yj_[idx];

    // drk / dTji
    _jacobianOplus[0](idx, 0) = dxfx * dpipz;
    _jacobianOplus[0](idx, 1) = dyfy * dpipz;
    _jacobianOplus[0](idx, 2) = -dxfx * dpipz * pxpz - dyfy * dpipz * pypz;
    _jacobianOplus[0](idx, 3) = -dxfx * pxpz * pypz - dyfy * (1 + pypz * pypz);
    _jacobianOplus[0](idx, 4) = dxfx * (1 + pxpz * pxpz) + dyfy * pxpz * pypz;
    _jacobianOplus[0](idx, 5) = -dxfx * pypz + dyfy * pxpz;

    // drk / didepth
    float item1 = (tji[0] - pxpz * tji[2]);
    float item2 = (tji[1] - pypz * tji[2]);
    _jacobianOplus[1](idx, 0) = (dxfx * item1 + dyfy * item2) / (temp_pjz + 1e-8);
    maxstep[idx] = 0.25 * temp_pjz / (Vector2f(fx_ * item1, fy_ * item2).norm() + 1e-8);

    // drk / affine_ji
    _jacobianOplus[2](idx, 0) = -Ii_[idx] * exp_aji_;
    _jacobianOplus[2](idx, 1) = -1;
  };

  __m128 one_sse = _mm_set1_ps(1.0);
  __m128 zero_sse = _mm_set1_ps(0.0);
  __m128 tji_sse_x = _mm_set1_ps(tji[0]);
  __m128 tji_sse_y = _mm_set1_ps(tji[1]);
  __m128 tji_sse_z = _mm_set1_ps(tji[2]);
  __m128 exp_aji_sse = _mm_set1_ps(exp_aji_);

  auto simd_process = [&](const int &start)
  {
    alignas(16) float temp_pjx_data[4], temp_pjy_data[4], temp_pjz_data[4], Ii_data[4];
    alignas(16) float dpipz_data[4], pxpz_data[4], pypz_data[4], dxfx_data[4], dyfy_data[4];
    alignas(16) float dx_data[4], dy_data[4], maxstep_data[4];

    for (int idx = start; idx < start + 4; ++idx)
    {
      temp_pjx_data[idx - start] = temp_j_[idx][0];
      temp_pjy_data[idx - start] = temp_j_[idx][1];
      temp_pjz_data[idx - start] = temp_j_[idx][2] + 1e-8;

      dpipz_data[idx - start] = idepth_->estimate() / temp_pjz_data[idx - start];
      pxpz_data[idx - start] = temp_pjx_data[idx - start] / temp_pjz_data[idx - start];
      pypz_data[idx - start] = temp_pjy_data[idx - start] / temp_pjz_data[idx - start];

      dxfx_data[idx - start] = fx_ * grads_xj_[idx];
      dyfy_data[idx - start] = fy_ * grads_yj_[idx];
      dx_data[idx - start] = grads_xj_[idx];
      dy_data[idx - start] = grads_yj_[idx];

      Ii_data[idx - start] = Ii_[idx];
    }

    auto temp_pjz_sse = _mm_load_ps(temp_pjz_data);
    auto dpipz_sse = _mm_load_ps(dpipz_data);
    auto pxpz_sse = _mm_load_ps(pxpz_data);
    auto pypz_sse = _mm_load_ps(pypz_data);
    auto dxfx_sse = _mm_load_ps(dxfx_data);
    auto dyfy_sse = _mm_load_ps(dyfy_data);
    auto Ii_sse = _mm_load_ps(Ii_data);
    auto fx_sse = _mm_set1_ps(fx_);
    auto fy_sse = _mm_set1_ps(fy_);
    auto dx_sse = _mm_load_ps(dx_data);
    auto dy_sse = _mm_load_ps(dy_data);

    // clang-format off
    // drk / dTji
    auto drkdTji0_sse = _mm_mul_ps(dxfx_sse, dpipz_sse);
    auto drkdTji1_sse = _mm_mul_ps(dyfy_sse, dpipz_sse);
    auto drkdTji2_sse = _mm_sub_ps(_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(zero_sse, dxfx_sse), dpipz_sse), pxpz_sse), _mm_mul_ps(_mm_mul_ps(dyfy_sse, dpipz_sse), pypz_sse));
    auto drkdTji3_sse = _mm_sub_ps(_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(zero_sse, dxfx_sse), pxpz_sse), pypz_sse),
                                    _mm_mul_ps(dyfy_sse, _mm_add_ps(one_sse, _mm_mul_ps(pypz_sse, pypz_sse))));
    auto drkdTji4_sse = _mm_add_ps(_mm_mul_ps(dxfx_sse, _mm_add_ps(one_sse, _mm_mul_ps(pxpz_sse, pxpz_sse))), 
                                    _mm_mul_ps(_mm_mul_ps(dyfy_sse, pxpz_sse), pypz_sse));
    auto drkdTji5_sse = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(zero_sse, dxfx_sse), pypz_sse), _mm_mul_ps(dyfy_sse, pxpz_sse));
    
    // drk / ddpi
    auto item1_sse = _mm_mul_ps(fx_sse, _mm_sub_ps(tji_sse_x, _mm_mul_ps(pxpz_sse, tji_sse_z)));
    auto item2_sse = _mm_mul_ps(fy_sse, _mm_sub_ps(tji_sse_y, _mm_mul_ps(pypz_sse, tji_sse_z)));
    auto drkddpi_sse = _mm_div_ps(_mm_add_ps(_mm_mul_ps(dx_sse, item1_sse), _mm_mul_ps(dy_sse, item2_sse)), temp_pjz_sse);
    
    auto norm_sse = _mm_add_ps(_mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(item1_sse, item1_sse), _mm_mul_ps(item2_sse, item2_sse))), _mm_set1_ps(1e-8));
    auto maxstep_sse = _mm_div_ps(_mm_mul_ps(_mm_set1_ps(0.25) , temp_pjz_sse), norm_sse);

    // clang-format on
    // drk / daji bji
    auto drkdAffine0_sse = _mm_sub_ps(zero_sse, _mm_mul_ps(Ii_sse, exp_aji_sse));
    auto drkdAffine1_sse = _mm_set1_ps(-1);

    alignas(16) float drkdTji0_result[4], drkdTji1_result[4], drkdTji2_result[4], drkdTji3_result[4], drkdTji4_result[4];
    alignas(16) float drkdTji5_result[4], drkddpi_result[4], drkdAffine0_result[4], drkdAffine1_result[4];
    _mm_store_ps(drkdTji0_result, drkdTji0_sse);
    _mm_store_ps(drkdTji1_result, drkdTji1_sse);
    _mm_store_ps(drkdTji2_result, drkdTji2_sse);
    _mm_store_ps(drkdTji3_result, drkdTji3_sse);
    _mm_store_ps(drkdTji4_result, drkdTji4_sse);
    _mm_store_ps(drkdTji5_result, drkdTji5_sse);

    _mm_store_ps(drkddpi_result, drkddpi_sse);
    _mm_store_ps(maxstep_data, maxstep_sse);

    _mm_store_ps(drkdAffine0_result, drkdAffine0_sse);
    _mm_store_ps(drkdAffine1_result, drkdAffine1_sse);

    for (int idx = 0; idx < 4; ++idx)
    {
      // drk / dTji
      _jacobianOplus[0](idx + start, 0) = drkdTji0_result[idx];
      _jacobianOplus[0](idx + start, 1) = drkdTji1_result[idx];
      _jacobianOplus[0](idx + start, 2) = drkdTji2_result[idx];
      _jacobianOplus[0](idx + start, 3) = drkdTji3_result[idx];
      _jacobianOplus[0](idx + start, 4) = drkdTji4_result[idx];
      _jacobianOplus[0](idx + start, 5) = drkdTji5_result[idx];

      // drk / didepth
      _jacobianOplus[1](idx + start, 0) = drkddpi_result[idx];
      maxstep[start + idx] = maxstep_data[idx];

      // drk / affine_ji
      _jacobianOplus[2](idx + start, 0) = drkdAffine0_result[idx];
      _jacobianOplus[2](idx + start, 1) = drkdAffine1_result[idx];
    }
  };

  // todo 为了方便调试，先注释掉多线程实现
  // parallel::ParallelWrapper(0, pattern_positions.size(), 4, simd_process, single_process);
  int start = 0;
  for (; start < pattern_positions.size() - 3; start += 4)
    simd_process(start);

  for (int idx = start; idx < pattern_positions.size(); ++idx)
    single_process(idx);

  auto minmax = std::minmax_element(maxstep.begin(), maxstep.end());
  idepth_->SetMaxstep(*minmax.second);
}

/// 计算IR正则化残差
void IdepthNormIRResidual::computeError()
{
  idepth_vertex_ = dynamic_cast<IdepthVertex *>(_vertices[0]);
  _error[0] = idepth_vertex_->estimate() - idepth_vertex_->pixel_point_->idepth_avg_;
}

} // namespace dso_ssl
