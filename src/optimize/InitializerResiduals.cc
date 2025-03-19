#include "optimize/InitializerResiduals.hpp"

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
std::vector<float> InitializerPointResidual::ComputeErrorSSE(const std::vector<float> &Ii_data, const std::vector<float> &Ij_data, const __m128 &aji_sse,
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
 */
void InitializerPointResidual::computeError()
{
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

    auto simd_process = [&](const int &start)
    {
        // 投影过程，使用SSE加速，并且保存中间变量
        Vector2fArray points_i(4), points_j;
        for (int idx = start; idx < start + 4; ++idx)
            points_i[idx - start] = pattern_positions[idx] + idepth_->pixel_position_;

        Vector3fArray points_temp_j(4);

        project::Pixel2PixelSSE(Tji, points_i, idepth_value, fx_, fy_, cx_, cy_, points_temp_j, points_j);

        float uj[4] = {points_j[0][0], points_j[1][0], points_j[2][0], points_j[3][0]};
        float vj[4] = {points_j[0][1], points_j[1][1], points_j[2][1], points_j[3][1]};
        float ui[4] = {points_i[0][0], points_i[1][0], points_i[2][0], points_i[3][0]};
        float vi[4] = {points_i[0][1], points_i[1][1], points_i[2][1], points_i[3][1]};

        // 使用双线性差值，计算pi和pj点的灰度值，并计算残差项
        // auto Ii_result = interp::BilinInterpSSE(*image_j_, ui, vi);
        // i 帧并不需要双线性差值
        std::vector<float> Ii_result(4, 0);
        for (int idx = 0; idx < 4; ++idx)
        {
            int ui_int = static_cast<int>(ui[idx]);
            int vi_int = static_cast<int>(vi[idx]);
            Ii_result[idx] = image_i_->at<float>(vi_int, ui_int);
        }

        auto Ij_result = interp::BilinInterpSSE(*image_i_, uj, vj);
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
    _error.resize(pattern_positions.size());
    Ii_.resize(pattern_positions.size());
    parallel::ParallelWrapper(0, pattern_positions.size(), 4, simd_process, single_process);
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
void InitializerPointResidual::linearizeOplus()
{
    auto pattern_positions = pattern_->GetPattern();
    auto tji = relative_pose_->estimate().translation();

    // drk / dTji
    _jacobianOplus[0].resize(pattern_positions.size(), 6);
    _jacobianOplus[0].setZero();

    // drk / didepth
    _jacobianOplus[1].resize(pattern_positions.size(), 1);
    _jacobianOplus[1].setZero();

    // drk / affine_ji
    _jacobianOplus[2].resize(pattern_positions.size(), 2);
    _jacobianOplus[2].setZero();

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
        _jacobianOplus[1](idx, 0) = (dxfx * (tji[0] - pxpz * tji[2]) + dyfy * (tji[1] - pypz * tji[2])) / (temp_pjz + 1e-8);

        // drk / affine_ji
        _jacobianOplus[2](idx, 0) = Ii_[idx] * exp_aji_;
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

            Ii_data[idx - start] = Ii_[idx];
        }

        auto temp_pjz_sse = _mm_load_ps(temp_pjz_data);
        auto dpipz_sse = _mm_load_ps(dpipz_data);
        auto pxpz_sse = _mm_load_ps(pxpz_data);
        auto pypz_sse = _mm_load_ps(pypz_data);
        auto dxfx_sse = _mm_load_ps(dxfx_data);
        auto dyfy_sse = _mm_load_ps(dyfy_data);
        auto Ii_sse = _mm_load_ps(Ii_data);

        // clang-format off
        auto drkdTji0_sse = _mm_mul_ps(dxfx_sse, dpipz_sse);
        auto drkdTji1_sse = _mm_mul_ps(dyfy_sse, dpipz_sse);
        auto drkdTji2_sse = _mm_sub_ps(_mm_mul_ps(_mm_mul_ps(-dxfx_sse, dpipz_sse), pxpz_sse), _mm_mul_ps(_mm_mul_ps(dyfy_sse, dpipz_sse), pypz_sse));
        auto drkdTji3_sse = _mm_sub_ps(_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(zero_sse, dxfx_sse), pxpz_sse), pypz_sse),
                                       _mm_mul_ps(dyfy_sse, _mm_add_ps(one_sse, _mm_mul_ps(pypz_sse, pypz_sse))));
        auto drkdTji4_sse = _mm_add_ps(_mm_mul_ps(dxfx_sse, _mm_add_ps(one_sse, _mm_mul_ps(pxpz_sse, pxpz_sse))), 
                                       _mm_mul_ps(_mm_mul_ps(dyfy_sse, pxpz_sse), pypz_sse));
        auto drkdTji5_sse = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(zero_sse, dxfx_sse), pypz_sse), _mm_mul_ps(dyfy_sse, pxpz_sse));
        
        auto drkddpi_sse = _mm_div_ps(_mm_add_ps(_mm_mul_ps(dxfx_sse, _mm_sub_ps(tji_sse_x, _mm_mul_ps(pxpz_sse, tji_sse_z))),
                                                 (dyfy_sse * _mm_sub_ps(tji_sse_y, _mm_mul_ps(pypz_sse, tji_sse_z)))), temp_pjz_sse);

        // clang-format on
        auto drkdAffine0_sse = _mm_mul_ps(Ii_sse, exp_aji_sse);
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

            // drk / affine_ji
            _jacobianOplus[2](idx + start, 0) = drkdAffine0_result[idx];
            _jacobianOplus[2](idx + start, 1) = drkdAffine1_result[idx];
        }
    };

    parallel::ParallelWrapper(0, pattern_positions.size(), 4, simd_process, single_process);
}

} // namespace dso_ssl
