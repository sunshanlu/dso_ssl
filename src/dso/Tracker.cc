#include "dso/Tracker.hpp"
#include "dso/PhotoAffine.hpp"
#include "utils/Project.hpp"

namespace dso_ssl
{

/**
 * @brief 将滑动窗口中的关键帧向最新关键帧进行投影，后端线程调用
 *
 * 1. 要求关键帧的逆深度点状态里面没有外点
 * 2. 使用的逆深度点时、Tcw等滑动窗口状态时，不能在逆深度更新过程中
 * 3. 由于使用四舍五入，因此存在多对一的情况，使用高斯归一化积
 *
 * @param sliding_window          输入的滑动窗口，用于逆深度点投影
 * @param ref_idepth_sum          输出的逆深度点高斯归一化积的分布情况，第0层
 * @param ref_hessian_sum         输出的逆深度点hessian的分布情况，第0层
 */
void Tracker::ProjectWindow2Ref(const std::vector<KeyFrame::SharedPtr> &sliding_window, cv::Mat &ref_idepth_sum, cv::Mat &ref_hessian_sum)
{
  cv::Mat square_grad = ref_keyframe_->GetSqureGrad();
  cv::Mat track_idpeth_points(square_grad.rows, square_grad.cols, CV_32F, 0.f);
  cv::Mat track_idepth_hessian(square_grad.rows, square_grad.cols, CV_32F, 0.f);

  const float &fx0 = track_fx_[0];
  const float &fy0 = track_fy_[0];
  const float &cx0 = track_cx_[0];
  const float &cy0 = track_cy_[0];

  Mat3f Ki;
  Ki << 1.f / fx0, 0.f, -cx0 / fx0, 0.f, 1.f / fy0, -cy0 / fy0, 0.f, 0.f, 1.f;

  for (int idx = 0; idx < sliding_window.size() - 1; ++idx)
  {
    auto keyframe = sliding_window[idx];
    auto map_points = keyframe->GetMapPoints();

    SE3f Trk = ref_keyframe_->GetTcw() * keyframe->GetTcw().inverse();
    Mat3f RKi = Trk.rotationMatrix() * Ki;
    Vec3f trk = Trk.translation();

    for (int point_idx = 0; point_idx < map_points.size(); ++point_idx)
    {
      Vec2f point_k = map_points[point_idx]->GetHostPixel();
      float idepth_k = map_points[point_idx]->GetIdepth();

      // 检查pk向跟踪参考关键帧投影时，逆深度是否合法，在初始化优化结束后检查
      assert(idepth_k > 0 && std::isfinite(idepth_k));

      Vec3f pr_temp = RKi * Vec3f(point_k[0], point_k[1], 1.f) + trk * idepth_k;
      float idepth_r = idepth_k / pr_temp[2];

      // 检查pr点逆深度的合法性
      assert(idepth_r > 0 && std::isfinite(idepth_r));

      float ur = pr_temp[0] / pr_temp[2], vr = pr_temp[1] / pr_temp[2];
      int kur = ur * fx0 + cx0 + 0.5;
      int kvr = vr * fy0 + cy0 + 0.5;

      if (kur < 0 || kur > square_grad.cols - 1 || kvr < 0 || kvr > square_grad.rows - 1)
        continue;

      float hessian = map_points[point_idx]->GetHessian();

      // 检查优化逆深度的hessian
      assert(std::isfinite(hessian) && hessian > 0);

      track_idpeth_points.at<float>(kvr, kur) += idepth_r * hessian;
      track_idepth_hessian.at<float>(kvr, kur) += hessian;
    }
  }

  ref_idepth_sum = track_idpeth_points;
  ref_hessian_sum = track_idepth_hessian;
}

/**
 * @brief 基于当前层的idpeth_sum和hessian_sum信息，向上一层进行投影
 *
 * @param curr_idepth_sum   输入的当前金字塔idepth_sum
 * @param curr_hessian_sum  输入的当前金字塔hessian_sum
 * @param prev_idepth_sum   输出的上一层金字塔idepth_sum
 * @param prev_hessian_sum  输出的上一层金字塔hessian_sum
 */
void Tracker::PropagateUp(const cv::Mat &curr_idepth_sum, const cv::Mat &curr_hessian_sum, cv::Mat &prev_idepth_sum, cv::Mat &prev_hessian_sum)
{
  if (curr_idepth_sum.rows % 2 != 0 || curr_idepth_sum.cols % 2 != 0)
    throw std::runtime_error("Tracker::PropagateUp: curr_idepth_sum must be even");

  int prev_rows = curr_idepth_sum.rows / 2, prev_cols = curr_idepth_sum.cols / 2;
  prev_idepth_sum = cv::Mat(prev_rows, prev_cols, CV_32F, 0.f);
  prev_hessian_sum = cv::Mat(prev_rows, prev_cols, CV_32F, 0.f);

  auto position_process = [&](const int &idx)
  {
    int row = idx / prev_cols;
    int col = idx % prev_cols;

    float idepth_sum0 = curr_idepth_sum.at<float>(2 * row, 2 * col);
    float idepth_sum1 = curr_idepth_sum.at<float>(2 * row, 2 * col + 1);
    float idepth_sum2 = curr_idepth_sum.at<float>(2 * row + 1, 2 * col);
    float idepth_sum3 = curr_idepth_sum.at<float>(2 * row + 1, 2 * col + 1);
    prev_idepth_sum.at<float>(row, col) += (idepth_sum0 + idepth_sum1 + idepth_sum2 + idepth_sum3);

    // 验证向上投影过程中是否出现非法逆深度
    assert(std::isfinite(idepth_sum0) && std::isfinite(idepth_sum1) && std::isfinite(idepth_sum2) && std::isfinite(idepth_sum3) && idepth_sum0 >= 0 &&
           idepth_sum1 >= 0 && idepth_sum2 >= 0 && idepth_sum3 >= 0);

    float hessian_sum0 = curr_hessian_sum.at<float>(2 * row, 2 * col);
    float hessian_sum1 = curr_hessian_sum.at<float>(2 * row, 2 * col + 1);
    float hessian_sum2 = curr_hessian_sum.at<float>(2 * row + 1, 2 * col);
    float hessian_sum3 = curr_hessian_sum.at<float>(2 * row + 1, 2 * col + 1);
    prev_hessian_sum.at<float>(row, col) += (hessian_sum0 + hessian_sum1 + hessian_sum2 + hessian_sum3);

    // 验证向上投影过程中是否出现非法hessian
    assert(std::isfinite(hessian_sum0) && std::isfinite(hessian_sum1) && std::isfinite(hessian_sum2) && std::isfinite(hessian_sum3) &&
           hessian_sum0 >= 0 && idepth_sum1 >= 0 && idepth_sum2 >= 0 && idepth_sum3 >= 0);
  };

  std::vector<int> indices(prev_rows * prev_cols);
  std::iota(indices.begin(), indices.end(), 0);
  std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), position_process);
}

/**
 * @brief 逆深度点的膨胀，用于在track中取代pattern
 *
 *  1. 0层和1层使用斜侧膨胀方式，对于多个逆深度情况，使用高斯归一化积
 *  2. 2层及以上使用上下左右膨胀方式，对于多个逆深度情况，使用高斯归一化积
 *
 * @note 为了防止边遍历边修改的情况发生，仅在input_xxx中遍历，在level_xxx中修改，最后交换即可
 *
 * @param level             输入金字塔的层数
 * @param input_idepth_sum  输入输出的金字塔上的逆深度高斯归一化积
 * @param input_hessian_sum 输入输出的金字塔上的hessian和
 */
void Tracker::IdpethExpansion(const int &level, cv::Mat &input_idepth_sum, cv::Mat &input_hessian_sum)
{
  static std::vector<Vec2i> pattern0 = {Vec2i(-1, -1), Vec2i(1, -1), Vec2i(-1, 1), Vec2i(1, 1)};
  static std::vector<Vec2i> pattern1 = {Vec2i(0, -1), Vec2i(-1, 0), Vec2i(0, 1), Vec2i(1, 0)};

  const int &rows = input_idepth_sum.rows;
  const int &cols = input_idepth_sum.cols;

  std::vector<Vec2i> pattern;
  if (level < 2)
    pattern = pattern0;
  else
    pattern = pattern1;

  cv::Mat level_idepth_sum, level_hessian_sum;
  input_idepth_sum.copyTo(level_idepth_sum);
  input_hessian_sum.copyTo(level_hessian_sum);

  auto position_process = [&](const int &idx)
  {
    int row = idx / cols;
    int col = idx % cols;
    const float &hessian = input_hessian_sum.at<float>(row, col);

    // 查看膨胀中，hessian是否合法
    assert(std::isfinite(hessian) && hessian >= 0);

    if (hessian > 0)
      return;

    for (int inner_idx = 0; inner_idx < 4; ++inner_idx)
    {
      const Vec2i &pattern_positon = pattern[inner_idx];

      int target_row = row + pattern_positon[1];
      int target_col = col + pattern_positon[0];

      if (target_row < 0 || target_row > rows - 1 || target_col < 0 || target_col > cols - 1)
        continue;

      float &hessian_around = input_hessian_sum.at<float>(target_row, target_col);
      float &idepth_around = input_idepth_sum.at<float>(target_row, target_col);
      const float &hessian_sum = level_hessian_sum.at<float>(row, col) += hessian_around;
      const float &idepth_sum = level_idepth_sum.at<float>(row, col) += idepth_around;

      assert(std::isfinite(hessian_sum) && hessian_sum >= 0); // 验证膨胀过程中hessian是否非法
      assert(std::isfinite(idepth_sum) && idepth_sum >= 0);   // 验证膨胀过程中逆深度是否非法
    }
  };

  std::vector<int> indices(rows * cols);
  std::iota(indices.begin(), indices.end(), 0);
  std::for_each(std::execution::par, indices.begin(), indices.end(), position_process);

  std::swap(level_idepth_sum, input_idepth_sum);
  std::swap(level_hessian_sum, input_hessian_sum);
}

/**
 * @brief 构建参考点，将构造的参考点放到 track_idepth_points中，后端线程调用
 *
 * 1. 将滑动窗口中的关键帧逆深度点，投影到最新的关键帧上，得到第0层逆深度点分布状态
 *  1.1 投影过程需要考虑四舍五入，因此存在多个投影点对应一个投影点的情况，使用高斯归一化积
 *  1.2 逆深度点，向上投影，得到金字塔层级上的逆深度点分布情况
 * 2. Tracker中弃用pattern,使用逆深度点膨胀的方式实现跟踪点的扩充
 *  2.1 0层和1层使用斜侧膨胀方式，对于多个逆深度情况，使用高斯归一化积
 *  2.2 2层及以上使用上下左右膨胀方式，对于多个逆深度情况，使用高斯归一化积
 *
 * @param sliding_window 输入的滑动窗口，用于向最新关键帧投影
 */
void Tracker::BuildTrackerPoints(const std::vector<KeyFrame::SharedPtr> &sliding_window)
{
  ref_keyframe_ = sliding_window.back();
  std::vector<cv::Mat> pyramid_idepth_sum(options_->pyra_levels_);
  std::vector<cv::Mat> pyramid_hessian_sum(options_->pyra_levels_);

  // 将滑动窗口中的地图点，投影到最新的关键帧上
  ProjectWindow2Ref(sliding_window, pyramid_idepth_sum[0], pyramid_hessian_sum[0]);

  // 向上一层进行逆深度点的投影
  for (int level = 0; level < options_->pyra_levels_ - 1; ++level)
  {
    cv::Mat &curr_idepth_sum = pyramid_idepth_sum[level];
    cv::Mat &curr_hessian_sum = pyramid_hessian_sum[level];
    cv::Mat &prev_idepth_sum = pyramid_idepth_sum[level + 1];
    cv::Mat &prev_hessian_sum = pyramid_hessian_sum[level + 1];

    PropagateUp(curr_idepth_sum, curr_hessian_sum, prev_idepth_sum, prev_hessian_sum);
  }

  for (int level = 0; level < options_->pyra_levels_; ++level)
  {
    cv::Mat &curr_idepth_sum = pyramid_idepth_sum[level];
    cv::Mat &curr_hessian_sum = pyramid_hessian_sum[level];
    IdpethExpansion(level, curr_idepth_sum, curr_hessian_sum);

    // 求解高斯归一化部分
    auto position_process = [&](const int &idx)
    {
      int row = idx / curr_idepth_sum.cols;
      int col = idx % curr_idepth_sum.cols;

      const float &hessian_sum = curr_hessian_sum.at<float>(row, col);
      const float &idepth_sum = curr_idepth_sum.at<float>(row, col);

      if (hessian_sum == 0)
        return;

      assert(std::isfinite(hessian_sum) && hessian_sum > 0);
      assert(std::isfinite(idepth_sum) && idepth_sum > 0);
      auto tracker_point = std::make_shared<TrackIdpethPoint>();
      tracker_point->ref_idepth_ = idepth_sum / hessian_sum;
      tracker_point->ref_pixel_point_ = Vec2f(col, row);
      track_idepth_points_[level].push_back(tracker_point);

      assert(std::isfinite(tracker_point->ref_idepth_) && std::isfinite(tracker_point->ref_idepth_ > 0));
    };

    std::vector<int> indices(curr_idepth_sum.rows * curr_idepth_sum.cols);
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(indices.begin(), indices.end(), position_process);
  }
}

/**
 * @brief 当后端优化完成后，更新Tracker跟踪器信息，以匹配最新优化状态，后端线程调用
 *
 * 1. 更新参考关键帧、滑动窗口状态等信息
 * 2. 更新相机的内参矩阵
 *
 * @note 由于需要修改Tracker线程中的状态，因此需要先加锁来保证线程安全
 *
 * @param sliding_window 输入的滑动窗口，用于向最新关键帧投影
 * @param fx             后端优化后的fx
 * @param fy             后端优化后的fy
 * @param cx             后端优化后的cx
 * @param cy             后端优化后的cy
 */
void Tracker::UpdateTracker(const std::vector<KeyFrame::SharedPtr> &sliding_window, const float &fx, const float &fy, const float &cx,
                            const float &cy)
{
  mapper_changed_.store(true);
  std::lock_guard<std::mutex> lock(tracker_mutex_);

  UpdateCalib(fx, fy, cx, cy);
  BuildTrackerPoints(sliding_window);
}

/**
 * @brief 更新Tracker跟踪器的内参矩阵，后端线程调用
 *
 * @param fx 输入的后端更新后的fx
 * @param fy 输入的后端更新后的fy
 * @param cx 输入的后端更新后的cx
 * @param cy 输入的后端更新后的cy
 */
void Tracker::UpdateCalib(const float &fx, const float &fy, const float &cx, const float &cy)
{
  track_fx_[0] = fx;
  track_fy_[0] = fy;
  track_cx_[0] = cx;
  track_cy_[0] = cy;

  for (int level = 1; level < options_->pyra_levels_; ++level)
  {
    track_fx_[level] = track_fx_[level - 1] / 2.f;
    track_fy_[level] = track_fy_[level - 1] / 2.f;
    track_cx_[level] = track_cx_[level - 1] / 2.f;
    track_cy_[level] = track_cy_[level - 1] / 2.f;
  }
}


void Tracker::Pixel2pixel(const SE3f &Tji, const TrackIdpethPoint::SharedPtr &pi, const int &level, Vec3f &p_temp, Vec2f &pj)
{
  project::Pixel2Pixel(Tji, pi->ref_pixel_point_, pi->ref_idepth_, track_fx_self_[level], track_fy_self_[level], track_cx_self_[level],
                       track_cy_self_[level], p_temp, pj);
}


/**
 * 计算jacobian和error误差，并构建正规方程，使用huber核函数
 *
 * 1. 获取level层的参考帧的逆深度点 lvl_pts
 * 2. 将lvl_pts上的点pi投影到当前帧pj点上（投影过程+差值过程）
 * 3. 使用Tji、ai、bi、aj、bj和上述得到的投影点和图像差值构建残差和hw
 * 4. 计算雅可比矩阵
 * 5. 根据hw、雅可比矩阵和残差构建正规方程
 *
 * @param level       输入的金字塔层级
 * @param Tji         输入的参考帧到当前帧的变换
 * @param ai          输入的参考帧的绝对仿射参数ai
 * @param bi          输入的参考帧的绝对仿射参数bi
 * @param aj          输入的当前帧的绝对仿射参数aj
 * @param bj          输入的当前帧的绝对仿射参数bj
 * @param H           输出的正规方程中的H矩阵
 * @param b           输出的正规方程中的b向量
 * @param adjust_flag 输出的是否调整了外点阈值
 * @return Vec2f  [energy_photo, ngood]
 */
Tracker::Vec2f Tracker::ComputeJacobianAndError(const int &level, const SE3f &Tji, const float &ai, const float &bi, const float &aj, const float &bj,
                                                Mat8f &H, Vec8f &b, bool &adjust_flag)
{
  double energy_photo = 0;
  Mat8d H_d = Mat8d::Zero();
  Vec8d b_d = Vec8d::Zero();

  float exposure_ti = ref_frame_state_.ti;
  float exposure_tj = curr_frame_->GetExposureTime();
  if (exposure_ti < 0 || exposure_tj < 0)
  {
    exposure_ti = 1.f;
    exposure_tj = 1.f;
  }

  const float &lfx = track_fx_self_[level], &lfy = track_fy_self_[level];
  const float &lcx = track_cx_self_[level], &lcy = track_cy_self_[level];
  float exp_aji = exposure_tj * std::exp(ai) / (exposure_ti * std::exp(aj));
  float bji = bj - exp_aji * bi;

  Mat3f Ki = Mat3f::Zero();
  Ki << 1.0 / lfx, 0, -lcx / lfx, 0, 1.0 / lfy, -lcy / lfy, 0, 0, 1;

  Mat3f RKi = Tji.rotationMatrix() * Ki;
  Vec3f tji = Tji.translation();

  cv::Mat ref_image_and_grad = ref_keyframe_self_->GetPyrdImageAndGrads()[level];
  cv::Mat cur_image_and_grad = curr_frame_->GetPyrdImageAndGrads()[level];

  // 1. 获取level层的参考帧的逆深度点 lvl_pts
  int ngood = 0;
  auto &lvl_pts = track_idepth_points_self_[level];

  std::vector<Vec3f, Eigen::aligned_allocator<Vec3f>> Ij_and_grad_buf(lvl_pts.size(), Vec3f::Zero());
  std::vector<float> idepth_pj_buf(lvl_pts.size(), 0);
  std::vector<float> uj_buf(lvl_pts.size(), 0);
  std::vector<float> vj_buf(lvl_pts.size(), 0);
  std::vector<float> Ii_buf(lvl_pts.size(), 0);
  std::vector<float> residual_buf(lvl_pts.size(), 0);
  std::vector<bool> is_inlier(lvl_pts.size(), true);

  for (int idx = 0; idx < lvl_pts.size(); ++idx)
  {
    const auto &pi = lvl_pts[idx]->ref_pixel_point_;
    const auto &dpi = lvl_pts[idx]->ref_idepth_;

    // 外点条件1：逆深度<=0
    if (dpi <= 0)
    {
      is_inlier[idx] = false;
      continue;
    }

    // 2. 将lvl_pts上的点pi投影到当前帧pj点上（投影过程+差值过程）
    Vec3f pj_temp = RKi * Vec3f(pi[0], pi[1], 1) + tji * dpi;
    float idepth_pj = dpi / pj_temp[2];
    float uj = pj_temp[0] / pj_temp[2], vj = pj_temp[1] / pj_temp[2];
    float kuj = lfx * uj + lcx, kvj = lfy * vj + lcy;

    // 外点条件2：当投影的pj点逃出边界
    if (kuj < 1 || kuj > ref_image_and_grad.cols - 2 || kvj < 1 || kvj > ref_image_and_grad.rows - 2)
    {
      is_inlier[idx] = false;
      continue;
    }

    const float &Ii = ref_image_and_grad.at<cv::Vec3f>(pi[1], pi[0])[0];
    Vec3f Ij_and_grad = interp::BilinInterp3(cur_image_and_grad, kuj, kvj);

    // 3. 使用Tji、ai、bi、aj、bj和上述得到的投影点和图像差值构建残差和hw
    float residual = Ij_and_grad[0] - exp_aji * Ii - bji;

    Ij_and_grad_buf[idx] = Ij_and_grad;
    idepth_pj_buf[idx] = idepth_pj;
    uj_buf[idx] = uj;
    vj_buf[idx] = vj;
    Ii_buf[idx] = Ii;
    residual_buf[idx] = residual;

    if (std::abs(residual) > options_->outlier_threshold_)
      continue;

    ++ngood;
  }

  bool adjust_threshold = false;
  int adjust_threshold_factor = 1;
  while (static_cast<float>(ngood) / lvl_pts.size() < 0.6 && adjust_threshold_factor <= 32)
  {
    ngood = 0;

    if (!adjust_threshold)
      adjust_threshold = true;

    adjust_threshold_factor *= 2;
    float outlier_threshold = options_->outlier_threshold_ * adjust_threshold_factor;
    for (int idx = 0; idx < lvl_pts.size(); ++idx)
    {
      if (!is_inlier[idx])
        continue;

      if (std::abs(residual_buf[idx]) > outlier_threshold)
        continue;

      ++ngood;
    }
  }

  // 这种情况明显为错误的猜测值
  if (adjust_threshold_factor > 32)
    return Vec2f(0, 0);

  // 4.计算雅可比矩阵
  float outlier_threshold = options_->outlier_threshold_ * adjust_threshold_factor;
  for (int idx = 0; idx < is_inlier.size(); ++idx)
  {
    if (!is_inlier[idx])
      continue;

    if (std::abs(residual_buf[idx]) > outlier_threshold)
      continue;


    const auto &Ij_and_grad = Ij_and_grad_buf[idx];
    const auto &residual = residual_buf[idx];
    const auto &idepth_pj = idepth_pj_buf[idx];
    const auto &uj = uj_buf[idx];
    const auto &vj = vj_buf[idx];
    const auto &Ii = Ii_buf[idx];

    Vec8d Jacobian;
    float dxfx = Ij_and_grad[1] * lfx;
    float dyfy = Ij_and_grad[2] * lfy;

    Jacobian[0] = dxfx * idepth_pj;
    Jacobian[1] = dyfy * idepth_pj;
    Jacobian[2] = -(dxfx * uj * idepth_pj + dyfy * vj * idepth_pj);
    Jacobian[3] = -dxfx * uj * vj - dyfy * (1 + vj * vj);
    Jacobian[4] = dxfx * (1 + uj * uj) + dyfy * uj * vj;
    Jacobian[5] = -dxfx * vj + dyfy * uj;
    Jacobian[6] = exp_aji * (Ii - bi);
    Jacobian[7] = -1;

    // 5.根据hw、雅可比矩阵和残差构建正规方程
    float hw = std::abs(residual) <= options_->huber_threshold_ ? 1 : options_->huber_threshold_ / std::abs(residual);
    energy_photo += hw * residual * residual * (2 - hw);
    H_d += hw * Jacobian * Jacobian.transpose();
    b_d += hw * Jacobian * residual;
  }

  H = H_d.cast<float>();
  b = b_d.cast<float>();
  adjust_flag = adjust_threshold;
  return Vec2f(energy_photo, ngood);
}


/**
 * 根据Tji,尝试一次优化
 *
 * 1. 从粗到精遍历金字塔层级
 * 2. 使用lm方法，进行优化
 *
 * 优化失败触发条件，可以做到提前退出
 *  1. ComputeJacobianAndError函数中，内点数量始终不能大于0.6，则认定失败
 *  2. 层级优化完成后，如果发现优化的RMSE结果，大于1.5倍的中断abort_res，则认定失败
 *
 * @param Tji_estimate  输入输出的Tji，参考帧和普通帧之间的位姿变换
 * @param aj_estimate   输入输出的aj，当前帧的绝对仿射参数aj
 * @param bj_estimate   输入输出的bj，当前帧的绝对仿射参数bj
 * @param ai            输入的ai，参考帧绝对仿射参数
 * @param bi            输入的bi，参考帧绝对仿射参数bj
 * @param abort_res     输入的各层级中优化的RMSE中断阈值
 * @return true         优化成功
 * @return false        优化失败
 */
bool Tracker::TryOnce(SE3f &Tji_estimate, float &aj_estimate, float &bj_estimate, const float &ai, const float &bi,
                      const std::vector<float> &abort_res)
{
  static float eps = 1e-4, min_lambda = 1e-4, max_lambda = 1e4;
  static int max_fails_times = 2;

  // 1. 从粗到精遍历金字塔层级，并在金字塔层级上进行优化
  std::vector<bool> have_repeated(options_->pyra_levels_, false);
  for (int level = options_->pyra_levels_ - 1; level >= 0; --level)
  {
    bool adjust = false;
    Mat8f H = Mat8f::Zero();
    Vec8f b = Vec8f::Zero();
    Vec2f res_old = ComputeJacobianAndError(level, Tji_estimate, ai, bi, aj_estimate, bj_estimate, H, b, adjust);
    if (!res_old[1])
      return false;

    float lambda = 0.1;
    int fails_times = 0;
    for (int iteration = 0; iteration < options_->max_iterations_[level]; ++iteration)
    {
      Mat8f H_backup = H;
      Vec8f b_backup = b;
      SE3f Tji_backup = Tji_estimate;
      float aj_backup = aj_estimate;
      float bj_backup = bj_estimate;

      for (int i = 0; i < 8; ++i)
        H(i, i) *= (1 + lambda);

      Vec8f inc = -H.ldlt().solve(b);

      Tji_estimate = Sophus::SE3f::exp(inc.head<6>()) * Tji_estimate;
      aj_estimate += inc[6];
      bj_estimate += inc[7];

      Vec2f res_new = ComputeJacobianAndError(level, Tji_estimate, ai, bi, aj_estimate, bj_estimate, H, b, adjust);
      if (!res_new[1])
        return false;

      float mean_energy_old = res_old[0] / res_old[1];
      float mean_energy_new = res_new[1] / res_new[1];

      bool accept = mean_energy_new < mean_energy_old;
      if (accept)
      {
        lambda *= 0.5;
        fails_times = 0;
        if (lambda < min_lambda)
          lambda = min_lambda;
        std::swap(res_old, res_new);
      }
      else
      {
        lambda *= 4;
        ++fails_times;
        if (lambda > max_lambda)
          lambda = max_lambda;

        // 将优化量进行backup
        H = H_backup;
        b = b_backup;
        Tji_estimate = Tji_backup;
        aj_estimate = aj_backup;
        bj_estimate = bj_backup;
      }

      if (options_->verbose_)
      {
        if (!accept)
          std::cout << "level: " << level << "\titeration: " << iteration << "\tsqrt(energy / nums): " << std::sqrt(res_old[0] / res_old[1]) << "->"
                    << std::sqrt(res_new[0] / res_new[1]) << "\tinlier num: " << static_cast<int>(res_old[1]) << "->" << static_cast<int>(res_new[1])
                    << "\tReject" << std::endl;
        else
          std::cout << "level: " << level << "\titeration: " << iteration << "\tsqrt(energy / nums): " << std::sqrt(res_new[0] / res_new[1]) << "->"
                    << std::sqrt(res_old[0] / res_old[1]) << "\tinlier num: " << static_cast<int>(res_new[1]) << "->" << static_cast<int>(res_old[1])
                    << "\tAccept" << std::endl;
      }
      if (inc.norm() < eps || fails_times >= max_fails_times)
        break;
    }

    // 计算RMSE，判读层级中断条件，用来提前中断不符合条件的层级
    float rmse = std::sqrt(res_old[0] / res_old[1]);
    if (rmse >= 1.5 * abort_res[level])
      return false;

    // 对于那些产生outlier调整的层级，会再次进行优化
    if (adjust && !have_repeated[level])
    {
      have_repeated[level] = true;
      ++level;
    }

    last_rmse_[level] = rmse;
  }

  return true;
}

/**
 * 从后端线程中同步数据，然后才能开始跟踪
 *
 * 1. 同步参考帧状态
 *  1.1 同步参考帧指针到ref_keyframe_self_中，保证能够获取最新的位姿状态
 *  1.2 同步参考帧状态到ref_frame_state_中，与后端产生数据分离，保证线程安全
 *  1.3 同步上一帧参考关键帧的位姿，与后端产生数据分离
 * 2. 同步逆深度点指针，与后端线程保证数据隔离
 * 3. 同步相机内参，与后端线程保证数据隔离
 */
void Tracker::SyncFromLocalMapper()
{
  if (!mapper_changed_)
    return;

  {
    std::lock_guard<std::mutex> lock(tracker_mutex_);
    ref_keyframe_self_ = ref_keyframe_;
    ref_frame_state_.ti = ref_keyframe_->GetExposureTime();
    ref_keyframe_->GetFrameStatus(ref_frame_state_.Trw_, ref_frame_state_.ai, ref_frame_state_.bi);

    if (last_ref_keyframe_)
    {
      last_ref_frame_state_.ti = last_ref_keyframe_->GetExposureTime();
      last_ref_keyframe_->GetFrameStatus(last_ref_frame_state_.Trw_, last_ref_frame_state_.ai, last_ref_frame_state_.bi);
    }

    track_idepth_points_self_ = track_idepth_points_;
    track_fx_self_ = track_fx_;
    track_fy_self_ = track_fy_;
    track_cx_self_ = track_cx_;
    track_cy_self_ = track_cy_;
  }
  mapper_changed_.store(false);
  refframe_changed_ = true;
}

/**
 * 根据Tracker跟踪器的状态，跟踪当前帧
 *
 * 1. 有 last frame 的跟踪尝试
 * 2. 没有 last frame 的跟踪尝试
 *
 * @param frame   输入的待跟踪的帧
 * @return true   跟踪成功
 * @return false  跟踪失败
 */
bool Tracker::TrackActivateFrame(Frame::SharedPtr frame)
{
  curr_frame_ = std::move(frame);
  if (!last_ref_keyframe_)
    return TrackFrameWithoutLast();

  return TrackFrameWithLast();
}


/**
 * 对于没有上一帧条件下的跟踪frame
 *
 * 1. 做线程同步，保证线程安全，取后端线程更新数据，做数据隔离和数据关联
 * 2. 仅使用相对参考关键帧静止策略进行跟踪尝试
 * 3. 对 velocity 等参数进行初始化
 *
 * @return true   跟踪成功
 * @return false  跟踪失败
 */
bool Tracker::TrackFrameWithoutLast()
{
  SyncFromLocalMapper();

  SE3f Tji_estimate;
  float aj = 0, bj = 0;

  // 若参考关键帧发生改变，则abort_rmse_则没有效力，重置为max float
  if (refframe_changed_)
    abort_rmse_ = std::vector<float>(options_->pyra_levels_, std::numeric_limits<float>::max());

  bool track_ret = TryOnce(Tji_estimate, aj, bj, ref_frame_state_.ai, ref_frame_state_.bi, abort_rmse_);
  if (!track_ret)
    return false;

  for (int level = 0; level < options_->pyra_levels_; ++level)
  {
    // 若refframe_changed_ == false，abort_rmse_继续进行更新
    if (!refframe_changed_ && abort_rmse_[level] > last_rmse_[level])
      abort_rmse_[level] = last_rmse_[level];
  }

  if (refframe_changed_)
  {
    abort_rmse_ = last_rmse_;
    first_ref_rmse_ = last_rmse_;
  }

  SE3f Tcw = Tji_estimate * ref_frame_state_.Trw_;
  curr_frame_->SetEstimate(Tcw, aj, bj);

  Tlr_ = Tji_estimate;
  last_ref_keyframe_ = ref_keyframe_self_;
  last_ref_frame_state_ = ref_frame_state_;
  velocity_ = Tji_estimate;

  return true;
}

/**
 * 根据Tracker跟踪器的状态，跟踪当前帧
 *
 * 1. 做线程同步，保证线程安全，取后端线程更新数据，做数据隔离和数据关联
 * 2. 尝试不同的优化假设初值，进行TryOnce，实现提前退出
 * 3. 对 velocity 等参数进行初始化
 *
 * @return true   跟踪成功
 * @return false  跟踪失败
 */
bool Tracker::TrackFrameWithLast()
{
  SyncFromLocalMapper();
  SE3f Tji_estimate;
  float aj = 0, bj = 0;

  SE3f Tlw = Tlr_ * last_ref_frame_state_.Trw_;
  velocity_tries_[0] = velocity_.log();                               ///< 恒速运动模型
  velocity_tries_[1] = velocity_tries_[0] * 2;                        ///< 倍速运动模型
  velocity_tries_[2] = velocity_tries_[0] * 0.5;                      ///< 半速运动模型
  velocity_tries_[4] = (ref_frame_state_.Trw_ * Tlw.inverse()).log(); ///< 相对参考帧静止

  bool have_one_good = false;

  if (refframe_changed_)
    abort_rmse_ = std::vector<float>(options_->pyra_levels_, std::numeric_limits<float>::max());

  for (const auto &velocity_try: velocity_tries_)
  {
    SE3f Tcw = SE3f::exp(velocity_try) * Tlw;
    Tji_estimate = Tcw * ref_frame_state_.Trw_.inverse();

    if (TryOnce(Tji_estimate, aj, bj, ref_frame_state_.ai, ref_frame_state_.bi, abort_rmse_))
    {
      have_one_good = true;
      break;
    }
  }

  if (!have_one_good)
    return false;

  for (int level = 0; level < options_->pyra_levels_ - 1; ++level)
  {
    if (!refframe_changed_ && abort_rmse_[level] > last_rmse_[level])
      abort_rmse_[level] = last_rmse_[level];
  }

  if (refframe_changed_)
  {
    abort_rmse_ = last_rmse_;
    first_ref_rmse_ = last_rmse_;
  }

  SE3f Tcw = Tji_estimate * ref_frame_state_.Trw_;
  curr_frame_->SetEstimate(Tcw, aj, bj);

  Tlr_ = Tji_estimate;
  last_ref_keyframe_ = ref_keyframe_self_;
  last_ref_frame_state_ = ref_frame_state_;
  velocity_ = Tcw * Tlw.inverse();
  return true;
}

/**
 * 使用回溯算法，构建跟踪器的小转角尝试生成（不包含静止条件）
 *
 * @param try_out 输入的尝试向量
 * @param idx     输入的更改索引idx
 *
 */
void Tracker::GenerateTries(Vec3f &try_out, const int &idx)
{
  static std::vector<float> tries_item = {-0.03, 0, 0.03};
  if (idx == 3)
  {
    if (!(!try_out[0] && !try_out[1] && !try_out[2]))
    {
      Vec6f try_out_6f = Vec6f::Zero();
      try_out_6f.tail<3>() = try_out;
      velocity_tries_.push_back(try_out_6f);
    }
    return;
  }

  for (const float &item: tries_item)
  {
    try_out[idx] = item;
    GenerateTries(try_out, idx + 1);
    try_out[idx] = NAN;
  }
}

/**
 * 判断当前跟踪帧是否是关键帧
 *
 * 1. 时间间隔考虑，当间隔超过规定的间隔时间阈值时，插入关键帧
 * 2. 空间间隔考虑，使用[Rji,tji]、[Rji,-tji]、[I,tji]、[I,-tji]计算平均光流，与阈值进行判断
 * 3. 从光度仿射参数考虑，当相对仿射参数exp_aji变化较大时，说明光照条件发生改变，要创建关键帧了
 * 4. 从优化角度考虑，如果当前帧跟踪跟踪的能量值超过当前参考帧第一次跟踪的两倍，则创建关键帧
 *
 * @return true   创建关键帧
 * @return false  不创建关键帧
 */
bool Tracker::IsNeedKeyFrame()
{
  // 1. 时间间隔
  auto cur_stamp = curr_frame_->GetTimestamp();
  auto ref_stamp = ref_keyframe_self_->GetTimestamp();
  double time_interval = cur_stamp - ref_stamp;
  if (options_->time_interval_ > 0 && time_interval > options_->time_interval_)
  {
    logger_->debug("时间间隔超过阈值，创建关键帧");
    return true;
  }

  // 2. 空间间隔，每隔32个点计算一次光流，最后求平均
  int rows = ref_keyframe_self_->GetSqureGrad().rows;
  int cols = ref_keyframe_self_->GetSqureGrad().cols;

  SE3f Tji = curr_frame_->GetTcw() * ref_frame_state_.Trw_.inverse();
  Mat3f Rji = Tji.rotationMatrix();
  Vec3f tji = Tji.translation();

  Mat3f Ki = Mat3f::Zero(), K = Mat3f::Zero();
  Ki << 1.0 / track_fx_[0], 0, -track_cx_[0] / track_fx_[0], 0, 1.0 / track_fy_[0], -track_cy_[0] / track_fy_[0], 0, 0, 1;
  K << track_fx_[0], 0, track_cx_[0], 0, track_fy_[0], track_cy_[0], 0, 0, 1;

  Mat3f KRKi = K * Rji * Ki;
  Vec3f Kt = K * tji;

  std::vector<TrackIdpethPoint::SharedPtr> positions;
  const auto &points_i = track_idepth_points_self_[0];
  for (int idx = 0; idx < points_i.size(); idx += 32)
    positions.push_back(points_i[idx]);

  std::vector<float> tji_flow(positions.size(), 0);
  std::vector<float> Tji_flow(positions.size(), 0);
  std::vector<int> indices(positions.size());
  std::iota(indices.begin(), indices.end(), 0);

  auto position_process = [&](const int &idx)
  {
    const TrackIdpethPoint::SharedPtr &pi = positions[idx];
    Vec3f position_pi(pi->ref_pixel_point_[0], pi->ref_pixel_point_[1], 1);
    float u = position_pi[0], v = position_pi[1];

    // 2.1 positive tji
    Vec3f p_tji_pos = K * Ki * position_pi + Kt * pi->ref_idepth_;
    float u_tji_pos = p_tji_pos[0] / p_tji_pos[2];
    float v_tji_pos = p_tji_pos[1] / p_tji_pos[2];

    // 2.2 negative tji
    Vec3f p_tji_neg = K * Ki * position_pi - Kt * pi->ref_idepth_;
    float u_tji_neg = p_tji_neg[0] / p_tji_neg[2];
    float v_tji_neg = p_tji_neg[1] / p_tji_neg[2];

    // 2.3 positive Tji
    Vec3f p_Tji_pos = KRKi * position_pi + Kt * pi->ref_idepth_;
    float u_Tji_pos = p_Tji_pos[0] / p_Tji_pos[2];
    float v_Tji_pos = p_Tji_pos[1] / p_Tji_pos[2];

    // 2.4 negative Tji
    Vec3f p_Tji_neg = KRKi * position_pi - Kt * pi->ref_idepth_;
    float u_Tji_neg = p_Tji_neg[0] / p_Tji_neg[2];
    float v_Tji_neg = p_Tji_neg[1] / p_Tji_neg[2];

    tji_flow[idx] += (u_tji_pos - u) * (u_tji_pos - u) + (v_tji_pos - v) * (v_tji_pos - v);
    tji_flow[idx] += (u_tji_neg - u) * (u_tji_neg - u) + (v_tji_neg - v) * (v_tji_neg - v);
    Tji_flow[idx] += (u_Tji_pos - u) * (u_Tji_pos - u) + (v_Tji_pos - v) * (v_Tji_pos - v);
    Tji_flow[idx] += (u_Tji_neg - u) * (u_Tji_neg - u) + (v_Tji_neg - v) * (v_Tji_neg - v);
  };

  std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), position_process);
  float tji_flow_sum = std::sqrt(std::accumulate(tji_flow.begin(), tji_flow.end(), 0.f) / (2 * indices.size()));
  float Tji_flow_sum = std::sqrt(std::accumulate(Tji_flow.begin(), Tji_flow.end(), 0.f) / (2 * indices.size()));
  float tji_weight = tji_flow_sum * options_->shift_weight_t_ / (rows + cols);
  float Tji_weight = Tji_flow_sum * options_->shift_weight_rt_ / (rows + cols);
  if (tji_weight + Tji_weight > options_->shift_treshold_)
  {
    logger_->debug("空间间隔超过阈值，创建关键帧");
    return true;
  }

  // 3. 光度仿射参数
  SE3f _;
  float aj, bj;
  curr_frame_->GetFrameStatus(_, aj, bj);

  float tj = curr_frame_->GetExposureTime();
  float ti = ref_frame_state_.ti;
  if (tj <= 0 || ti <= 0)
  {
    tj = 1;
    ti = 1;
  }

  Vec2f ab_ji = PhotoAffine::GetRelative(ref_frame_state_.ai, aj, ref_frame_state_.bi, bj, ti, tj);
  const float &exp_aji = ab_ji[0];
  if (exp_aji > options_->max_exp_aji_threshold_ || exp_aji < options_->min_exp_aji_threshold_)
  {
    logger_->debug("光度仿射参数超过优化阈值，创建关键帧");
    return true;
  }

  // 4. 从优化角度考虑
  if (refframe_changed_)
  {
    refframe_changed_ = false;
    return false;
  }

  if (first_ref_rmse_[0] * options_->rmse_threshold_factor_ < last_rmse_[0])
  {
    logger_->debug("超过优化阈值，创建关键帧");
    return true;
  }

  return false;
}


/**
 * @brief 构造跟踪器配置文件
 *
 * @param filepath 输入的跟踪器配置文件路径
 */
Tracker::Options::Options(const std::string &filepath)
{
  if (!std::filesystem::exists(filepath))
    throw std::runtime_error("配置文件不存在");

  auto info = YAML::LoadFile(filepath);
  pyra_levels_ = info["PyraidLevelsUsed"].as<int>();
  huber_threshold_ = info["HuberThreshold"].as<float>();
  max_iterations_ = info["MaxIterations"].as<std::vector<int>>();
  verbose_ = info["Verbose"].as<bool>();
  outlier_threshold_ = info["OutlierThreshold"].as<float>();
  time_interval_ = info["TimeInterval"].as<float>();
  shift_weight_t_ = info["ShiftWeightT"].as<float>();
  shift_weight_rt_ = info["ShiftWeightRT"].as<float>();
  shift_treshold_ = info["ShiftTreshold"].as<float>();
  max_exp_aji_threshold_ = info["MaxExpAjiThreshold"].as<float>();
  min_exp_aji_threshold_ = info["MinExpAjiThreshold"].as<float>();
  rmse_threshold_factor_ = info["RmseThresholdFactor"].as<float>();
}


} // namespace dso_ssl
