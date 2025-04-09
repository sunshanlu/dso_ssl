#include "dso/Tracker.hpp"

#include <spdlog/common.h>

namespace dso_ssl
{

/**
 * @brief 将滑动窗口中的关键帧向最新关键帧进行投影
 *
 * 1. 要求关键帧的逆深度点状态里面没有外点
 * 2. 使用的逆深度点时、Tcw等滑动窗口状态时，不能在逆深度更新过程中
 * 3. 由于使用四舍五入，因此存在多对一的情况，使用高斯归一化积
 *
 * @param sliding_window          输入的滑动窗口，用于逆深度点投影
 * @param ref_idepth_sum          输出的逆深度点高斯归一化积的分布情况，第0层
 * @param ref_hessian_sum         输出的逆深度点hessian的分布情况，第0层
 */
void Tracker::ProjectWindow2Ref(const std::vector<KeyFrame::SharedPtr> &sliding_window, cv::Mat &ref_idepth_sum,
                                cv::Mat &ref_hessian_sum)
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

    SE3f Trk = ref_keyframe_->GetTcw().inverse() * keyframe->GetTcw();
    Mat3f RKi = Trk.rotationMatrix() * Ki;
    Vec3f trk = Trk.translation();

    for (int point_idx = 0; point_idx < map_points.size(); ++point_idx)
    {
      Vec2f point_k = map_points[point_idx]->GetHostPixel();
      float idepth_k = map_points[point_idx]->GetIdepth();
      Vec3f pr_temp = RKi * Vec3f(point_k[0], point_k[1], 1.f) + trk * idepth_k;
      float idepth_r = idepth_k / pr_temp[2];
      float ur = pr_temp[0] / pr_temp[2], vr = pr_temp[1] / pr_temp[2];
      int kur = ur * fx0 + cx0 + 0.5;
      int kvr = vr * fy0 + cy0 + 0.5;

      if (kur < 0 || kur > square_grad.cols - 1 || kvr < 0 || kvr > square_grad.rows - 1)
        continue;

      float hessian = map_points[point_idx]->GetHessian();
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
void Tracker::PropagateUp(const cv::Mat &curr_idepth_sum, const cv::Mat &curr_hessian_sum, cv::Mat &prev_idepth_sum,
                          cv::Mat &prev_hessian_sum)
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

    float hessian_sum0 = curr_hessian_sum.at<float>(2 * row, 2 * col);
    float hessian_sum1 = curr_hessian_sum.at<float>(2 * row, 2 * col + 1);
    float hessian_sum2 = curr_hessian_sum.at<float>(2 * row + 1, 2 * col);
    float hessian_sum3 = curr_hessian_sum.at<float>(2 * row + 1, 2 * col + 1);
    prev_hessian_sum.at<float>(row, col) += (hessian_sum0 + hessian_sum1 + hessian_sum2 + hessian_sum3);
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
    if (hessian > 0)
      return;

    for (int inner_idx = 0; inner_idx < 4; ++inner_idx)
    {
      const Vec2i &pattern_positon = pattern[inner_idx];
      float &hessian_around = input_hessian_sum.at<float>(row + pattern_positon[1], col + pattern_positon[0]);
      float &idepth_around = input_idepth_sum.at<float>(row + pattern_positon[1], col + pattern_positon[0]);
      level_hessian_sum.at<float>(row, col) += hessian_around;
      level_idepth_sum.at<float>(row, col) += idepth_around;
    }
  };

  std::vector<int> indices(rows * cols);
  std::iota(indices.begin(), indices.end(), 0);
  std::for_each(std::execution::par, indices.begin(), indices.end(), position_process);

  std::swap(level_idepth_sum, input_idepth_sum);
  std::swap(level_hessian_sum, input_hessian_sum);
}

/**
 * @brief 构建参考点，将构造的参考点放到 track_idepth_points中
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


  for (int level = 0; level < options_->pyra_levels_; ++level)
  {
    cv::Mat &curr_idepth_sum = pyramid_idepth_sum[level];
    cv::Mat &curr_hessian_sum = pyramid_hessian_sum[level];
    if (level < options_->pyra_levels_ - 1)
    {
      cv::Mat &prev_idepth_sum = pyramid_idepth_sum[level + 1];
      cv::Mat &prev_hessian_sum = pyramid_hessian_sum[level + 1];
      PropagateUp(curr_idepth_sum, curr_hessian_sum, prev_idepth_sum, prev_hessian_sum);
    }
    IdpethExpansion(level, curr_idepth_sum, curr_hessian_sum);

    // 求解高斯归一化部分
    auto position_process = [&](const int &idx)
    {
      int row = idx / curr_idepth_sum.cols;
      int col = idx % curr_idepth_sum.cols;

      const float &hessian = curr_hessian_sum.at<float>(row, col);
      if (hessian <= 0)
        return;

      auto tracker_point = std::make_shared<TrackIdpethPoint>();
      tracker_point->ref_idepth_ = curr_idepth_sum.at<float>(row, col) / hessian;
      tracker_point->ref_pixel_point_ = Vec2f(col, row);
      track_idepth_points_[level].push_back(tracker_point);
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
 * @param sliding_window 输入的滑动窗口，用于向最新关键帧投影
 * @param fx             后端优化后的fx
 * @param fy             后端优化后的fy
 * @param cx             后端优化后的cx
 * @param cy             后端优化后的cy
 */
void Tracker::UpdateTracker(const std::vector<KeyFrame::SharedPtr> &sliding_window, const float &fx, const float &fy,
                            const float &cx, const float &cy)
{
  std::unique_lock<std::mutex> lock;

  // 条件变量在谓词lambda执行过程中，已经拿到了lock的所有权
  tracker_cond_var_.wait(lock, [&]() { return is_notified_; });
  UpdateCalib(fx, fy, cx, cy);
  BuildTrackerPoints(sliding_window);
  is_notified_ = false;
}

/**
 * @brief 更新Tracker跟踪器的内参矩阵
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
}


} // namespace dso_ssl
