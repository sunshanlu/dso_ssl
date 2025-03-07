#include <dso/Undistorter.hpp>

namespace dso_ssl
{

/**
 * @brief 获取相机的内参矩阵
 *
 * @return K::Mat33f 输出相机的内参矩阵
 */
K::Mat33f K::GetMatrix()
{
    Mat33f K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    return K;
}

/**
 * @brief RadTan 3参数畸变函数
 *
 * @param undistorted_point 输入的无畸变归一化坐标系坐标点
 * @return Eigen::Vector2f  输出的畸变归一化坐标系坐标点
 */
Eigen::Vector2f PixelUndistorter::RT3Config::Distort(const Eigen::Vector2f &undistorted_point)
{
    auto rt3_distorted_params = std::dynamic_pointer_cast<undistort::RadTanParams<3>>(distorted_params_);
    if (!rt3_distorted_params)
        throw std::runtime_error("Invalid dynamic cast to RadTanParams<3>");

    return undistort::RadTan<3>(*rt3_distorted_params, undistorted_point);
}

/**
 * @brief RadTan 5参数畸变函数
 *
 * @param undistorted_point 输入的无畸变归一化坐标系坐标点
 * @return Eigen::Vector2f  输出的畸变归一化坐标系坐标点
 */
Eigen::Vector2f PixelUndistorter::RT5Config::Distort(const Eigen::Vector2f &undistorted_point)
{
    auto rt5_distorted_params = std::dynamic_pointer_cast<undistort::RadTanParams<5>>(distorted_params_);
    if (!rt5_distorted_params)
        throw std::runtime_error("Invalid dynamic cast to RadTanParams<3>");

    return undistort::RadTan<5>(*rt5_distorted_params, undistorted_point);
}

/**
 * @brief FOV 畸变函数
 *
 * @param undistorted_point 输入的无畸变归一化坐标系坐标点
 * @return Eigen::Vector2f  输出的畸变归一化坐标系坐标点
 */
Eigen::Vector2f PixelUndistorter::FOVConfig::Distort(const Eigen::Vector2f &undistorted_point)
{
    auto fov_distorted_params = std::dynamic_pointer_cast<undistort::FOVParams>(distorted_params_);
    if (!fov_distorted_params)
        throw std::runtime_error("Invalid dynamic cast to RadTanParams<3>");

    return undistort::FOV(*fov_distorted_params, undistorted_point);
}

/**
 * @brief KB 畸变函数
 *
 * @param undistorted_point 输入的无畸变归一化坐标系坐标点
 * @return Eigen::Vector2f  输出的畸变归一化坐标系坐标点
 */
Eigen::Vector2f PixelUndistorter::KBConfig::Distort(const Eigen::Vector2f &undistorted_point)
{
    auto kb_distorted_params = std::dynamic_pointer_cast<undistort::KBParams>(distorted_params_);
    if (!kb_distorted_params)
        throw std::runtime_error("Invalid dynamic cast to RadTanParams<3>");

    return undistort::KB(*kb_distorted_params, undistorted_point);
}

/**
 * @brief 根据得到的轴位置推算出无畸变图像的虚拟内参
 *
 */
void PixelUndistorter::ComputeTargetK()
{
    if (config_->distorted_type_ == undistort::DistortionT::Pinhole)
    {
        PinholeProcess();
        return;
    }

    float X_max, Y_max, X_min, Y_min;
    ComputeLimitAxis(X_max, Y_max, X_min, Y_min);
    ComputeRealAxis(X_max, Y_max, X_min, Y_min);

    target_K_.fx = config_->target_size_[0] / (X_max - X_min);
    target_K_.fy = config_->target_size_[1] / (Y_max - Y_min);
    target_K_.cx = -X_min * target_K_.fx;
    target_K_.cy = -Y_min * target_K_.fy;
}

/**
 * @brief 构建映射关系表 remap_x_ 和 remap_y_
 *
 */
void PixelUndistorter::BuildRemap()
{

    int pixels_num = config_->target_size_[0] * config_->target_size_[1];
    std::vector<float> remap_x(pixels_num, 0), remap_y(pixels_num, 0);

    std::vector<int> map_idx(pixels_num, 0);
    std::iota(map_idx.begin(), map_idx.end(), 0);

    std::for_each(std::execution::par_unseq, map_idx.begin(), map_idx.end(),
                  [&](const int &idx)
                  {
                      int v = idx / config_->target_size_[0];
                      int u = idx % config_->target_size_[0];

                      float xn = (u - target_K_.cx) / target_K_.fx;
                      float yn = (v - target_K_.cy) / target_K_.fy;

                      Eigen::Vector2f undistorted_point(xn, yn);
                      auto distorted_point = config_->Distort(undistorted_point);

                      remap_x[idx] = config_->source_K_.fx * distorted_point[0] + config_->source_K_.cx;
                      remap_y[idx] = config_->source_K_.fy * distorted_point[1] + config_->source_K_.cy;
                  });

    // std::cout << "x_max: " << *std::max_element(remap_x.begin(), remap_x.end()) << std::endl;
    // std::cout << "x_min: " << *std::min_element(remap_x.begin(), remap_x.end()) << std::endl;
    // std::cout << "y_max: " << *std::max_element(remap_y.begin(), remap_y.end()) << std::endl;
    // std::cout << "y_min: " << *std::min_element(remap_y.begin(), remap_y.end()) << std::endl;

    cv::Mat(config_->target_size_[1], config_->target_size_[0], CV_32F, remap_x.data()).copyTo(remap_x_);
    cv::Mat(config_->target_size_[1], config_->target_size_[0], CV_32F, remap_y.data()).copyTo(remap_y_);
}

/**
 * @brief 找到所有极限轴上的点，都能投影到畸变图像上的归一化坐标轴位置
 * @details
 *      1. 从四个轴上，以目标宽和目标高为粒度进行点划分
 *      2. 遍历所有点，判断是否所有点都能投影到畸变图像上
 *          2.1 如果能投影到畸变图像上，则标志某个轴满足要求
 *          2.2 如果存在不能投影到图像上的点，则需要调整某个轴的值
 */
void PixelUndistorter::ComputeRealAxis(float &x_max, float &y_max, float &x_min, float &y_min)
{
    bool x_max_flag, x_min_flag, y_max_flag, y_min_flag;
    x_max_flag = x_min_flag = y_max_flag = y_min_flag = false;
    int adjust_iteraiton = 0;
    while (!(x_max_flag && x_min_flag && y_max_flag && y_min_flag) && adjust_iteraiton < config_->max_adjust_iters_)
    {
        x_max_flag = MatchAxisProject(x_max, Axis::XAxis, y_min, y_max, x_max_flag);
        x_min_flag = MatchAxisProject(x_min, Axis::XAxis, y_min, y_max, x_min_flag);
        y_max_flag = MatchAxisProject(y_max, Axis::YAxis, x_min, x_max, y_max_flag);
        y_min_flag = MatchAxisProject(y_min, Axis::YAxis, x_min, x_max, y_min_flag);

        /// 如果x和y轴都需要调整时，仅调整那个间距大的轴部分
        if (!(x_max_flag && x_min_flag) && !(y_max_flag && y_min_flag))
        {
            if (x_max - x_min > y_max - y_min)
            {
                if (!y_max_flag)
                    y_max *= 0.995;
                if (!y_min_flag)
                    y_min *= 0.995;
            }
            else
            {
                if (!x_max_flag)
                    x_max *= 0.995;
                if (!x_min_flag)
                    x_min *= 0.995;
            }
            adjust_iteraiton++;
            continue;
        }

        if (!x_max_flag)
            x_max *= 0.995;
        if (!x_min_flag)
            x_min *= 0.995;
        if (!y_max_flag)
            y_max *= 0.995;
        if (!y_min_flag)
            y_min *= 0.995;
        adjust_iteraiton++;
    }

    if (adjust_iteraiton >= config_->max_adjust_iters_)
        throw std::runtime_error("Adjust axis failed, may increate max_adjust_iters_");
}

/**
 * @brief 判断某个轴上的点是否都能投影到畸变图像上
 *
 * @note 留出一个3px的边框，保证图像差值时不会越界
 *
 * @param axis_value        输入输出的轴值
 * @param axis_direction    输入的轴方向（x || y）
 * @param other_min_value   输入的另一个轴的最小值
 * @param other_max_value   输入的另一个轴的最大值
 * @return true             轴上的点都能投影到畸变图像上
 * @return false            轴上的点有的不能都投影到畸变图像上
 */
bool PixelUndistorter::MatchAxisProject(float &axis_value, Axis axis_direction, const float &other_min_value, const float &other_max_value,
                                        const bool &axis_flag)
{
    if (axis_flag)
        return true;

    float axis_gap = (other_max_value - other_min_value) / config_->target_size_[0];
    if (axis_direction == Axis::YAxis)
        axis_gap = (other_max_value - other_min_value) / config_->target_size_[1];

    bool any_oob_flag = false;
    for (float other_axis_value = other_min_value; other_axis_value <= other_max_value; other_axis_value += axis_gap)
    {
        Eigen::Vector2f undistorted_point(axis_value, other_axis_value);
        if (axis_direction == Axis::YAxis)
            std::swap(undistorted_point[0], undistorted_point[1]);

        auto distorted_point = config_->Distort(undistorted_point);
        float u = config_->source_K_.fx * distorted_point[0] + config_->source_K_.cx;
        float v = config_->source_K_.fy * distorted_point[1] + config_->source_K_.cy;

        /// 留出一个3px的边框，保证图像差值时不会越界
        if (u >= config_->source_size_[0] - 3 || u < 3 || v >= config_->source_size_[1] - 3 || v < 3)
        {
            any_oob_flag = true;
            break;
        }
    }

    return !any_oob_flag;
}

/**
 * @brief 获取投影极限位置的坐标轴 lx_max，ly_max，lx_min，ly_min
 * @details
 *      1. 在x=0的坐标轴上，从[-5, 5]取100001个点，投影到畸变图像上，找到一个能投影到图像上的极限位置
 *      2. 在y=0的坐标轴上，从[-5, 5]取100001个点，投影到畸变图像上，找到一个能投影到图像上的极限位置
 * @param lx_max 输出的x轴最大值
 * @param ly_max 输出的y轴最大值
 * @param lx_min 输出的x轴最小值
 * @param ly_min 输出的y轴最小值
 */
void PixelUndistorter::ComputeLimitAxis(float &lx_max, float &ly_max, float &lx_min, float &ly_min)
{
    std::vector<float> limits_axis_pos(4, 0);
    std::vector<int> indices(4, 0);
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<Grid> grids = CreateMeshgrid();
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
                  [&](const int &idx)
                  {
                      const auto &grid = grids[idx];
                      for (const auto &undistorted_point : grid)
                      {
                          auto distorted_point = config_->Distort(undistorted_point);
                          float u = config_->source_K_.fx * distorted_point[0] + config_->source_K_.cx;
                          float v = config_->source_K_.fy * distorted_point[1] + config_->source_K_.cy;

                          if (u < config_->source_size_[0] - 3 && u >= 3 && v < config_->source_size_[1] - 3 && v >= 3)
                          {
                              if (idx < 2)
                                  limits_axis_pos[idx] = undistorted_point[0];
                              else
                                  limits_axis_pos[idx] = undistorted_point[1];
                              break;
                          }
                      }
                  });
    lx_min = limits_axis_pos[0];
    lx_max = limits_axis_pos[1];
    ly_min = limits_axis_pos[2];
    ly_max = limits_axis_pos[3];
}

/// 在无畸变归一化坐标系上，找到2 * 10000个点
std::vector<PixelUndistorter::Grid> PixelUndistorter::CreateMeshgrid()
{
    const float gap = 10.f / 100000;
    Grid grid(50000, Eigen::Vector2f(0, 0));
    std::vector<Grid> grids(4, Grid(50000, Eigen::Vector2f(0, 0)));

    // 投影不到->投影到
    float positive_num = 5.0f, negative_num = -5.0f;
    for (int i = 0; i < 50000; ++i)
    {
        grids[0][i] = Eigen::Vector2f(negative_num, 0);
        grids[1][i] = Eigen::Vector2f(positive_num, 0);
        grids[2][i] = Eigen::Vector2f(0, negative_num);
        grids[3][i] = Eigen::Vector2f(0, positive_num);
        positive_num -= gap;
        negative_num += gap;
    }
    return grids;
}

} // namespace dso_ssl
