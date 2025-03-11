#include <fstream>

#include "dso/Undistorter.hpp"
#include "utils/Interpolate.hpp"
#include "utils/ParallelProcess.hpp"

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

/**
 * @brief 根据输入的file_path构造光度去畸变器的配置
 *
 * @param file_path 配置文件的路径
 *
 * @throw std::runtime_error 当配置文件，配置文件中指定的文件路径不存在时抛出运行时错误。
 *
 * 该构造函数接收一个配置文件路径，然后根据该路径加载配置信息。它首先尝试获取配置文件的绝对路径，
 * 并检查文件是否存在。如果文件不存在，将抛出一个运行时错误。接着，使用YAML库加载配置文件内容，并从中
 * 提取配置信息。根据配置信息中的标志决定是否加载特定的功能参数。
 */
PhotoUndistorter::Config::Config(std::string file_path)
{
    // 获取配置文件的绝对路径
    auto abs_config_path = std::filesystem::absolute(std::filesystem::path(file_path)).lexically_normal();
    if (!std::filesystem::exists(abs_config_path))
        throw std::runtime_error("Config file not found");

    auto info = YAML::LoadFile(abs_config_path);

    gfunc_inv_flag_ = info["GInvFunction"]["UseFlag"].as<bool>();
    vignette_flag_ = info["Vignette"]["UseFlag"].as<bool>();

    if (gfunc_inv_flag_)
    {
        std::filesystem::path gfunc_file_path(info["GInvFunction"]["FilePath"].as<std::string>());
        if (!gfunc_file_path.is_absolute())
            gfunc_file_path = (abs_config_path.parent_path() / gfunc_file_path).lexically_normal();

        if (!std::filesystem::exists(gfunc_file_path))
            throw std::runtime_error("GInvFunction: file not found");

        LoadGInvFunParams(gfunc_file_path);
        // NormalizeGInv();
        NormalizeGInvParallel();
    }

    if (vignette_flag_)
    {
        std::filesystem::path vignette_path(info["Vignette"]["FilePath"].as<std::string>());

        if (!vignette_path.is_absolute())
            vignette_path = (abs_config_path.parent_path() / vignette_path).lexically_normal();

        if (!std::filesystem::exists(vignette_path))
            throw std::runtime_error("Vignette: file not found");

        vignette_map_ = cv::imread(vignette_path, cv::IMREAD_GRAYSCALE);
        vignette_map_.convertTo(vignette_map_, CV_32F);
        NormalizeVignette();
    }
}

/**
 * 加载GInv函数的参数
 *
 * 本函数从指定的文件路径中读取GInv函数的参数，并确保参数数量正确
 * 参数被读入到一个预分配了256个元素空间的向量中，如果读取的参数数量不等于256，
 * 则抛出运行时错误，如果GInv不是递增函数，则抛出运行时错误
 *
 * @param ginv_params_path 包含GInv函数参数的文件路径
 */
void PhotoUndistorter::Config::LoadGInvFunParams(const std::string &ginv_params_path)
{
    std::ifstream ginv_params_file(ginv_params_path);
    float ginv_param = 0.f;
    std::vector<float> ginv_params;
    ginv_params.reserve(256);

    while (ginv_params_file >> ginv_param)
    {
        if (ginv_params[ginv_params.size() - 1] > ginv_param)
            throw std::runtime_error("GInvFunction: ginv_params_file must be sorted");

        ginv_params.push_back(ginv_param);
    }

    if (ginv_params.size() != 256)
        throw std::runtime_error("GInvFunction: ginv_params_file must have 256 elements");

    std::swap(gfunc_inv_, ginv_params);
}

/**
 * @brief 实现图像去畸变功能，使用非线性映射的逆过程对畸变图像进行光度校正
 *
 * @param distorted_img 输入的畸变图像，必须是8U或8UC3类型（整个畸变矫正的最始端）
 * @return cv::Mat 输出的去G作用后的图像，类型变成了Float
 *
 * @note 即便配置中指定没有GInv函数，该函数仍然会对输入图像进行类型转换为Float
 * @see PhotoUndistorter::GinvUndistortOne
 */
cv::Mat PhotoUndistorter::GinvUndistort(const cv::Mat &distorted_img)
{
    // 输入类型检查，必须为8U或8UC3类型，因为是最初的矫正处理
    if (!(distorted_img.type() == CV_8U || distorted_img.type() == CV_8UC3))
        throw std::runtime_error("GinvUndistort: input image type must be CV_8U or CV_8UC3");

    cv::Mat ginv_undistorted_img;
    if (!config_->gfunc_inv_flag_)
    {
        distorted_img.copyTo(ginv_undistorted_img);
        switch (distorted_img.type())
        {
        case CV_8U:
            ginv_undistorted_img.convertTo(ginv_undistorted_img, CV_32F);
            break;

        case CV_8UC3:
            ginv_undistorted_img.convertTo(ginv_undistorted_img, CV_32FC3);
            break;
        default:
            throw std::runtime_error("GinvUndistort: unsupported image type");
        }

        return ginv_undistorted_img;
    }

    std::vector<cv::Mat> channels;
    if (distorted_img.channels() == 1)
        channels.push_back(distorted_img);
    else
        cv::split(distorted_img, channels);

    // 非线性响应函数检查，必须为256个元素
    if (config_->gfunc_inv_.size() != 256)
        throw std::runtime_error("GinvUndistort: gfunc_inv_ must be 256 elements");

    std::vector<cv::Mat> out_channels(channels.size());
    auto process_func = [&](const int &idx) { out_channels[idx] = GinvUndistortOne(channels[idx]); };

    std::vector<int> indices(channels.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), process_func);

    if (distorted_img.channels() == 3)
        cv::merge(out_channels, ginv_undistorted_img);
    else
        out_channels[0].copyTo(ginv_undistorted_img);

    return ginv_undistorted_img;
}

/**
 * @brief 渐晕校正
 *
 * @param distorted_img 输入的畸变图像，必须是32F或32FC3类型
 * @return cv::Mat  输出的去渐晕作用后的图像，类型与输入类型一致
 *
 */
cv::Mat PhotoUndistorter::VignUndistort(const cv::Mat &distorted_img)
{
    if (!(distorted_img.type() == CV_32F || distorted_img.type() == CV_32FC3))
        throw std::runtime_error("VignUndistort: input image type must be CV_32F or CV_32FC3");

    // 输入尺寸检查，必须与渐晕map图尺寸相同
    if (distorted_img.rows != config_->vignette_map_.rows || distorted_img.cols != config_->vignette_map_.cols)
        throw std::runtime_error("VignUndistort: input image size must be equal to vignette map size");

    return mat_op::MatOperatorCorr<mat_op::Operators::Div>(distorted_img, config_->vignette_map_);
}

/**
 * @brief 对单通道CV_32F进行处理
 *
 * @param distorted_img 输入的单通道，类型为CV_32F的图像
 * @return cv::Mat  输出的去G作用后的图像，类型与输入类型一致
 */
cv::Mat PhotoUndistorter::GinvUndistortOne(const cv::Mat &distorted_img)
{
    cv::Mat ginv_undistorted_img(distorted_img.rows, distorted_img.cols, CV_32F);

    std::vector<int> indices(distorted_img.rows * distorted_img.cols, 0);
    std::iota(indices.begin(), indices.end(), 0);

    auto process_func = [&](const int &idx)
    {
        int row = idx / distorted_img.cols;
        int col = idx % distorted_img.cols;

        const uchar &color = distorted_img.at<uchar>(row, col);
        ginv_undistorted_img.at<float>(row, col) = config_->gfunc_inv_[static_cast<int>(color)];
    };

    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), process_func);
    return ginv_undistorted_img;
}

/**
 * @brief 标准化GInv函数
 *
 * 在区间[0,255]之间分布
 */
void PhotoUndistorter::Config::NormalizeGInv()
{
    std::vector<float> temp(256, 0);
    float factor = 255.f / (gfunc_inv_[255] - gfunc_inv_[0]);
    auto factor_sse = _mm_set1_ps(factor);

    auto process_func = [&](const tbb::blocked_range<int> &range)
    {
        int remains = (range.end() - range.begin()) % 4;

        for (int start = range.begin(); start < range.end() - 3; start += 4)
        {
            alignas(16) float data[4];
            int position[4];
            for (int idx = start; idx < start + 4; idx++)
            {
                position[idx - start] = idx;
                data[idx - start] = gfunc_inv_[idx];
            }

            auto data_sse = _mm_load_ps(data);
            auto result = _mm_mul_ps(data_sse, factor_sse);

            alignas(16) float result_data[4];
            _mm_store_ps(result_data, result);

            for (int idx = 0; idx < 4; ++idx)
                temp[position[idx]] = result_data[idx];
        }

        for (int idx = 1; idx <= remains; ++idx)
            temp[range.end() - idx] = gfunc_inv_[range.end() - idx] * factor;
    };

    tbb::parallel_for(tbb::blocked_range<int>(0, 256, 4), process_func);

    std::swap(temp, gfunc_inv_);
}

/**
 * @brief 标准化GInv函数
 *
 * 在区间[0,255]之间分布
 */
void PhotoUndistorter::Config::NormalizeGInvParallel()
{
    std::vector<float> temp(256, 0);
    float factor = 255.f / (gfunc_inv_[255] - gfunc_inv_[0]);
    auto factor_sse = _mm_set1_ps(factor);

    auto process_simd = [&](const int &start)
    {
        alignas(16) float data[4];
        int position[4];
        for (int idx = start; idx < start + 4; idx++)
        {
            position[idx - start] = idx;
            data[idx - start] = gfunc_inv_[idx];
        }

        auto data_sse = _mm_load_ps(data);
        auto result = _mm_mul_ps(data_sse, factor_sse);

        alignas(16) float result_data[4];
        _mm_store_ps(result_data, result);

        for (int idx = 0; idx < 4; ++idx)
            temp[position[idx]] = result_data[idx];
    };

    auto process_single = [&](const int &idx) { temp[idx] = gfunc_inv_[idx] * factor; };

    parallel::ParallelWrapper(0, 256, 4, process_simd, process_single);

    std::swap(temp, gfunc_inv_);
}

/**
 * @brief 最大值归一化渐晕map图
 *
 * 在区间[0, 1]之间分布
 */
void PhotoUndistorter::Config::NormalizeVignette()
{
    double max_value, min_value;

    cv::minMaxLoc(vignette_map_, &min_value, &max_value);

    if (max_value == 0)
        throw std::runtime_error("NormalizeVignette: vignette map max item is zero");

    vignette_map_ /= max_value;
}

/**
 * @brief 根据Ginv 计算 G 函数
 *
 */
void PhotoUndistorter::Config::ComputeGFunction()
{
    gfunc_ = std::vector<float>(256, 0);
    gfunc_[255] = 255;
    for (int I = 0; I < 255; ++I)
        for (int idx = 0; idx < 255; ++idx)
        {
            if (gfunc_inv_[idx] <= I && gfunc_inv_[idx + 1] > I)
            {
                gfunc_[I] = (I - gfunc_inv_[idx]) * (idx + 1) + (gfunc_inv_[idx + 1] - I) * idx;
                break;
            }
        }
}

/**
 * @brief 计算G函数相对于I'v(x)的Jacobian矩阵
 *
 * @param Ivalue 输入的I'v(x)
 * @return float 输出的 dG / d(I'v)
 */
float PhotoUndistorter::Config::ComputeGJacobian(float Ivalue)
{
    int idx = Ivalue + 0.5;

    // 边界条件处理
    if (idx > 254)
        idx = 254;

    return gfunc_[idx + 1] - gfunc_[idx];
}

/**
 * @brief 去畸变器构造
 *
 * 构造像素去畸变器和光度去畸变器，然后根据像素去畸变器的像素坐标映射标更新vignette，渐晕函数
 *
 * @param pixel_config 像素去畸变器配置
 * @param photo_config 光度去畸变器配置
 */
Undistorter::Undistorter(PixelUndistorter::Config::SharedPtr pixel_config, PhotoUndistorter::Config::SharedPtr photo_config, Config::SharedPtr config)
    : config_(std::move(config))
{
    pixel_undistorter_ = std::make_shared<PixelUndistorter>(pixel_config);
    photo_undistorter_ = std::make_shared<PhotoUndistorter>(photo_config);
}

/**
 * @brief 去畸变器核心去畸变函数
 *
 * 如果在点选过程中，要求使用原始图像梯度，就需要
 * only_pixel_undistorted_image 像素去畸变图像作为中间产物输出
 *
 * @param distorted_image               输入的畸变图像
 * @param only_pixel_undistorted_image  输出的去像素畸变图像
 * @return cv::Mat 输出的去畸变图像
 */
cv::Mat Undistorter::Undistort(const cv::Mat &distorted_image, cv::Mat &only_pixel_undistorted_image)
{
    // 如果在点选中，使用原始图像梯度，那么需要这个中间变量
    if (config_->use_origin_grad_)
    {
        cv::Mat distorted_image_fp;
        distorted_image.convertTo(distorted_image_fp, CV_32F);
        only_pixel_undistorted_image = pixel_undistorter_->Undistort(distorted_image_fp);
    }

    auto photo_undistorted_image = photo_undistorter_->Undistort(distorted_image);
    auto pixel_undistorted_image = pixel_undistorter_->Undistort(photo_undistorted_image);

    return pixel_undistorted_image;
}

} // namespace dso_ssl