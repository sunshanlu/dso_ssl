#include <fstream>

#include "dso/PhotoUndistorter.hpp"

namespace dso_ssl
{

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
        NormalizeGInv();
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
            temp[range.end() - remains] = gfunc_inv_[range.end() - remains] * factor;
    };

    tbb::parallel_for(tbb::blocked_range<int>(0, 256, 4), process_func);

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

} // namespace dso_ssl