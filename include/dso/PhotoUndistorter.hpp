#pragma once

#include <execution>
#include <filesystem>
#include <memory>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "utils/MatOperators.hpp"

namespace dso_ssl
{

class PhotoUndistorter
{
public:
    using SharedPtr = std::shared_ptr<PhotoUndistorter>;

    /// 光度去畸变器的配置
    struct Config
    {
        using SharedPtr = std::shared_ptr<Config>;

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
        Config(std::string file_path);

        /**
         * 加载GInv函数的参数
         *
         * 本函数从指定的文件路径中读取GInv函数的参数，并确保参数数量正确
         * 参数被读入到一个预分配了256个元素空间的向量中，如果读取的参数数量不等于256，
         * 则抛出运行时错误
         *
         * @param ginv_params_path 包含GInv函数参数的文件路径
         */
        void LoadGInvFunParams(const std::string &ginv_params_path);

        /**
         * @brief 标准化GInv函数
         *
         * 在区间[0,255]之间分布
         */
        void NormalizeGInv();

        /**
         * @brief 标准化渐晕map图
         *
         * 在区间[0, 1]之间分布
         */
        void NormalizeVignette();

        cv::Mat vignette_map_;         ///< 渐晕map图
        std::vector<float> gfunc_inv_; ///< 非线性响应函数G^-1
        bool vignette_flag_;           ///< 是否使用渐晕处理
        bool gfunc_inv_flag_;          ///< 是否使用非线性响应函数G^-1
    };

    PhotoUndistorter(Config::SharedPtr config)
        : config_(std::move(config))
    {
    }

    /**
     * @brief 光度校正核心函数
     *
     * 1. 对非线性响应函数G^-1进行校正
     * 2. 对渐晕map图进行校正
     * 3. 适合单通道和三通道图像的光度矫正（CV_8U, CV_8UC3）
     *
     * @param distorted_img 输入的去畸变图像
     * @return cv::Mat  输出的去光度畸变的校正图像
     */
    cv::Mat Undistort(const cv::Mat &distorted_img)
    {
        auto ginv_undistorted_img = GinvUndistort(distorted_img);
        return VignUndistort(ginv_undistorted_img);
    }

    /**
     * @brief 实现图像去畸变功能，使用非线性映射的逆过程对畸变图像进行光度校正
     *
     * @param distorted_img 输入的畸变图像，必须是8U或8UC3类型（整个畸变矫正的最始端）
     * @return cv::Mat 输出的去G作用后的图像，类型变成了Float
     *
     * @note 即便配置中指定没有GInv函数，该函数仍然会对输入图像进行类型转换为Float
     */
    cv::Mat GinvUndistort(const cv::Mat &distorted_img);

    /**
     * @brief 渐晕校正
     *
     * @param distorted_img 输入的畸变图像，必须是32F或32FC3类型
     * @return cv::Mat  输出的去渐晕作用后的图像，类型与输入类型一致
     */
    cv::Mat VignUndistort(const cv::Mat &distorted_img);

private:
    /**
     * @brief 对单通道CV_32F进行处理
     *
     * @param distorted_img 输入的单通道，类型为CV_32F的图像
     * @return cv::Mat  输出的去G作用后的图像，类型与输入类型一致
     */
    cv::Mat GinvUndistortOne(const cv::Mat &distorted_img);

    Config::SharedPtr config_; ///< 光度去畸变器配置
};

} // namespace dso_ssl