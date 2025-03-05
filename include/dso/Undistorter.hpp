#pragma once

#include <algorithm>
#include <execution>
#include <memory>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <tbb/parallel_for_each.h>
#include <yaml-cpp/yaml.h>

#include "utils/Interpolate.hpp"
#include "utils/PixelUndistorted.hpp"

namespace dso_ssl
{

/// 相机内参存储结构
struct K
{
    using Mat33f = Eigen::Matrix<float, 3, 3>;

    /// 获取相机内参矩阵
    Mat33f GetMatrix();

    float fx, fy;
    float cx, cy;
};

/// 像素去畸变器
class PixelUndistorter
{
public:
    using SharedPtr = std::shared_ptr<PixelUndistorter>;
    using Grid = std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>;

    enum class Axis
    {
        XAxis,
        YAxis
    };

    /// 像素去畸变的配置信息
    struct Config
    {
        using SharedPtr = std::shared_ptr<Config>;
        using DistortionT = undistort::DistortionT;
        using DistortedParams = undistort::DistortedParams;

        /// 基于yaml的构造函数
        Config(const std::string &yaml_path)
        {
            YAML::Node info = YAML::LoadFile(yaml_path);

            if (!info)
                throw std::runtime_error("yaml file not found");

            LoadCameraInfo(info);
        }

        /// 去畸变函数部分
        virtual Eigen::Vector2f Distort(const Eigen::Vector2f &undistorted_point) = 0;

        /// 加载相机的配置文件
        void LoadCameraInfo(const YAML::Node &info)
        {
            source_K_.fx = info["CameraIntrinsics"]["Fx"].as<float>();
            source_K_.fy = info["CameraIntrinsics"]["Fy"].as<float>();
            source_K_.cx = info["CameraIntrinsics"]["Cx"].as<float>();
            source_K_.cy = info["CameraIntrinsics"]["Cy"].as<float>();

            source_size_.x() = info["ImageSize"]["SourceImgSz"]["Width"].as<int>();
            source_size_.y() = info["ImageSize"]["SourceImgSz"]["Height"].as<int>();
            target_size_.x() = info["ImageSize"]["TargetImgSz"]["Width"].as<int>();
            target_size_.y() = info["ImageSize"]["TargetImgSz"]["Height"].as<int>();

            max_adjust_iters_ = info["MaxIterations"].as<int>();

            LoadDistortedParams(info);
        }

        /// 加载相机的畸变参数
        virtual void LoadDistortedParams(const YAML::Node &info)
        {
            auto distorted_params = info["DistortedMessage"]["DistortedMode"].as<std::vector<float>>();
            distorted_params_->LoadParams(distorted_params);
        }

        virtual ~Config() = default;

        Eigen::Vector2i target_size_;                 ///< 指定虚拟的图像尺寸（w, h）
        Eigen::Vector2i source_size_;                 ///< 实际输入的图像尺寸（w, h）
        K source_K_;                                  ///< 畸变图像的相机内参
        DistortionT distorted_type_;                  ///< 相机畸变类型
        DistortedParams::SharedPtr distorted_params_; ///< 畸变参数
        int max_adjust_iters_;                        ///< 坐标轴最大调整次数
    };

    /// Pinhole 无畸变配置信息
    struct PHConfig : public Config
    {
        PHConfig(const std::string &yaml_path)
            : Config(yaml_path)
        {
        }

        Eigen::Vector2f Distort(const Eigen::Vector2f &undistorted_point) override { return undistorted_point; }

        void LoadDistortedParams(const YAML::Node &info)
        {
            distorted_type_ = undistort::DistortionT::Pinhole;
            distorted_params_ = std::make_shared<undistort::PinholeParams>();

            Config::LoadDistortedParams(info);
        }
    };

    /// RadTan 3参数畸变模型配置
    struct RT3Config : public Config
    {
        RT3Config(const std::string &yaml_path)
            : Config(yaml_path)
        {
        }

        Eigen::Vector2f Distort(const Eigen::Vector2f &undistorted_point) override;

        void LoadDistortedParams(const YAML::Node &info)
        {
            distorted_type_ = undistort::DistortionT::RadTan3;
            distorted_params_ = std::make_shared<undistort::RadTanParams<3>>();

            Config::LoadDistortedParams(info);
        }
    };

    /// RadTan 5参数畸变模型配置
    struct RT5Config : public Config
    {
        RT5Config(const std::string &yaml_path)
            : Config(yaml_path)
        {
        }

        Eigen::Vector2f Distort(const Eigen::Vector2f &undistorted_point) override;

        void LoadDistortedParams(const YAML::Node &info)
        {
            distorted_type_ = undistort::DistortionT::RadTan5;
            distorted_params_ = std::make_shared<undistort::RadTanParams<5>>();

            Config::LoadDistortedParams(info);
        }
    };

    /// FOV 畸变模型配置
    struct FOVConfig : public Config
    {
        FOVConfig(const std::string &yaml_path)
            : Config(yaml_path)
        {
        }

        Eigen::Vector2f Distort(const Eigen::Vector2f &undistorted_point) override;

        void LoadDistortedParams(const YAML::Node &info)
        {
            distorted_type_ = undistort::DistortionT::FOV;
            distorted_params_ = std::make_shared<undistort::FOVParams>();

            Config::LoadDistortedParams(info);
        }
    };

    /// KB 畸变模型配置
    struct KBConfig : public Config
    {
        KBConfig(const std::string &yaml_path)
            : Config(yaml_path)
        {
        }

        Eigen::Vector2f Distort(const Eigen::Vector2f &undistorted_point) override;

        void LoadDistortedParams(const YAML::Node &info)
        {
            distorted_type_ = undistort::DistortionT::KB;
            distorted_params_ = std::make_shared<undistort::KBParams>();

            Config::LoadDistortedParams(info);
        }
    };

    PixelUndistorter(Config::SharedPtr config)
        : config_(std::move(config))
    {
        assert(config_ && "config_ is nullptr");

        Initialize();
    }

    /**
     * @brief 对给定的畸变图像进行像素位置去畸变处理
     *
     * 本函数通过重映射方法对输入的畸变图像进行去畸变，以校正图像中的径向和切向畸变
     * 它首先检查自身是否已正确初始化，然后根据输入图像的类型执行相应的处理
     * 如果输入图像不是单通道或三通道的32位浮点数图像，则抛出异常
     * 如果输入图像为空，同样抛出异常
     * 对于多通道图像，将其分割为单个通道进行处理，对于单通道图像直接进行处理
     * 使用双线性插值方法进行重映射，以生成最终的去畸变图像
     *
     * @param distorted_img 畸变的输入图像
     * @return cv::Mat 去畸变后的图像
     * @throws std::runtime_error 如果未初始化或输入图像类型不匹配或输入图像为空
     * @see Initialize() 用于初始化去畸变器
     * @see BuildRemap() 用于构建重映射表
     * @see interp::BilinInterpCVMat() 用于双线性插值
     */
    cv::Mat Undistort(const cv::Mat &distorted_img)
    {
        if (!is_inited_)
            throw std::runtime_error("PixelUndistorter is not initialized");

        if (distorted_img.type() != CV_32FC1 || distorted_img.type() != CV_32FC3)
            throw std::runtime_error("Input image type must be CV_32FC1 or CV_32FC3");

        if (distorted_img.empty())
            throw std::runtime_error("Input image is empty");

        std::vector<cv::Mat> distorted_channels;

        if (distorted_img.channels() != 1)
            cv::split(distorted_img, distorted_channels);
        else
            distorted_channels.push_back(distorted_img);

        std::vector<cv::Mat> undistorted_channels(distorted_channels.size());
        std::vector<int> indices(distorted_channels.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::for_each(indices.begin(), indices.end(),
                      [&](const int &idx)
                      {
                          cv::Mat result = interp::BilinInterpCVMat(distorted_channels[idx], remap_x_, remap_y_);
                          undistorted_channels[idx] = result;
                      });
        cv::Mat undistorted_image;
        if (distorted_img.channels() != 1)
            cv::merge(undistorted_channels, undistorted_image);
        else
            undistorted_image = undistorted_channels[0];
        return undistorted_image;
    }

    void PinholeProcess()
    {
        /// 处理fx，fy,cx,cy
        float u_ratio = config_->target_size_[0] / config_->source_size_[0];
        float v_ratio = config_->target_size_[1] / config_->source_size_[1];
        target_K_.fx = u_ratio * config_->source_K_.fx;
        target_K_.fy = v_ratio * config_->source_K_.fy;
        target_K_.cx = u_ratio * config_->source_K_.cx;
        target_K_.cy = v_ratio * config_->source_K_.cy;

        /// 处理remap
        BuildRemap();
    }

    /**
     * @brief 初始化像素去畸变器
     * @details
     *      1. 根据输入的畸变参数信息和虚拟图像的尺寸，计算出虚拟相机的内参fx, fy, cx, cy
     *      2. 根据虚拟相机的内参、相机畸变参数和实际相机内参，计算虚拟图形和真实图形之间的映射表
     */
    void Initialize()
    {
        ComputeTargetK();
        BuildRemap();
        is_inited_ = true;
    }

    /// 根据得到的轴位置推算出无畸变图像的虚拟内参
    void ComputeTargetK();

    /// 构建映射关系表 remap_x_ 和 remap_y_
    void BuildRemap();

private:
    /// 找到所有极限轴上的点，都能投影到畸变图像上的归一化坐标轴位置
    void ComputeRealAxis(float &x_max, float &y_max, float &x_min, float &y_min);

    /// 判断某个轴上的点是否都能投影到畸变图像上
    bool MatchAxisProject(float &axis_value, Axis axis_direction, const float &other_min_value, const float &other_max_value, const bool &axis_flag);

    /// 获取投影极限位置的坐标轴 lx_max，ly_max，lx_min，ly_min
    void ComputeLimitAxis(float &lx_max, float &ly_max, float &lx_min, float &ly_min);

    /// 在无畸变归一化坐标系上，找到2 * 10001个点
    std::vector<Grid> CreateMeshgrid();

    Config::SharedPtr config_; ///< 像素去畸变的配置信息
    bool is_inited_;           ///< 是否完成了初始化
    K target_K_;               ///< 无畸变图像的相机内参
    cv::Mat remap_x_;          ///< 无畸变图像坐标存储的畸变图像x映射
    cv::Mat remap_y_;          ///< 无畸变图像坐标存储的畸变图像y映射
};

} // namespace dso_ssl
