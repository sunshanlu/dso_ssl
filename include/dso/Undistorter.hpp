#pragma once

#include <algorithm>
#include <execution>
#include <filesystem>
#include <memory>
#include <numeric>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <tbb/parallel_for_each.h>
#include <yaml-cpp/yaml.h>

#include "utils/Interpolate.hpp"
#include "utils/MatOperators.hpp"
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
    struct Options
    {
        using SharedPtr = std::shared_ptr<Options>;
        using DistortionT = undistort::DistortionT;
        using DistortedParams = undistort::DistortedParams;

        /// 基于yaml的构造函数
        Options(const std::string &yaml_path)
        {
            info_ = std::make_shared<YAML::Node>(YAML::LoadFile(yaml_path));

            if (!info_)
                throw std::runtime_error("yaml file not found");
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
            auto distorted_params = info["DistortedMessage"]["DistortedParams"].as<std::vector<float>>();
            distorted_params_->LoadParams(distorted_params);
        }

        virtual ~Options() = default;

        Eigen::Vector2i target_size_;                 ///< 指定虚拟的图像尺寸（w, h）
        Eigen::Vector2i source_size_;                 ///< 实际输入的图像尺寸（w, h）
        K source_K_;                                  ///< 畸变图像的相机内参
        DistortionT distorted_type_;                  ///< 相机畸变类型
        DistortedParams::SharedPtr distorted_params_; ///< 畸变参数
        int max_adjust_iters_;                        ///< 坐标轴最大调整次数
        std::shared_ptr<YAML::Node> info_;            ///< yaml配置文件信息
    };

    /// Pinhole 无畸变配置信息
    struct PHConfig : public Options
    {
        PHConfig(const std::string &yaml_path)
            : Options(yaml_path)
        {
            LoadCameraInfo(*info_);
            info_ = nullptr;
        }

        Eigen::Vector2f Distort(const Eigen::Vector2f &undistorted_point) override { return undistorted_point; }

        void LoadDistortedParams(const YAML::Node &info) override
        {
            distorted_type_ = undistort::DistortionT::Pinhole;
            distorted_params_ = std::make_shared<undistort::PinholeParams>();

            Options::LoadDistortedParams(info);
        }
    };

    /// RadTan 3参数畸变模型配置
    struct RT3Config : public Options
    {
        RT3Config(const std::string &yaml_path)
            : Options(yaml_path)
        {
            LoadCameraInfo(*info_);
            info_ = nullptr;
        }

        Eigen::Vector2f Distort(const Eigen::Vector2f &undistorted_point) override;

        void LoadDistortedParams(const YAML::Node &info) override
        {
            distorted_type_ = undistort::DistortionT::RadTan3;
            distorted_params_ = std::make_shared<undistort::RadTanParams<3>>();

            Options::LoadDistortedParams(info);
        }
    };

    /// RadTan 5参数畸变模型配置
    struct RT5Config : public Options
    {
        RT5Config(const std::string &yaml_path)
            : Options(yaml_path)
        {
            LoadCameraInfo(*info_);
            info_ = nullptr;
        }

        Eigen::Vector2f Distort(const Eigen::Vector2f &undistorted_point) override;

        void LoadDistortedParams(const YAML::Node &info) override
        {
            distorted_type_ = undistort::DistortionT::RadTan5;
            distorted_params_ = std::make_shared<undistort::RadTanParams<5>>();

            Options::LoadDistortedParams(info);
        }
    };

    /// FOV 畸变模型配置
    struct FOVConfig : public Options
    {
        FOVConfig(const std::string &yaml_path)
            : Options(yaml_path)
        {
            LoadCameraInfo(*info_);
            info_ = nullptr;
        }

        Eigen::Vector2f Distort(const Eigen::Vector2f &undistorted_point) override;

        void LoadDistortedParams(const YAML::Node &info) override
        {
            distorted_type_ = undistort::DistortionT::FOV;
            distorted_params_ = std::make_shared<undistort::FOVParams>();

            Options::LoadDistortedParams(info);
        }
    };

    /// KB 畸变模型配置
    struct KBConfig : public Options
    {
        KBConfig(const std::string &yaml_path)
            : Options(yaml_path)
        {
            LoadCameraInfo(*info_);
            info_ = nullptr;
        }

        Eigen::Vector2f Distort(const Eigen::Vector2f &undistorted_point) override;

        void LoadDistortedParams(const YAML::Node &info) override
        {
            distorted_type_ = undistort::DistortionT::KB;
            distorted_params_ = std::make_shared<undistort::KBParams>();

            Options::LoadDistortedParams(info);
        }
    };

    PixelUndistorter(Options::SharedPtr config)
        : config_(std::move(config))
    {
        assert(config_ && "config_ is nullptr");

        Initialize();
    }

    /// 获取u坐标映射
    const cv::Mat &GetRemapX() const { return remap_x_; }

    /// 获取v坐标映射
    const cv::Mat &GetRemapY() const { return remap_y_; }

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

        if (!(distorted_img.type() == CV_32FC1 || distorted_img.type() == CV_32FC3))
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

    void GetTargetK(float &fx, float &fy, float &cx, float &cy)
    {
        fx = target_K_.fx;
        fy = target_K_.fy;
        cx = target_K_.cx;
        cy = target_K_.cy;
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

    /// 使用remapx和remapy获取畸变坐标系下的像素坐标
    void GetDistortedPixel(const int &undistorted_y, const int &undistorted_x, float &distorted_y, float &distorted_x)
    {
        distorted_y = remap_y_.at<float>(undistorted_y, undistorted_x);
        distorted_x = remap_x_.at<float>(undistorted_y, undistorted_x);
    }

private:
    /// 找到所有极限轴上的点，都能投影到畸变图像上的归一化坐标轴位置
    void ComputeRealAxis(float &x_max, float &y_max, float &x_min, float &y_min);

    /// 判断某个轴上的点是否都能投影到畸变图像上
    bool MatchAxisProject(float &axis_value, Axis axis_direction, const float &other_min_value, const float &other_max_value, const bool &axis_flag);

    /// 获取投影极限位置的坐标轴 lx_max，ly_max，lx_min，ly_min
    void ComputeLimitAxis(float &lx_max, float &ly_max, float &lx_min, float &ly_min);

    /// 在无畸变归一化坐标系上，找到2 * 10001个点
    std::vector<Grid> CreateMeshgrid();

    Options::SharedPtr config_; ///< 像素去畸变的配置信息
    bool is_inited_;           ///< 是否完成了初始化
    K target_K_;               ///< 无畸变图像的相机内参
    cv::Mat remap_x_;          ///< 无畸变图像坐标存储的畸变图像x映射
    cv::Mat remap_y_;          ///< 无畸变图像坐标存储的畸变图像y映射
};

class PhotoUndistorter
{
public:
    using SharedPtr = std::shared_ptr<PhotoUndistorter>;

    /// 光度去畸变器的配置
    struct Options
    {
        using SharedPtr = std::shared_ptr<Options>;

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
        Options(std::string file_path);

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
         * @brief 计算 dG / dI
         *
         * 根据GInv函数计算
         *
         * @return float dG / dI
         */
        float ComputeGJacobian(float Ivalue);

        /**
         * @brief 根据Ginv 计算 G 函数
         *
         */
        void ComputeGFunction();

        /**
         * @brief 标准化GInv函数，使用了自定义并行化模板
         *
         * 在区间[0,255]之间分布
         */
        void NormalizeGInvParallel();

        /**
         * @brief 标准化渐晕map图
         *
         * 在区间[0, 1]之间分布
         */
        void NormalizeVignette();

        cv::Mat vignette_map_;         ///< 渐晕map图
        std::vector<float> gfunc_inv_; ///< 非线性响应函数G^-1
        std::vector<float> gfunc_;     ///< 非线性响应函数G
        bool vignette_flag_;           ///< 是否使用渐晕处理
        bool gfunc_inv_flag_;          ///< 是否使用非线性响应函数G^-1
    };

    PhotoUndistorter(Options::SharedPtr config)
        : config_(std::move(config))
    {
    }

    const cv::Mat &GetVignette() const { return config_->vignette_map_; }

    /**
     * @brief 计算 dG / d(I'v)
     *
     * 根据GInv函数计算，非线性响应函数相对于I'v的雅可比矩阵
     *
     * @return float dG / d(I'v)
     */
    float ComputeGJacobian(float Ivalue) { return config_->ComputeGJacobian(Ivalue); }

    /**
     * @brief Get the Vigneete Value object
     *
     * @param x         输入畸变像素坐标系下的x坐标
     * @param y         输入畸变像素坐标系下的y坐标
     * @return float    输出的渐晕值
     */
    const float &GetVigneeteValue(const int &x, const int &y) { return config_->vignette_map_.at<float>(y, x); }

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

    Options::SharedPtr config_; ///< 光度去畸变器配置
};

class Undistorter
{
public:
    using SharedPtr = std::shared_ptr<Undistorter>;

    struct Options
    {
        using SharedPtr = std::shared_ptr<Options>;

        Options(const std::string &file_path)
        {
            auto info = YAML::LoadFile(file_path);
            use_origin_grad_ = info["UseDistortedGrad"].as<bool>();
        }

        bool use_origin_grad_;
    };

    Undistorter(PixelUndistorter::Options::SharedPtr pixel_config, PhotoUndistorter::Options::SharedPtr photo_config, Options::SharedPtr config);

    void GetTargetK(float &fx, float &fy, float &cx, float &cy) { pixel_undistorter_->GetTargetK(fx, fy, cx, cy); }

    /**
     * @brief 去畸变器核心去畸变函数
     *
     * pixel_undistorted_image 像素去畸变图像作为中间产物输出，
     * 与DSO源码里面形成了鲜明的对比，使用先像素去畸变，然后光度去畸变过程
     *
     * @param distorted_image           输入的畸变图像
     * @param pixel_undistorted_image   输出的去像素畸变图像
     * @return cv::Mat 输出的去畸变图像
     */
    cv::Mat Undistort(const cv::Mat &distorted_image, cv::Mat &pixel_undistorted_image);

private:
    Options::SharedPtr config_;
    PixelUndistorter::SharedPtr pixel_undistorter_;
    PhotoUndistorter::SharedPtr photo_undistorter_;
};

} // namespace dso_ssl