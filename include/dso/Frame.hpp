#pragma once

#include <memory>

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include "dso/PhotoAffine.hpp"
#include "dso/Undistorter.hpp"

namespace dso_ssl
{

/// 帧核心
struct FrameKernel
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using SharedPtr = std::shared_ptr<FrameKernel>;

    Sophus::SE3f T_cw_;     ///< 当前帧的位姿
    PhotoAffine affine_ab_; ///< 当前帧的光度仿射参数
    double timestamp_;      ///< 帧的时间戳
    float exposure_time_;   ///< 帧的曝光时间
};

/// 定义DSO中的普通帧
class Frame
{
public:
    using SharedPtr = std::shared_ptr<Frame>;

    friend void ShowPyraidImagesAndGrads(Frame::SharedPtr frameptr); // 展示图像金字塔，供测试使用

    struct Config
    {
        using SharedPtr = std::shared_ptr<Config>;

        Config(std::string file_path)
        {
            auto info = YAML::LoadFile(file_path);
            pyra_levels_ = info["PyraidLevelsUsed"].as<int>();
            use_origin_grad_ = info["UseOriginGrad"].as<bool>();
        }

        int pyra_levels_;      ///< 图像金字塔层数
        bool use_origin_grad_; ///< 梯度平方和是否使用原图的梯度
    };

    Frame(const Config::SharedPtr &config, const cv::Mat &first_layer_image, const cv::Mat &only_pixel_undistorted_image);

private:
    /// 构造图像金字塔 --> 4合1 + 均值滤波
    void MakePyrdImages(const cv::Mat &first_layer_image);

    /// 计算第0层梯度平方和，用于后续第0层的点选操作
    void MakeSqureGrad(const cv::Mat &only_pixel_undistorted_image);

    Config::SharedPtr config_;                  ///< Frame的配置信息
    FrameKernel::SharedPtr frame_kernel_;       ///< 保存的帧核心参数信息
    std::vector<cv::Mat> pyrd_image_and_grads_; ///< 维护的图像金字塔上的图像和梯度信息
    cv::Mat image_and_grad_;                    ///< 金字塔第0层图像和梯度信息
    cv::Mat squre_grad_;                        ///< 梯度平方和
};

} // namespace dso_ssl
