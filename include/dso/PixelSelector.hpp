#pragma once

#include <algorithm>
#include <memory>
#include <random>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace dso_ssl
{

/// 像素选择器
class PixelSelector
{
public:
    using SharedPtr = std::shared_ptr<PixelSelector>;
    using Vector2iArray = std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>>;

    struct Options
    {
        using SharedPtr = std::shared_ptr<Options>;

        /// 读取yaml文件的构造函数
        Options(std::string file_path);

        int first_layer_init_pot_;       ///< 第一层金字塔初始化的pot大小
        float down_sacle_factor_;        ///< 梯度阈值缩放比例
        float grad_thresh_bias_;         ///< 计算像素块的梯度偏移
        float grad_thresh_percentile_;   ///< 计算像素块的梯度百分比
        int patch_size_;                 ///< 图像块大小
        int first_recu_depth_;           ///< 第0层选点时最大递归深度
        int other_recu_depth_;           ///< 其他层选点时最大递归深度
        int other_layers_pot_size_;      ///< 其他层提取点时，pot的初始大小
        float other_layers_grad_thresh_; ///< 金字塔其他层上选点梯度阈值
        float other_layers_factor_th_;   ///< 其他层选点梯度阈值缩放比例
    };

    PixelSelector(Options::SharedPtr config)
        : config_(std::move(config))
    {
    }

    /// 构建图像块梯度阈值
    void BuildGradientThreshold(const cv::Mat &image);

    /// 做图像块梯度平滑
    void SoomthGradientThreshold();

    /// 指定pot大小，在金字塔第一层上选择点
    void SelectFirstLayerInternal(const int &potinal, const cv::Mat &image_grad_squre, std::vector<int> &selected_points_conf,
                                  Vector2iArray &selected_points_out);

    /// 在金字塔第一层上选择点
    void SelectFirstLayer(const int &numwant, const cv::Mat &image_grad_squre, Vector2iArray &selected_points, std::vector<int> &selected_points_conf);

    /// 指定pot大小，在金字塔其他层上选择点
    void SelectOtherLayerInternal(const int &potinal, const cv::Mat &image_and_grads, Vector2iArray &selected_points, float factor_th);

    /// 在金字塔其他层上选择点
    void SelectOtherLayer(const int &numwant, const cv::Mat &image_and_grads, Vector2iArray &selected_points);

private:
    Options::SharedPtr config_;  ///< 点选配置
    cv::Mat grad_thresh_;       ///< 图像块梯度阈值
    int row_start_, col_start_; ///< 对grad_thresh_的补充
    int row_end_, col_end_;     ///< 对grad_thresh_的补充
};

} // namespace dso_ssl
