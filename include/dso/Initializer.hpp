#pragma once
#include <memory>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "dso/Frame.hpp"
#include "dso/Pattern.hpp"
#include "dso/PixelSelector.hpp"

namespace dso_ssl
{

/// dso系统的初始化器
class Initializer
{
public:
    using PointT = pcl::PointXY;
    using CloudT = pcl::PointCloud<PointT>;
    using KdTree2d = pcl::KdTreeFLANN<PointT>;
    using SharedPtr = std::shared_ptr<Initializer>;
    using Vector2f = Eigen::Vector2f;
    using Vector2fArray = std::vector<Vector2f, Eigen::aligned_allocator<Vector2f>>;

    struct Config
    {
        using SharedPtr = std::shared_ptr<Config>;

        Config(const std::string &config_path);

        int pyra_levels_;                     ///< 使用的图像金字塔层级
        int neighbor_nums_;                   ///< 邻居点的数量
        std::vector<float> select_densities_; ///< 选择的点密度
    };

    /// 像素点，维护父节点和相邻节点的关系信息
    struct PixelPoint
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using SharedPtr = std::shared_ptr<PixelPoint>;

        int id_;                        ///< 当前点的id
        int nlevel_;                    ///< 所在的金字塔层级
        int parent_id_;                 ///< 上一层父节点id，可以为当点提供优化初值
        std::vector<int> children_ids_; ///< 下一层子节点id，可以使用高斯归一化更新当前逆深度
        std::vector<int> neighbor_ids_; ///< 邻居节点id，用于联系当前层相邻点关系
        Vector2f pixel_position_;       ///< 当前层的像素值

        float idepth_;         ///< 逆深度值
        float idepth_avg_;     ///< 考虑了相邻点关系的逆深度值
        float idepth_hessian_; ///< 逆深度的Hessian值
    };

    /// 金字塔层级图像内容维护
    struct LayerFrame
    {
        using SharedPtr = std::shared_ptr<LayerFrame>;

        int nlevel_;                                      ///< 金字塔层级
        cv::Mat layer_image_;                             ///< 金字塔层级图像
        cv::Mat layer_gradx_;                             ///< 金字塔层级图像的x方向梯度
        cv::Mat layer_grady_;                             ///< 金字塔层级图像的y方向梯度
        std::vector<PixelPoint::SharedPtr> pixel_points_; ///< 金字塔层级中的像素点信息
    };

    Initializer(Config::SharedPtr config)
        : config_(std::move(config))
        , initialized_(false)
        , ref_frame_(nullptr)
        , pixel_selector_(nullptr)
        , pattern_(nullptr)
    {
    }

    Initializer(Config::SharedPtr config, PixelSelector::SharedPtr pixel_selector, Pattern::SharedPtr pattern)
        : config_(std::move(config))
        , initialized_(false)
        , ref_frame_(nullptr)
        , pixel_selector_(std::move(pixel_selector))
        , pattern_(pattern)
    {
    }

    const std::vector<LayerFrame::SharedPtr> &GetRefLayerInfo() const { return ref_layer_info_; }

    /**
     * @brief 向初始化器中添加帧，试图完成初始化过程
     *
     * 要求连续k帧，满足tji的阈值条件，才认为初始化成功
     *
     * @param frame_ptr 输入的帧
     * @return true     初始化成功
     * @return false    初始化失败
     */
    bool AddActivateFrame(Frame::SharedPtr frame_ptr);

    /**
     * @brief 根据输入的参考帧信息，构建初始化器的图像金字塔信息
     *
     * 1. 构建相邻层的父子点关系，以便使用高斯归一化积进行逆深度传播
     * 2. 构建同一层的相邻关系，以便考虑相邻点之间的逆深度关系
     *
     * @param frame_ptr 输入的参考帧
     */
    void SetReferenceFrame(Frame::SharedPtr frame_ptr);

    /**
     * @brief 构建同层金字塔提取点之间的相邻关系
     *
     * @param layer_frame       输入输出的金字塔层级信息
     * @param selected_points   输入的选择点信息
     * @param layer_kdtree      输出的金字塔层级的kdtree信息
     */
    void BuildNeighborRelation(LayerFrame::SharedPtr &layer_frame, const PixelSelector::Vector2iArray &selected_points, KdTree2d::Ptr &layer_kdtree);

    /**
     * @brief 构建相邻层金字塔提取点之间的父子关系
     *
     * @param cur_layer_frame       输入输出的当前层金字塔层级信息
     * @param last_layer_frame      输入输出的上一层金字塔层级信息
     * @param cur_kdtrees           输入的当前层金字塔层级的kdtree信息
     */
    void BuildParentChildRelation(LayerFrame::SharedPtr &cur_layer_frame, LayerFrame::SharedPtr &last_layer_frame, const KdTree2d::Ptr &cur_kdtrees);

    /**
     * @brief 针对输入的帧，构建初始化器约束，进行优化
     *
     * 1. 当位移tcr 小于阈值时，使用额外的逆深度1约束和位移tcr的模约束，这里可以保证tcr的鲁棒性
     * 2. 当位移tcr 大于阈值时，使用使用idepth_avg_对逆深度进行约束
     * 3. 注意，当tcr优化没有满足阈值条件时，仅保留优化的旋转要求
     */
    void Optimize(Frame::SharedPtr frame_ptr);

private:
    Config::SharedPtr config_;                          ///< 初始化器配置信息
    bool initialized_;                                  ///< 初始化过程是否完成
    Frame::SharedPtr ref_frame_;                        ///< 初始化器中的参考帧
    std::vector<LayerFrame::SharedPtr> ref_layer_info_; ///< 参考帧的图像金字塔信息
    PixelSelector::SharedPtr pixel_selector_;           ///< 参考帧的像素选择器
    Pattern::SharedPtr pattern_;                        ///< 初始化器构建残差时，使用的pattern
};

} // namespace dso_ssl