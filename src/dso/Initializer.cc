#include <execution>
#include <filesystem>
#include <mutex>

#include "dso/Initializer.hpp"

namespace dso_ssl
{

/**
 * @brief 根据输入的参考帧信息，构建初始化器的图像金字塔信息
 *
 * 1. 构建相邻层的父子点关系，以便使用高斯归一化积进行逆深度传播
 * 2. 构建同一层的相邻关系，以便考虑相邻点之间的逆深度关系
 *
 * @param frame_ptr 输入的参考帧
 */
void Initializer::SetReferenceFrame(Frame::SharedPtr frame_ptr)
{
    if (ref_frame_)
        return;

    if (!frame_ptr)
        throw std::runtime_error("SetReferenceFrame: frame_ptr is nullptr");

    if (!pattern_)
        throw std::runtime_error("SetReferenceFrame: pattern_ is nullptr");

    if (!pixel_selector_)
        throw std::runtime_error("SetReferenceFrame: pixel_selector_ is nullptr");

    auto image_and_grads = frame_ptr->GetPyrdImageAndGrads();
    auto squre_grad = frame_ptr->GetSqureGrad();

    ref_layer_info_.clear();
    ref_layer_info_.resize(config_->pyra_levels_);
    std::vector<pcl::KdTreeFLANN<PointT>::Ptr> pyra_kdtrees(config_->pyra_levels_, nullptr);
    for (int nlevel = 0; nlevel < config_->pyra_levels_; ++nlevel)
    {
        // 维护layer_frame的层级信息
        LayerFrame::SharedPtr layer_frame = std::make_shared<LayerFrame>();
        layer_frame->nlevel_ = nlevel;

        // 维护layer_frame的图像和梯度信息
        std::vector<cv::Mat> image_and_grads_vec;
        cv::split(image_and_grads[nlevel], image_and_grads_vec);
        layer_frame->layer_image_ = image_and_grads_vec[0];
        layer_frame->layer_gradx_ = image_and_grads_vec[1];
        layer_frame->layer_grady_ = image_and_grads_vec[2];

        // 像素选择器提取像素位置
        int select_num = config_->select_densities_[nlevel] * image_and_grads[nlevel].rows * image_and_grads[nlevel].cols;
        PixelSelector::Vector2iArray selected_points;
        if (nlevel == 0)
        {
            std::vector<int> selected_points_conf;
            pixel_selector_->SelectFirstLayer(select_num, squre_grad, selected_points, selected_points_conf);
        }
        else
            pixel_selector_->SelectOtherLayer(select_num, image_and_grads[nlevel], selected_points);

        // 由于pattern的作用，需要进一步处理selected_points里面不合法的像素点
        decltype(selected_points) temp;
        auto half_pattern_size = pattern_->GetHalfPatternSize();
        for (const auto &point : selected_points)
        {
            if (point[0] > half_pattern_size && point[0] < image_and_grads[nlevel].cols - 4 && point[1] > half_pattern_size &&
                point[1] < image_and_grads[nlevel].rows - 4)
                temp.push_back(point);
        }
        std::swap(selected_points, temp);
        temp.clear();

        // 维护金字塔层级中的像素点信息，相邻关系和父子关系
        BuildNeighborRelation(layer_frame, selected_points, pyra_kdtrees[nlevel]);
        if (nlevel > 0)
            BuildParentChildRelation(layer_frame, ref_layer_info_[nlevel - 1], pyra_kdtrees[nlevel]);

        ref_layer_info_[nlevel] = layer_frame;
    }
}

/**
 * @brief 构建同层金字塔提取点之间的相邻关系
 *
 * 0. 维护LayerFrame::pixel_points_
 * 1. 构建pcl的KD-Tree结构，维护当前点的信息
 * 2. 然后遍历点，试图找到相邻点，并维护相邻关系
 *
 * @param layer_kdtree      输出的金字塔层级的kdtree信息
 * @param layer_frame       输入输出的金字塔层级信息
 * @param selected_points   输入的选择点信息
 */
void Initializer::BuildNeighborRelation(LayerFrame::SharedPtr &layer_frame, const PixelSelector::Vector2iArray &selected_points, KdTree2d::Ptr &layer_kdtree)
{
    CloudT::Ptr layer_cloud = pcl::make_shared<CloudT>();

    layer_cloud->resize(selected_points.size());
    layer_frame->pixel_points_.resize(selected_points.size(), nullptr);
    std::vector<int> indices(selected_points.size());
    std::iota(indices.begin(), indices.end(), 0);

    // 构建点云layer_cloud，并填充layer_frame的像素信息
    auto point_process = [&](const int &idx)
    {
        PointT p;
        p.x = static_cast<float>(selected_points[idx][0]);
        p.y = static_cast<float>(selected_points[idx][1]);
        layer_cloud->points[idx] = p;

        auto pixel_point = std::make_shared<PixelPoint>();
        pixel_point->id_ = idx;
        pixel_point->nlevel_ = layer_frame->nlevel_;
        pixel_point->pixel_position_ = selected_points[idx].cast<float>();
        layer_frame->pixel_points_[idx] = pixel_point;
    };

    // 构建和当前层config_->neighbor_nums_个最近临点之间的关系
    auto neighbor_process = [&](const int &idx)
    {
        std::vector<float> dists;
        std::vector<int> neightbor_idx;
        if (layer_kdtree->nearestKSearch(layer_cloud->points[idx], config_->neighbor_nums_ + 1, neightbor_idx, dists) != config_->neighbor_nums_ + 1)
            throw std::runtime_error("neighbor search error");

        for (int i = 1; i < config_->neighbor_nums_ + 1; ++i)
            layer_frame->pixel_points_[idx]->neighbor_ids_.push_back(neightbor_idx[i]);
    };

    // 并行处理构建点云，并维护kdtree信息
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), point_process);
    layer_kdtree = pcl::make_shared<KdTree2d>();
    layer_kdtree->setInputCloud(layer_cloud);

    // 寻找邻居点
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), neighbor_process);
}

/**
 * @brief 构建相邻层金字塔提取点之间的父子关系
 *
 * 1. 将 last_layer_frame 上的点，投影到 cur_layer_frame 中
 * 2. 根据 cur_kdtrees 和 投影点 在cur_layer_frame找到最近点
 * 3. 维护父子关系
 *
 * @param cur_layer_frame       输入输出的当前层金字塔层级信息
 * @param last_layer_frame      输入输出的上一层金字塔层级信息
 * @param cur_kdtrees           输入的当前层金字塔层级的kdtree信息
 */
void Initializer::BuildParentChildRelation(LayerFrame::SharedPtr &cur_layer_frame, LayerFrame::SharedPtr &last_layer_frame, const KdTree2d::Ptr &cur_kdtrees)
{
    int last_points_num = last_layer_frame->pixel_points_.size();
    int curr_points_num = cur_layer_frame->pixel_points_.size();

    std::vector<int> indices(last_points_num, 0);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<std::mutex> cur_mutexes(curr_points_num);

    auto parent_child_process = [&](const int &idx)
    {
        // 将 last_layer_frame 上的点，投影到 cur_layer_frame 中
        PointT pc;
        auto &pl = last_layer_frame->pixel_points_[idx];
        pc.x = pl->pixel_position_[0] / 2 - 0.5;
        pc.y = pl->pixel_position_[1] / 2 - 0.5;

        // 搜索最近邻居点
        std::vector<int> k_indices;
        std::vector<float> k_distances;
        if (cur_kdtrees->nearestKSearch(pc, 1, k_indices, k_distances) != 1)
            throw std::runtime_error("kdtree search error");

        // 维护父子关系
        pl->parent_id_ = k_indices[0];
        {
            std::lock_guard<std::mutex> lock(cur_mutexes[k_indices[0]]);
            cur_layer_frame->pixel_points_[k_indices[0]]->children_ids_.push_back(idx);
        }
    };

    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), parent_child_process);
}

/**
 * @brief 初始化器配置构造
 *
 * @param config_path 输入的初始化器配置文件路径
 */
Initializer::Config::Config(const std::string &config_path)
{
    if (!std::filesystem::exists(config_path))
        throw std::runtime_error("配置文件不存在");

    auto info = YAML::LoadFile(config_path);
    pyra_levels_ = info["PyraidLevelsUsed"].as<int>();
    neighbor_nums_ = info["NeighborNum"].as<int>();
    select_densities_ = info["SelectDensities"].as<std::vector<float>>();
}

} // namespace dso_ssl
