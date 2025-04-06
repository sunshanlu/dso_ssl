#pragma once

#include <sstream>
#include <thread>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <slam_viewer/core/ImageShower.h>
#include <slam_viewer/core/PointTypes.h>
#include <slam_viewer/core/WindowImpl.h>
#include <slam_viewer/ui/CloudUI.hpp>
#include <slam_viewer/ui/CoordinateUI.h>
#include <slam_viewer/ui/FrameUI.h>
#include <slam_viewer/ui/TrajectoryUI.h>

#include "dso/Frame.hpp"
#include "dso/Initializer2.hpp"
#include "utils/Project.hpp"

namespace dso_ssl
{

/**
 * @brief 构建可视化器
 *
 * 1. 可视化点云地图
 * 2. 可视化待跟踪帧（包括跟踪成功的点）
 * 3. 可视化跟踪参考帧
 */
class Visualizer
{
public:
  using SharedPtr = std::shared_ptr<Visualizer>;
  using PointT = pcl::PointXYZRGB;
  using CloudT = pcl::PointCloud<PointT>;
  using ThreadPtr = std::shared_ptr<std::thread>;

  struct Options
  {
    using SharedPtr = std::shared_ptr<Options>;

    /// 可视化器配置构造
    Options(const std::string &filepath);

    float world_coor_length_;
    float camera_coor_length_;
    float trajectory_line_width_;
    float trajectory_point_size_;
    slam_viewer::Vec3 trajectory_color_;
    int window_width_;
    int window_height_;
    int image_shower_width_;
    int image_shower_height_;
  };

  /**
   * @brief 可视化器
   *
   * 1. 创建窗口
   * 2. 创建相机
   * 3. 创建菜单
   * 4. 创建3d显示、图像显示区域
   * 5. 创建世界坐标系、相机坐标系和相机轨迹
   */
  Visualizer(Options::SharedPtr options);

  /**
   * @brief 开启窗口可视化线程
   *
   * 考虑了重复调用时的情况，不会盲目多开启很多的线程
   */
  void Run()
  {
    if (!runing_thread_)
      runing_thread_ = std::make_shared<std::thread>(&slam_viewer::WindowImpl::Run, window_impl_);
  }

  /**
   * @brief 停止窗口可视化线程
   *
   * 当函数执行完毕时，窗口可视化线程停止成功，可以在停止后继续执行run函数
   */
  void Stop()
  {
    window_impl_->RequestStop();

    if (runing_thread_)
      runing_thread_->join();

    runing_thread_ = nullptr;
  }

  /**
   * @brief 等待窗口可视化线程
   *
   */
  void Join()
  {
    if (runing_thread_)
      runing_thread_->join();
  }

  /**
   * @brief 更新参考帧，初始化线程或者跟踪线程调用
   *
   * 这里是线程安全的，因为该函数的调用发生在初始化线程或者跟踪线程中，因此不需要加锁
   *
   * @param ref_frame     输入的参考帧
   * @param idepth_points 输入的参考帧对应的0层逆深度点
   */
  void UpdateReferenceFrame(const Frame::SharedPtr &ref_frame, const std::vector<InitIdepthPoint::SharedPtr> &idepth_points);

  /**
   * @brief 更新跟踪帧，初始化线程或者跟踪线程调用
   *
   * @param cur_frame 输入的待跟踪帧
   */
  void UpdateTrackingFrame(const Frame::SharedPtr &cur_frame);

  /**
   * @brief 配置菜单内容，用户ui交互
   *
   * 1 follow 相机跟踪模式
   * 2 设置上帝视角
   * 3 设置前置视角
   */
  void ConfigMenu();

  /**
   * @brief 更新参考帧逆深度点
   *
   * 在初始化过程中，用于更新参考关键帧的逆深度点的可视化行为
   * 1. 将那些没有被当前帧优化的逆深度点置为灰色
   * 2. 将那些被当前帧优化的逆深度点置为红色
   * 3. 这里默认为初始化器的参考关键帧为世界坐标系
   *
   * @param idepth_points 逆深度点
   * @param fx            fx
   * @param fy            fy
   * @param cx            cx
   * @param cy            cy
   */
  void UpdateInitializerCloud(const std::vector<InitIdepthPoint::SharedPtr> &idepth_points, const float &fx, const float &fy, const float &cx, const float &cy);

private:
  Options::SharedPtr options_; ///< 可视化器参数配置

  slam_viewer::WindowImpl::Ptr window_impl_;   ///< 可视化窗口
  slam_viewer::Camera::Ptr camera_;            ///< 相机句柄
  slam_viewer::View3D::Ptr view_3d_;           ///< 3d视图
  slam_viewer::Menu::Ptr view_menu_;           ///< 菜单视图
  slam_viewer::ImageShower::Ptr image_shower_; ///< 图像显示器

  slam_viewer::CoordinateUI::Ptr world_coordinate_;  ///< 世界坐标系
  slam_viewer::CoordinateUI::Ptr camera_coordinate_; ///< 相机坐标系
  slam_viewer::TrajectoryUI::Ptr camera_trajectory_; ///< 相机轨迹
  slam_viewer::CloudUI::Ptr initializer_cloud_;      ///< 初始化器的点云

  ThreadPtr runing_thread_; ///< 运行的窗口线程
};

} // namespace dso_ssl