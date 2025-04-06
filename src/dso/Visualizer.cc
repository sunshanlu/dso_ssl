#include "dso/Visualizer.hpp"

namespace dso_ssl
{

/**
 * @brief 更新参考帧，初始化线程或者跟踪线程调用
 *
 * 这里是线程安全的，因为该函数的调用发生在初始化线程或者跟踪线程中，因此不需要加锁
 *
 * @param ref_frame     输入的参考帧
 * @param idepth_points 输入的参考帧对应的0层逆深度点
 */
void Visualizer::UpdateReferenceFrame(const Frame::SharedPtr &ref_frame,
                                      const std::vector<InitIdepthPoint::SharedPtr> &idepth_points)
{
  cv::Mat images_and_grads = ref_frame->GetPyrdImageAndGrads()[0];

  std::vector<cv::Mat> image_and_grad;
  cv::split(images_and_grads, image_and_grad);

  cv::Mat image = image_and_grad[0];
  image.convertTo(image, CV_8U);
  cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

  int ntracked = 0;
  for (const auto &point: idepth_points)
  {
    if (!point->is_good_)
      continue;

    cv::Point pixel_center(point->host_pixel_positon_[0], point->host_pixel_positon_[1]);
    cv::circle(image, pixel_center, 2, cv::Scalar(0, 255, 0), -1);
    ++ntracked;
  }

  std::stringstream ss;
  ss << "Exposure: " << ref_frame->GetExposureTime() << "ms" << " Tracked Points: " << ntracked;
  std::string ref_content = ss.str();

  image_shower_->UpdateImage("Reference Frame", ref_content, image);
}

/**
 * @brief 更新跟踪帧，初始化线程或者跟踪线程调用
 *
 * @param cur_frame 输入的待跟踪帧
 */
void Visualizer::UpdateTrackingFrame(const Frame::SharedPtr &cur_frame)
{
  cv::Mat images_and_grads = cur_frame->GetPyrdImageAndGrads()[0];

  std::vector<cv::Mat> image_and_grad;
  cv::split(images_and_grads, image_and_grad);

  cv::Mat image = image_and_grad[0];
  image.convertTo(image, CV_8U);
  cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

  std::stringstream ss;
  ss << "Exposure: " << cur_frame->GetExposureTime() << "ms";
  std::string cur_content = ss.str();

  image_shower_->UpdateImage("Tracking Frame", cur_content, image);
}

/**
 * @brief 配置菜单内容，用户ui交互
 *
 * 1 follow 相机跟踪模式
 * 2 设置上帝视角
 * 3 设置前置视角
 */
void Visualizer::ConfigMenu()
{
  assert(view_menu_ && camera_ && "menu or camera is nullptr!");

  /// 1 follow 相机跟踪模式
  view_menu_->AddCheckBoxItem("Follow",
                              [=](bool checked)
                              {
                                if (checked)
                                  camera_->SetFollow();
                                else
                                  camera_->SetFree();
                              });

  /// 2 设置上帝视角
  view_menu_->AddButtonItem("God View",
                            [=](bool pushed)
                            {
                              if (pushed)
                              {
                                camera_->SetFixedPose(slam_viewer::SE3());
                                camera_->SetModelView(pangolin::ModelViewLookAt(0, 0, 300, 0, 0, 0, pangolin::AxisX));
                                camera_->SetFree();
                              }
                            });

  /// 3 设置前置视角
  view_menu_->AddButtonItem("Front View",
                            [=](bool pushed)
                            {
                              if (pushed)
                              {
                                camera_->SetFollow();
                                camera_->SetModelView(pangolin::ModelViewLookAt(-50, 0, 10, 0, 0, 0, pangolin::AxisZ));
                                camera_->SetFree();
                              }
                            });
}

/**
 * @brief 更新参考帧逆深度点
 *
 * 在初始化过程中，用于更新参考关键帧的逆深度点的可视化行为
 * 1. 将那些没有被当前帧优化的逆深度点置为灰色
 * 2. 将那些被当前帧优化的逆深度点置为红色
 * 3. 这里默认为初始化器的参考关键帧为世界坐标系
 */
void Visualizer::UpdateInitializerCloud(const std::vector<InitIdepthPoint::SharedPtr> &idepth_points, const float &fx,
                                        const float &fy, const float &cx, const float &cy)
{
  auto cloud = pcl::make_shared<CloudT>();
  cloud->resize(idepth_points.size());

  float fx_inv = 1.0 / fx;
  float fy_inv = 1.0 / fy;
  float cx_inv = cx / fx;
  float cy_inv = cy / fy;

  std::vector<int> indices(idepth_points.size());
  std::iota(indices.begin(), indices.end(), 0);

  auto point_process = [&](const int &idx)
  {
    auto point = idepth_points[idx];
    Eigen::Vector3f point_eig = project::Pixel2Norm(point->host_pixel_positon_, fx_inv, fy_inv, cx_inv, cy_inv);
    point_eig /= point->idepth_;

    PointT point_rgb;
    point_rgb.x = point_eig[0];
    point_rgb.y = point_eig[1];
    point_rgb.z = point_eig[2];

    if (point->is_good_)
    {
      point_rgb.r = 255;
      point_rgb.g = 0;
      point_rgb.b = 0;
    }
    else
    {
      point_rgb.r = 127;
      point_rgb.g = 127;
      point_rgb.b = 127;
    }

    cloud->points[idx] = point_rgb;
  };

  std::for_each(std::execution::par, indices.begin(), indices.end(), point_process);

  auto self_factory = std::make_shared<slam_viewer::SelfColor<PointT>>(cloud);
  auto new_cloudui = std::make_shared<slam_viewer::CloudUI>(slam_viewer::Vec3(0.5, 0.5, 0.5), 3.0, 5.0);
  new_cloudui->SetCloud<PointT>(cloud, slam_viewer::SE3(), self_factory);

  // 使用先移除后添加的替换方式
  if (initializer_cloud_)
    view_3d_->RemoveUIItem(initializer_cloud_);

  view_3d_->AddUIItem(new_cloudui);
  std::swap(new_cloudui, initializer_cloud_);
}

Visualizer::Options::Options(const std::string &option_file)
{
  if (!std::filesystem::exists(option_file))
    throw std::runtime_error("配置文件不存在");

  auto info = YAML::LoadFile(option_file);
  world_coor_length_ = info["WorldCoordianteLength"].as<float>();
  camera_coor_length_ = info["CameraCoordianteLength"].as<float>();
  trajectory_line_width_ = info["Trajectory"]["Width"].as<float>();
  trajectory_point_size_ = info["Trajectory"]["PointSize"].as<float>();
  window_width_ = info["Window"]["Width"].as<int>();
  window_height_ = info["Window"]["Height"].as<int>();
  image_shower_width_ = info["ImageShower"]["Width"].as<int>();
  image_shower_height_ = info["ImageShower"]["Height"].as<int>();

  std::vector<float> trajectory_color = info["Trajectory"]["Color"].as<std::vector<float>>();
  trajectory_color_[0] = trajectory_color[0];
  trajectory_color_[1] = trajectory_color[1];
  trajectory_color_[2] = trajectory_color[2];
}

/**
 * @brief 可视化器
 *
 * 1. 创建窗口
 * 2. 创建相机
 * 3. 创建菜单
 * 4. 创建3d显示、图像显示区域
 * 5. 创建世界坐标系、相机坐标系和相机轨迹
 */
Visualizer::Visualizer(Options::SharedPtr options)
    : options_(std::move(options))
{
  using namespace slam_viewer;

  world_coordinate_ = std::make_shared<CoordinateUI>(options_->world_coor_length_, slam_viewer::SE3());
  camera_coordinate_ = std::make_shared<CoordinateUI>(options_->camera_coor_length_, slam_viewer::SE3());
  camera_trajectory_ = std::make_shared<TrajectoryUI>(options_->trajectory_color_, options_->trajectory_line_width_,
                                                      options_->trajectory_point_size_);

  // todo 这里似乎需要坐标变换，保证空间系统的z轴和惯性系相同
  camera_ = std::make_shared<Camera>("camera", camera_coordinate_);
  window_impl_ = std::make_shared<WindowImpl>("DSO SSL Visualizer", options_->window_width_, options_->window_height_);
  image_shower_ = std::make_shared<ImageShower>("Visualizer Image Shower", 1, 2);
  view_3d_ = std::make_shared<View3D>("Visualizer 3d");
  view_menu_ = std::make_shared<Menu>("Viewer Menu");

  view_3d_->SetCamera(camera_);
  view_3d_->AddUIItem(world_coordinate_);
  view_3d_->AddUIItem(camera_coordinate_);
  view_3d_->AddUIItem(camera_trajectory_);

  window_impl_->AddView(view_menu_, 0, 1, 0.0, 0.1);
  window_impl_->AddView(view_3d_, 0, 1, 0.1, 0.9);
  window_impl_->AddView(image_shower_, 0, 0, 0, 0);

  ConfigMenu();

  image_shower_->AddImage("Reference Frame", options_->image_shower_height_, options_->image_shower_width_);
  image_shower_->AddImage("Tracking Frame", options_->image_shower_height_, options_->image_shower_width_);
}

} // namespace dso_ssl
