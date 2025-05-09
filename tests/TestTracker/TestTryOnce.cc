
/**
 * 1. 初始化器的设置 reference frame 参考关键帧
 * 2. 针对单个层级优化的方法测试
 * 3. 针对某个关键帧优化的方法测试
 * 4. 初始化器的整体流程测试，包含初始化器的可视化问题
 */

#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <opencv2/opencv.hpp>

#include "dso/Initializer2.hpp"
#include "dso/Tracker.hpp"
#include "dso/Visualizer.hpp"
#include "utils/TimerWrapper.hpp"

using namespace dso_ssl;

std::string TEST_IMG_PATH[17] = {"./tests/res/00000.jpg", "./tests/res/00001.jpg", "./tests/res/00002.jpg",
                                 "./tests/res/00003.jpg", "./tests/res/00004.jpg", "./tests/res/00005.jpg",
                                 "./tests/res/00006.jpg", "./tests/res/00007.jpg", "./tests/res/00008.jpg",
                                 "./tests/res/00009.jpg", "./tests/res/00010.jpg", "./tests/res/00011.jpg",
                                 "./tests/res/00012.jpg", "./tests/res/00013.jpg", "./tests/res/00014.jpg",
                                 "./tests/res/00015.jpg", "./tests/res/00016.jpg"};
std::string PHOTO_CONFIG_PATH = "./tests/config/PhotoUndistorter.yaml";
std::string PIXEL_CONFIG_PATH = "./tests/config/FOVPixelUndistorter.yaml";
std::string UNDIS_CONFIG_PATH = "./tests/config/Undistorter.yaml";
std::string FRAME_CONFIG_PATH = "./tests/config/Frame.yaml";
std::string INIT_CONFIG_PATH = "./tests/config/Initializer.yaml";
std::string SELECT_CONFIG_PATH = "./tests/config/PixelSelector.yaml";
std::string STAMP_AND_EXPOSURE_PATH = "./tests/res/times.txt";
std::string VISUALIZER_CONFIG_PATH = "./tests/config/Visualizer.yaml";
std::string TRACKER_CONFIG_PATH = "./tests/config/Tracker.yaml";

void NormFilePath()
{
  for (int idx = 0; idx < 17; ++idx)
    TEST_IMG_PATH[idx] = std::filesystem::absolute(TEST_IMG_PATH[idx]).lexically_normal();

  PHOTO_CONFIG_PATH = std::filesystem::absolute(PHOTO_CONFIG_PATH).lexically_normal();
  PIXEL_CONFIG_PATH = std::filesystem::absolute(PIXEL_CONFIG_PATH).lexically_normal();
  UNDIS_CONFIG_PATH = std::filesystem::absolute(UNDIS_CONFIG_PATH).lexically_normal();
  FRAME_CONFIG_PATH = std::filesystem::absolute(FRAME_CONFIG_PATH).lexically_normal();
  SELECT_CONFIG_PATH = std::filesystem::absolute(SELECT_CONFIG_PATH).lexically_normal();
  INIT_CONFIG_PATH = std::filesystem::absolute(INIT_CONFIG_PATH).lexically_normal();
  STAMP_AND_EXPOSURE_PATH = std::filesystem::absolute(STAMP_AND_EXPOSURE_PATH).lexically_normal();
  VISUALIZER_CONFIG_PATH = std::filesystem::absolute(VISUALIZER_CONFIG_PATH).lexically_normal();
  TRACKER_CONFIG_PATH = std::filesystem::absolute(TRACKER_CONFIG_PATH).lexically_normal();
}

void GetTimestampAndExposure(const std::string &time_path, std::vector<double> &timestamps,
                             std::vector<float> &exposure_times)
{
  double timestamp;
  float exposure_time;
  std::string path_idx;
  std::ifstream time_file(time_path);

  while (time_file >> path_idx >> timestamp >> exposure_time)
  {
    timestamps.push_back(timestamp);
    exposure_times.push_back(exposure_time);
  }
}

int main(int argc, char **argv)
{
  NormFilePath();
  timer::TimerWrapper timer_wrapper("Tracker Update Tracker");

  PixelUndistorter::Options::SharedPtr pixel_config = std::make_shared<PixelUndistorter::FOVConfig>(PIXEL_CONFIG_PATH);
  auto photo_config = std::make_shared<PhotoUndistorter::Options>(PHOTO_CONFIG_PATH);
  auto undis_config = std::make_shared<Undistorter::Options>(UNDIS_CONFIG_PATH);
  auto frame_config = std::make_shared<Frame::Options>(FRAME_CONFIG_PATH);
  auto select_config = std::make_shared<PixelSelector::Options>(SELECT_CONFIG_PATH);
  auto init_config = std::make_shared<Initializer2::Options>(INIT_CONFIG_PATH);
  auto visualizer_config = std::make_shared<Visualizer::Options>(VISUALIZER_CONFIG_PATH);
  auto tracker_config = std::make_shared<Tracker::Options>(TRACKER_CONFIG_PATH);

  auto undistorter = std::make_shared<Undistorter>(pixel_config, photo_config, undis_config);
  auto pixel_selector = std::make_shared<PixelSelector>(select_config);
  auto pattern = std::make_shared<Pattern>(8);
  auto visualizer = std::make_shared<Visualizer>(visualizer_config);
  auto tracker = std::make_shared<Tracker>(tracker_config);
  visualizer->Run();

  // 构造初始化器
  float fx, fy, cx, cy;
  undistorter->GetTargetK(fx, fy, cx, cy);
  auto initializer2 = std::make_shared<Initializer2>(init_config, pixel_selector, pattern, fx, fy, cx, cy);
  initializer2->SetVisualizer(visualizer);

  std::vector<double> timestamps;
  std::vector<float> exposure_times;
  GetTimestampAndExposure(STAMP_AND_EXPOSURE_PATH, timestamps, exposure_times);

  int idx = 0;
  bool initialized = false;
  bool initializer_ref = false;
  std::vector<KeyFrame::SharedPtr> sliding_window(2, nullptr);

  for (idx = 0; idx < 17; ++idx)
  {
    cv::Mat distorted_image = cv::imread(TEST_IMG_PATH[idx], cv::IMREAD_GRAYSCALE);
    cv::Mat only_pixel_undistorted_image;
    cv::Mat undistorted_image = undistorter->Undistort(distorted_image, only_pixel_undistorted_image);

    auto frame = std::make_shared<Frame>(frame_config, undistorted_image, only_pixel_undistorted_image, timestamps[idx],
                                         exposure_times[idx]);

    if (!initializer_ref)
    {
      initializer2->SetReference(frame);
      initializer_ref = true;
      continue;
    }

    std::cout << "=========================================" << std::endl;
    std::cout << frame->GetIdx() << std::endl;
    std::cout << "=========================================" << std::endl;

    if (!initialized)
    {
      initialized = initializer2->TrackActivateFrame(frame);

      if (initialized)
      {
        initializer2->GetKeyFrames(sliding_window[0], sliding_window[1]);
        tracker->UpdateTracker(sliding_window, fx, fy, cx, cy);
      }
    }

    if (initialized)
    {
      Sophus::SE3f Tji = Sophus::SE3f(), _;
      float aj = 0, bj = 0, ai, bi;
      sliding_window[1]->GetFrameStatus(_, ai, bi);

      tracker->SetCurrFrame(frame);
      tracker->TryOnce(Tji, aj, bj, ai, bi, {50, 50, 50, 50});
    }
  }


  visualizer->Join();

  return 0;
}
