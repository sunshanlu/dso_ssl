/**
 * 1. 初始化器的设置 reference frame 参考关键帧
 * 2. 针对单个层级优化的方法测试
 * 3. 针对某个关键帧优化的方法测试
 * 4. 初始化器的整体流程测试，包含初始化器的可视化问题
 */

#include <execution>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <opencv2/opencv.hpp>

#include "dso/Visualizer.hpp"
#include "dso/Initializer2.hpp"
#include "utils/TimerWrapper.hpp"

using namespace dso_ssl;

std::string TEST_IMG_PATH[17] = {"./tests/res/00000.jpg", "./tests/res/00001.jpg", "./tests/res/00002.jpg", "./tests/res/00003.jpg", "./tests/res/00004.jpg",
                                 "./tests/res/00005.jpg", "./tests/res/00006.jpg", "./tests/res/00007.jpg", "./tests/res/00008.jpg", "./tests/res/00009.jpg",
                                 "./tests/res/00010.jpg", "./tests/res/00011.jpg", "./tests/res/00012.jpg", "./tests/res/00013.jpg", "./tests/res/00014.jpg",
                                 "./tests/res/00015.jpg", "./tests/res/00016.jpg"};
std::string PHOTO_CONFIG_PATH = "./tests/config/PhotoUndistorter.yaml";
std::string PIXEL_CONFIG_PATH = "./tests/config/FOVPixelUndistorter.yaml";
std::string UNDIS_CONFIG_PATH = "./tests/config/Undistorter.yaml";
std::string FRAME_CONFIG_PATH = "./tests/config/Frame.yaml";
std::string INIT_CONFIG_PATH = "./tests/config/Initializer.yaml";
std::string SELECT_CONFIG_PATH = "./tests/config/PixelSelector.yaml";
std::string STAMP_AND_EXPOSURE_PATH = "./tests/res/times.txt";
std::string VISUALIZER_CONFIG_PATH = "./tests/config/Visualizer.yaml";

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
}

void GetTimestampAndExposure(const std::string &time_path, std::vector<double> &timestamps, std::vector<float> &exposure_times)
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
  timer::TimerWrapper timer_wrapper("Initializer2 SetReference Test");

  PhotoUndistorter::Options::SharedPtr photo_config = std::make_shared<PhotoUndistorter::Options>(PHOTO_CONFIG_PATH);
  PixelUndistorter::Options::SharedPtr pixel_config = std::make_shared<PixelUndistorter::FOVConfig>(PIXEL_CONFIG_PATH);
  Undistorter::Options::SharedPtr undis_config = std::make_shared<Undistorter::Options>(UNDIS_CONFIG_PATH);
  Frame::Options::SharedPtr frame_config = std::make_shared<Frame::Options>(FRAME_CONFIG_PATH);
  PixelSelector::Options::SharedPtr select_config = std::make_shared<PixelSelector::Options>(SELECT_CONFIG_PATH);
  Initializer2::Options::SharedPtr init_config = std::make_shared<Initializer2::Options>(INIT_CONFIG_PATH);
  Visualizer::Options::SharedPtr visualizer_config = std::make_shared<Visualizer::Options>(VISUALIZER_CONFIG_PATH);

  Undistorter::SharedPtr undistorter = std::make_shared<Undistorter>(pixel_config, photo_config, undis_config);
  PixelSelector::SharedPtr pixel_selector = std::make_shared<PixelSelector>(select_config);
  Pattern::SharedPtr pattern = std::make_shared<Pattern>(8);
  Visualizer::SharedPtr visualizer = std::make_shared<Visualizer>(visualizer_config);
  visualizer->Run();

  // 构造初始化器
  float fx, fy, cx, cy;
  undistorter->GetTargetK(fx, fy, cx, cy);
  Initializer2::SharedPtr initializer2 = std::make_shared<Initializer2>(init_config, pixel_selector, pattern, fx, fy, cx, cy);
  initializer2->SetVisualizer(visualizer);

  std::vector<double> timestamps;
  std::vector<float> exposure_times;
  GetTimestampAndExposure(STAMP_AND_EXPOSURE_PATH, timestamps, exposure_times);

  for (int idx = 0; idx < 17; ++idx)
  {
    cv::Mat distorted_image = cv::imread(TEST_IMG_PATH[idx], cv::IMREAD_GRAYSCALE);
    cv::Mat only_pixel_undistorted_image;
    cv::Mat undistorted_image = undistorter->Undistort(distorted_image, only_pixel_undistorted_image);

    Frame::SharedPtr frame = std::make_shared<Frame>(frame_config, undistorted_image, only_pixel_undistorted_image, timestamps[idx], exposure_times[idx]);

    if (idx == 0)
    {
      initializer2->SetReference(frame);
      continue;
    }

    std::cout << "=========================================" << std::endl;
    std::cout << frame->GetIdx() << std::endl;
    std::cout << "=========================================" << std::endl;

    auto ret = initializer2->TrackActivateFrame(frame);
    if (ret)
      break;
  }

  visualizer->Join();

  return 0;
}
