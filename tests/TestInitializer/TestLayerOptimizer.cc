/**
 * 测试 层级优化器的 LayerOptimizer::Optimize() 函数的效果
 */

#include <execution>
#include <random>
#include <vector>

#include <opencv2/opencv.hpp>

#include "dso/Initializer.hpp"
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

  // std::cout << "TEST_IMG_PATH: " << TEST_IMG_PATH << std::endl;
  // std::cout << "PHOTO_CONFIG_PATH: " << PHOTO_CONFIG_PATH << std::endl;
  // std::cout << "PIXEL_CONFIG_PATH: " << PIXEL_CONFIG_PATH << std::endl;
  // std::cout << "UNDIS_CONFIG_PATH: " << UNDIS_CONFIG_PATH << std::endl;
  // std::cout << "FRAME_CONFIG_PATH: " << FRAME_CONFIG_PATH << std::endl;
  // std::cout << "SELECT_CONFIG_PATH: " << SELECT_CONFIG_PATH << std::endl;
  // std::cout << "INIT_CONFIG_PATH: " << INIT_CONFIG_PATH << std::endl;
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
  timer::TimerWrapper timer_wrapper("Initializer Test");

  PhotoUndistorter::Config::SharedPtr photo_config = std::make_shared<PhotoUndistorter::Config>(PHOTO_CONFIG_PATH);
  PixelUndistorter::Config::SharedPtr pixel_config = std::make_shared<PixelUndistorter::FOVConfig>(PIXEL_CONFIG_PATH);
  Undistorter::Config::SharedPtr undis_config = std::make_shared<Undistorter::Config>(UNDIS_CONFIG_PATH);
  Frame::Config::SharedPtr frame_config = std::make_shared<Frame::Config>(FRAME_CONFIG_PATH);
  PixelSelector::Config::SharedPtr select_config = std::make_shared<PixelSelector::Config>(SELECT_CONFIG_PATH);
  Initializer::Config::SharedPtr init_config = std::make_shared<Initializer::Config>(INIT_CONFIG_PATH);

  Undistorter::SharedPtr undistorter = std::make_shared<Undistorter>(pixel_config, photo_config, undis_config);
  PixelSelector::SharedPtr pixel_selector = std::make_shared<PixelSelector>(select_config);

  std::vector<double> timestamps;
  std::vector<float> exposure_times;
  GetTimestampAndExposure(STAMP_AND_EXPOSURE_PATH, timestamps, exposure_times);

  // 构造初始化器
  float fx, fy, cx, cy;
  undistorter->GetTargetK(fx, fy, cx, cy);
  Pattern::SharedPtr pattern = std::make_shared<Pattern>(8);
  Initializer::SharedPtr initializer = std::make_shared<Initializer>(init_config, pixel_selector, pattern, fx, fy, cx, cy);

  cv::Mat distorted_image;
  cv::Mat only_pixel_undistorted_image, undistorted_image;
  double timestamp;
  float exposure_time;

  auto Undistort = [&]() -> cv::Mat { return undistorter->Undistort(distorted_image, only_pixel_undistorted_image); };
  auto FrameConstruct = [&]() -> Frame::SharedPtr
  {
    auto frame = std::make_shared<Frame>(frame_config, undistorted_image, only_pixel_undistorted_image, timestamp, exposure_time);
    return frame;
  };

  auto AddActivateFrame = [&](Frame::SharedPtr frame) -> bool { return initializer->AddActivateFrame(frame); };

  // 执行初始化器的初始化过程
  // int indices[] = {0, 3, 6, 10, 15, 16};
  int indices[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  for (auto &idx : indices)
  {
    timestamp = timestamps[idx];
    exposure_time = exposure_times[idx];
    distorted_image = cv::imread(TEST_IMG_PATH[idx], cv::IMREAD_GRAYSCALE);
    undistorted_image = timer_wrapper.ExecuteAndMeasure("Undistorter::Distort", Undistort);
    Frame::SharedPtr frame = timer_wrapper.ExecuteAndMeasure("Frame Constructor", FrameConstruct);

    bool initialized = timer_wrapper.ExecuteAndMeasure("Initializer::AddActivateFrame", AddActivateFrame, frame);
  }

  timer_wrapper.TimerShow();
  cv::destroyAllWindows();

  return 0;
}
