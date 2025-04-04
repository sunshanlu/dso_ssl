/**
 * 初始化器的 ComputeJacobianAndError 函数测试
 *  1. 测试 相同层级之间的相邻关系
 *  2. 测试 不同层级之间的父子关系
 */

#include <execution>
#include <iostream>
#include <random>
#include <vector>

#include <opencv2/opencv.hpp>

#include "dso/Initializer2.hpp"
#include "utils/TimerWrapper.hpp"

using namespace dso_ssl;

std::string TEST_IMG_PATH0 = "./tests/res/00000.jpg";
std::string TEST_IMG_PATH1 = "./tests/res/00004.jpg";
std::string PHOTO_CONFIG_PATH = "./tests/config/PhotoUndistorter.yaml";
std::string PIXEL_CONFIG_PATH = "./tests/config/FOVPixelUndistorter.yaml";
std::string UNDIS_CONFIG_PATH = "./tests/config/Undistorter.yaml";
std::string FRAME_CONFIG_PATH = "./tests/config/Frame.yaml";
std::string INIT_CONFIG_PATH = "./tests/config/Initializer.yaml";
std::string SELECT_CONFIG_PATH = "./tests/config/PixelSelector.yaml";

void NormFilePath()
{
  TEST_IMG_PATH0 = std::filesystem::absolute(TEST_IMG_PATH0).lexically_normal();
  TEST_IMG_PATH1 = std::filesystem::absolute(TEST_IMG_PATH1).lexically_normal();
  PHOTO_CONFIG_PATH = std::filesystem::absolute(PHOTO_CONFIG_PATH).lexically_normal();
  PIXEL_CONFIG_PATH = std::filesystem::absolute(PIXEL_CONFIG_PATH).lexically_normal();
  UNDIS_CONFIG_PATH = std::filesystem::absolute(UNDIS_CONFIG_PATH).lexically_normal();
  FRAME_CONFIG_PATH = std::filesystem::absolute(FRAME_CONFIG_PATH).lexically_normal();
  SELECT_CONFIG_PATH = std::filesystem::absolute(SELECT_CONFIG_PATH).lexically_normal();
  INIT_CONFIG_PATH = std::filesystem::absolute(INIT_CONFIG_PATH).lexically_normal();
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

  Undistorter::SharedPtr undistorter = std::make_shared<Undistorter>(pixel_config, photo_config, undis_config);
  PixelSelector::SharedPtr pixel_selector = std::make_shared<PixelSelector>(select_config);

  cv::Mat distorted_image0 = cv::imread(TEST_IMG_PATH0, cv::IMREAD_GRAYSCALE);
  cv::Mat only_pixel_undistorted_image0, undistorted_image0;

  cv::Mat distorted_image1 = cv::imread(TEST_IMG_PATH1, cv::IMREAD_GRAYSCALE);
  cv::Mat only_pixel_undistorted_image1, undistorted_image1;

  undistorted_image0 = undistorter->Undistort(distorted_image0, only_pixel_undistorted_image0);
  Frame::SharedPtr frame0 = std::make_shared<Frame>(frame_config, undistorted_image0, only_pixel_undistorted_image0, 0, 0);

  undistorted_image1 = undistorter->Undistort(distorted_image1, only_pixel_undistorted_image1);
  Frame::SharedPtr frame1 = std::make_shared<Frame>(frame_config, undistorted_image1, only_pixel_undistorted_image1, 0, 0);

  // 构造初始化器
  Pattern::SharedPtr pattern = std::make_shared<Pattern>(8);
  float fx, fy, cx, cy;
  undistorter->GetTargetK(fx, fy, cx, cy);
  Initializer2::SharedPtr initializer2 = std::make_shared<Initializer2>(init_config, pixel_selector, pattern, fx, fy, cx, cy);
  initializer2->SetReference(frame0);
  initializer2->SetCurrentFrame(frame1);

  Sophus::SE3f Tji;
  float aji = 0, bji = 0;

  Initializer2::Mat8f H, Hsc;
  Initializer2::Vec8f b, bsc;

  initializer2->SetOptimizationParams(Tji, aji, bji);
  auto result0 = initializer2->ComputeJacobianAndError(init_config->pyra_levels_ - 1, H, b, Hsc, bsc, false);
  for (int iteration = 0; iteration < 50; ++iteration)
  {
    Initializer2::Vec8f inc = -(H - Hsc).ldlt().solve(b - bsc);
    initializer2->ApplyStep(init_config->pyra_levels_ - 1, inc, 0);
    auto result1 = initializer2->ComputeJacobianAndError(init_config->pyra_levels_ - 1, H, b, Hsc, bsc, false);
    std::cout << "iteration: " << iteration << "\tsqrt(energy / nums): " << std::sqrt(result0[0] / result0[3]) << "->" << std::sqrt(result1[0] / result1[3])
              << "\tinlier num: " << (int)result0[3] << "->" << (int)result1[3] << std::endl;

    std::swap(result0, result1);
  }

  return 0;
}
