/**
 * 1. 去畸变图像输出，查看内容
 * 2. 仅像素去畸变图像输出，查看内容
 */

#include <filesystem>

#include "dso/Undistorter.hpp"
#include "utils/TimerWrapper.hpp"

std::string TEST_IMG_PATH = "./tests/res/00011.jpg";
std::string PHOTO_CONFIG_PATH = "./tests/config/PhotoUndistorter.yaml";
std::string PIXEL_CONFIG_PATH = "./tests/config/FOVPixelUndistorter.yaml";
std::string UNDIS_CONFIG_PATH = "./tests/config/Undistorter.yaml";

using namespace dso_ssl;

void NormFilePath()
{
    TEST_IMG_PATH = std::filesystem::absolute(TEST_IMG_PATH).lexically_normal();
    PHOTO_CONFIG_PATH = std::filesystem::absolute(PHOTO_CONFIG_PATH).lexically_normal();
    PIXEL_CONFIG_PATH = std::filesystem::absolute(PIXEL_CONFIG_PATH).lexically_normal();
    UNDIS_CONFIG_PATH = std::filesystem::absolute(UNDIS_CONFIG_PATH).lexically_normal();

    std::cout << "TEST_IMG_PATH: " << TEST_IMG_PATH << std::endl;
    std::cout << "PHOTO_CONFIG_PATH: " << PHOTO_CONFIG_PATH << std::endl;
    std::cout << "PIXEL_CONFIG_PATH: " << PIXEL_CONFIG_PATH << std::endl;
    std::cout << "UNDIS_CONFIG_PATH: " << UNDIS_CONFIG_PATH << std::endl;
}

int main()
{
    NormFilePath();
    timer::TimerWrapper timer_wrapper("Undistorter Test");

    PhotoUndistorter::Options::SharedPtr photo_config = std::make_shared<PhotoUndistorter::Options>(PHOTO_CONFIG_PATH);
    PixelUndistorter::Options::SharedPtr pixel_config = std::make_shared<PixelUndistorter::FOVConfig>(PIXEL_CONFIG_PATH);
    Undistorter::Options::SharedPtr undis_config = std::make_shared<Undistorter::Options>(UNDIS_CONFIG_PATH);

    Undistorter::SharedPtr undistorter = std::make_shared<Undistorter>(pixel_config, photo_config, undis_config);

    cv::Mat distorted_image = cv::imread(TEST_IMG_PATH, cv::IMREAD_GRAYSCALE);
    cv::Mat only_pixel_undistorted_image;

    cv::Mat undistorted_image = timer_wrapper.ExecuteAndMeasure("Undistorter::Distort", [&]() -> cv::Mat
                                                                { return undistorter->Undistort(distorted_image, only_pixel_undistorted_image); });

    undistorted_image.convertTo(undistorted_image, CV_8U);
    distorted_image.convertTo(distorted_image, CV_8U);
    only_pixel_undistorted_image.convertTo(only_pixel_undistorted_image, CV_8U);

    timer_wrapper.TimerShow();

    cv::imshow("Distorted Image", distorted_image);
    cv::imshow("Undistorted Image", undistorted_image);
    cv::imshow("Only Pixel Undistorted Image", only_pixel_undistorted_image);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
