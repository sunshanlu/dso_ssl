#include <filesystem>

#include "dso/Undistorter.hpp"
#include "utils/TimerWrapper.hpp"

std::string TEST_IMG_PATH = "./tests/res/00011.jpg";
std::string CONFIG_PATH = "./tests/config/PhotoUndistorter.yaml";

using namespace dso_ssl;

void NormFilePath()
{
    TEST_IMG_PATH = std::filesystem::absolute(TEST_IMG_PATH).lexically_normal();
    CONFIG_PATH = std::filesystem::absolute(CONFIG_PATH).lexically_normal();

    std::cout << "TEST_IMG_PATH: " << TEST_IMG_PATH << std::endl;
    std::cout << "CONFIG_PATH: " << CONFIG_PATH << std::endl;
}

int main()
{
    NormFilePath();
    timer::TimerWrapper timer_wrapper("PhotoUndistorter Test");

    PhotoUndistorter::Config::SharedPtr config = std::make_shared<PhotoUndistorter::Config>(CONFIG_PATH);
    PhotoUndistorter::SharedPtr undistorter = std::make_shared<PhotoUndistorter>(config);

    cv::Mat distorted_image = cv::imread(TEST_IMG_PATH, cv::IMREAD_GRAYSCALE);
    cv::Mat ginv_undistorted_image = timer_wrapper.ExecuteAndMeasure(
        "PhotoUndistorter::GinvUndistort", [undistorter](const cv::Mat &img) -> cv::Mat { return undistorter->GinvUndistort(img); }, distorted_image);

    cv::Mat vign_undistorted_image = timer_wrapper.ExecuteAndMeasure(
        "PhotoUndistorter::VignUndistort", [undistorter](const cv::Mat &img) -> cv::Mat { return undistorter->VignUndistort(img); }, ginv_undistorted_image);

    ginv_undistorted_image.convertTo(ginv_undistorted_image, CV_8U);
    vign_undistorted_image.convertTo(vign_undistorted_image, CV_8U);
    distorted_image.convertTo(distorted_image, CV_8U);
    timer_wrapper.TimerShow();

    cv::imshow("Distorted Image", distorted_image);
    cv::imshow("GinvUndistort Image", ginv_undistorted_image);
    cv::imshow("VignUndistort Image", vign_undistorted_image);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
