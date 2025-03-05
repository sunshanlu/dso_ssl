#include "dso/Undistorter.hpp"
#include "utils/TimerWrapper.hpp"

std::string TEST_IMG_PATH = "../tests/res/00011.jpg";
std::string CONFIG_PATH = "../test/config/FOVPixelUndistorter.yaml";

using namespace dso_ssl;

int main()
{
    timer::TimerWrapper timer_wrapper;

    PixelUndistorter::Config::SharedPtr config = std::make_shared<PixelUndistorter::FOVConfig>(CONFIG_PATH);
    PixelUndistorter::SharedPtr undistorter = std::make_shared<PixelUndistorter>(config);

    cv::Mat distorted_image = cv::imread(TEST_IMG_PATH);
    cv::Mat undistorted_image = timer_wrapper.ExecuteAndMeasure(
        "PixelUndistorter::Undistort", [undistorter](const cv::Mat &img) -> cv::Mat { return undistorter->Undistort(img); }, distorted_image);

    timer_wrapper.TimerShow();
    return 0;
}
