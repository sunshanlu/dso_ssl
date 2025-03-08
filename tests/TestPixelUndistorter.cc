#include "dso/PixelUndistorter.hpp"
#include "utils/TimerWrapper.hpp"

std::string TEST_IMG_PATH = "/home/lucky-lu/Projects/dso_ssl/tests/res/00011.jpg";
std::string CONFIG_PATH = "/home/lucky-lu/Projects/dso_ssl/tests/config/FOVPixelUndistorter.yaml";

using namespace dso_ssl;

int main()
{
    timer::TimerWrapper timer_wrapper("Undistorter Test");

    PixelUndistorter::Config::SharedPtr config = std::make_shared<PixelUndistorter::FOVConfig>(CONFIG_PATH);
    PixelUndistorter::SharedPtr undistorter = std::make_shared<PixelUndistorter>(config);

    cv::Mat distorted_image = cv::imread(TEST_IMG_PATH, cv::IMREAD_GRAYSCALE);
    distorted_image.convertTo(distorted_image, CV_32F);
    cv::Mat undistorted_image = timer_wrapper.ExecuteAndMeasure(
        "PixelUndistorter::Undistort", [undistorter](const cv::Mat &img) -> cv::Mat { return undistorter->Undistort(img); }, distorted_image);

    undistorted_image.convertTo(undistorted_image, CV_8U);
    distorted_image.convertTo(distorted_image, CV_8U);
    timer_wrapper.TimerShow();

    cv::imshow("Distorted Image", distorted_image);
    cv::imshow("Undistorted Image", undistorted_image);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
