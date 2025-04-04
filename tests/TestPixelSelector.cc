/**
 * 1. 像素选择器第0层点选测试
 * 2. 像素选择器第1层点选测试
 */

#include <filesystem>
#include <vector>

#include "dso/Frame.hpp"
#include "dso/PixelSelector.hpp"
#include "dso/Undistorter.hpp"
#include "utils/PreProImage.hpp"
#include "utils/TimerWrapper.hpp"

std::string TEST_IMG_PATH = "./tests/res/00011.jpg";
std::string PHOTO_CONFIG_PATH = "./tests/config/PhotoUndistorter.yaml";
std::string PIXEL_CONFIG_PATH = "./tests/config/FOVPixelUndistorter.yaml";
std::string UNDIS_CONFIG_PATH = "./tests/config/Undistorter.yaml";
std::string FRAME_CONFIG_PATH = "./tests/config/Frame.yaml";
std::string SELECT_CONFIG_PATH = "./tests/config/PixelSelector.yaml";

using namespace dso_ssl;

void NormFilePath()
{
    TEST_IMG_PATH = std::filesystem::absolute(TEST_IMG_PATH).lexically_normal();
    PHOTO_CONFIG_PATH = std::filesystem::absolute(PHOTO_CONFIG_PATH).lexically_normal();
    PIXEL_CONFIG_PATH = std::filesystem::absolute(PIXEL_CONFIG_PATH).lexically_normal();
    UNDIS_CONFIG_PATH = std::filesystem::absolute(UNDIS_CONFIG_PATH).lexically_normal();
    FRAME_CONFIG_PATH = std::filesystem::absolute(FRAME_CONFIG_PATH).lexically_normal();
    SELECT_CONFIG_PATH = std::filesystem::absolute(SELECT_CONFIG_PATH).lexically_normal();

    std::cout << "TEST_IMG_PATH: " << TEST_IMG_PATH << std::endl;
    std::cout << "PHOTO_CONFIG_PATH: " << PHOTO_CONFIG_PATH << std::endl;
    std::cout << "PIXEL_CONFIG_PATH: " << PIXEL_CONFIG_PATH << std::endl;
    std::cout << "UNDIS_CONFIG_PATH: " << UNDIS_CONFIG_PATH << std::endl;
    std::cout << "FRAME_CONFIG_PATH: " << UNDIS_CONFIG_PATH << std::endl;
    std::cout << "SELECT_CONFIG_PATH: " << SELECT_CONFIG_PATH << std::endl;
}

namespace dso_ssl
{

void ShowFirstLayer(const cv::Mat &image, const PixelSelector::Vector2iArray &selected_points, const std::vector<int> &selected_points_conf)
{
    // 创建一个副本，避免修改原始图像
    cv::Mat image_with_points = image.clone();
    image_with_points.convertTo(image_with_points, CV_8U);
    cv::cvtColor(image_with_points, image_with_points, cv::COLOR_GRAY2BGR);

    // 定义不同置信度的颜色
    cv::Scalar color_conf_0(0, 0, 255); // 红色
    cv::Scalar color_conf_1(0, 255, 0); // 绿色
    cv::Scalar color_conf_2(255, 0, 0); // 蓝色

    // 遍历所有选中的点并绘制
    for (size_t i = 0; i < selected_points.size(); ++i)
    {
        const Eigen::Vector2i &point = selected_points[i];
        int confidence = selected_points_conf[i];
        cv::Scalar color;

        // 根据置信度选择颜色
        switch (confidence)
        {
        case 0:
            color = color_conf_0;
            break;
        case 1:
            color = color_conf_1;
            break;
        case 2:
            color = color_conf_2;
            break;
        default:
            color = cv::Scalar(128, 128, 128); // 默认颜色（灰色）
            break;
        }

        // 绘制点
        cv::circle(image_with_points, cv::Point(point[0], point[1]), 2, color, -1);
    }

    // 显示图像
    cv::imshow("Layer 0", image_with_points);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ShowOtherLayer(const cv::Mat &image, const PixelSelector::Vector2iArray &selected_points, const int &layer)
{
    // 创建一个副本，避免修改原始图像
    cv::Mat image_with_points = image.clone();
    image_with_points.convertTo(image_with_points, CV_8U);
    cv::cvtColor(image_with_points, image_with_points, cv::COLOR_GRAY2BGR);

    // 定义不同置信度的颜色
    cv::Scalar color(0, 0, 255);

    // 遍历所有选中的点并绘制
    for (const auto &point : selected_points)
        cv::circle(image_with_points, cv::Point(point[0], point[1]), 2, color, -1);

    // 显示图像
    cv::imshow("Layer " + std::to_string(layer), image_with_points);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

} // namespace dso_ssl

int main()
{
    NormFilePath();
    timer::TimerWrapper timer_wrapper("PixelSelector Test");

    PhotoUndistorter::Options::SharedPtr photo_config = std::make_shared<PhotoUndistorter::Options>(PHOTO_CONFIG_PATH);
    PixelUndistorter::Options::SharedPtr pixel_config = std::make_shared<PixelUndistorter::FOVConfig>(PIXEL_CONFIG_PATH);
    Undistorter::Options::SharedPtr undis_config = std::make_shared<Undistorter::Options>(UNDIS_CONFIG_PATH);
    Frame::Options::SharedPtr frame_config = std::make_shared<Frame::Options>(FRAME_CONFIG_PATH);
    PixelSelector::Options::SharedPtr select_config = std::make_shared<PixelSelector::Options>(SELECT_CONFIG_PATH);

    Undistorter::SharedPtr undistorter = std::make_shared<Undistorter>(pixel_config, photo_config, undis_config);
    PixelSelector::SharedPtr pixel_selector = std::make_shared<PixelSelector>(select_config);

    cv::Mat distorted_image = cv::imread(TEST_IMG_PATH, cv::IMREAD_GRAYSCALE);
    cv::Mat only_pixel_undistorted_image;

    cv::Mat undistorted_image = undistorter->Undistort(distorted_image, only_pixel_undistorted_image);
    Frame::SharedPtr frame = std::make_shared<Frame>(frame_config, undistorted_image, only_pixel_undistorted_image, 0, 0);

    auto image_and_grads = frame->GetPyrdImageAndGrads();
    auto squre_grad = frame->GetSqureGrad();

    float densities[] = {0.03, 0.05, 0.15, 0.5, 0.8};

    for (int layer = 0; layer < frame_config->pyra_levels_; ++layer)
    {
        PixelSelector::Vector2iArray selected_points;
        std::vector<cv::Mat> image_and_grads_vec;
        cv::split(image_and_grads[layer], image_and_grads_vec);
        if (layer == 0)
        {
            std::vector<int> selected_points_conf;
            timer_wrapper.ExecuteAndMeasure("PixelSelector::SelectFirstLayer",
                                            [&]() -> int
                                            {
                                                pixel_selector->SelectFirstLayer(densities[0] * squre_grad.rows * squre_grad.cols, squre_grad, selected_points,
                                                                                 selected_points_conf);
                                                return 0;
                                            });
            ShowFirstLayer(image_and_grads_vec[0], selected_points, selected_points_conf);
            continue;
        }

        timer_wrapper.ExecuteAndMeasure("PixelSelector::SelectOtherLayer",
                                        [&]() -> int
                                        {
                                            pixel_selector->SelectOtherLayer(densities[layer] * image_and_grads[layer].rows * image_and_grads[layer].cols,
                                                                             image_and_grads[layer], selected_points);
                                            return 0;
                                        });
        ShowOtherLayer(image_and_grads_vec[0], selected_points, layer);
    }

    timer_wrapper.TimerShow();
    return 0;
}
