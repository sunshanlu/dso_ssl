/**
 * 1. 图像金字塔构建测试
 *      1.1 金字塔各层图像展示，看是否有明显差异
 *      1.2 图像梯度展示 gradx, grady
 * 2. 梯度平方和展示
 *      2.1 不使用原图的梯度平方和
 *      2.2 使用原图的梯度平方和
 * 3. 使用和不使用原图的时间对比
 */

#include <filesystem>

#include "dso/Frame.hpp"
#include "dso/Undistorter.hpp"
#include "utils/PreProImage.hpp"
#include "utils/TimerWrapper.hpp"

std::string TEST_IMG_PATH = "./tests/res/00011.jpg";
std::string PHOTO_CONFIG_PATH = "./tests/config/PhotoUndistorter.yaml";
std::string PIXEL_CONFIG_PATH = "./tests/config/FOVPixelUndistorter.yaml";
std::string UNDIS_CONFIG_PATH = "./tests/config/Undistorter.yaml";
std::string FRAME_CONFIG_PATH = "./tests/config/Frame.yaml";

using namespace dso_ssl;

void NormFilePath()
{
    TEST_IMG_PATH = std::filesystem::absolute(TEST_IMG_PATH).lexically_normal();
    PHOTO_CONFIG_PATH = std::filesystem::absolute(PHOTO_CONFIG_PATH).lexically_normal();
    PIXEL_CONFIG_PATH = std::filesystem::absolute(PIXEL_CONFIG_PATH).lexically_normal();
    UNDIS_CONFIG_PATH = std::filesystem::absolute(UNDIS_CONFIG_PATH).lexically_normal();
    FRAME_CONFIG_PATH = std::filesystem::absolute(FRAME_CONFIG_PATH).lexically_normal();

    std::cout << "TEST_IMG_PATH: " << TEST_IMG_PATH << std::endl;
    std::cout << "PHOTO_CONFIG_PATH: " << PHOTO_CONFIG_PATH << std::endl;
    std::cout << "PIXEL_CONFIG_PATH: " << PIXEL_CONFIG_PATH << std::endl;
    std::cout << "UNDIS_CONFIG_PATH: " << UNDIS_CONFIG_PATH << std::endl;
    std::cout << "FRAME_CONFIG_PATH: " << UNDIS_CONFIG_PATH << std::endl;
}

namespace dso_ssl
{
void ShowPyraidImagesAndGrads(Frame::SharedPtr frameptr)
{
    std::vector<cv::Mat> pyrd_images;
    std::vector<cv::Mat> pyrd_gradx;
    std::vector<cv::Mat> pyrd_grady;

    for (const auto &image_and_grad : frameptr->pyrd_image_and_grads_)
    {
        std::vector<cv::Mat> image_and_grad_vec;
        cv::split(image_and_grad, image_and_grad_vec);

        double max_value1, max_value2;
        cv::minMaxLoc(image_and_grad_vec[1], nullptr, &max_value1);
        cv::minMaxLoc(image_and_grad_vec[2], nullptr, &max_value2);

        image_and_grad_vec[1] = image_and_grad_vec[1] / max_value1 * 255;
        image_and_grad_vec[2] = image_and_grad_vec[2] / max_value2 * 255;

        image_and_grad_vec[0].convertTo(image_and_grad_vec[0], CV_8U);
        image_and_grad_vec[1].convertTo(image_and_grad_vec[1], CV_8U);
        image_and_grad_vec[2].convertTo(image_and_grad_vec[2], CV_8U);

        pyrd_images.push_back(image_and_grad_vec[0]);
        pyrd_gradx.push_back(image_and_grad_vec[1]);
        pyrd_grady.push_back(image_and_grad_vec[2]);
    }

    for (int idx = 0; idx < pyrd_images.size(); ++idx)
        cv::imshow("Image pyrd " + std::to_string(idx), pyrd_images[idx]);

    for (int idx = 0; idx < pyrd_gradx.size(); ++idx)
        cv::imshow("Image pyrd gradx " + std::to_string(idx), pyrd_gradx[idx]);

    for (int idx = 0; idx < pyrd_grady.size(); ++idx)
        cv::imshow("Image pyrd grady " + std::to_string(idx), pyrd_grady[idx]);

    // 展示梯度平方和
    cv::Mat squre_grad;
    double max_squre_value;
    cv::minMaxLoc(frameptr->squre_grad_, nullptr, &max_squre_value);
    squre_grad = frameptr->squre_grad_ / max_squre_value * 255;

    squre_grad.convertTo(squre_grad, CV_8U);
    cv::imshow("Image squre grad", squre_grad);

    cv::waitKey(0);
    cv::destroyAllWindows();
}
} // namespace dso_ssl

int main()
{
    NormFilePath();
    timer::TimerWrapper timer_wrapper("Frame Test");

    PhotoUndistorter::Config::SharedPtr photo_config = std::make_shared<PhotoUndistorter::Config>(PHOTO_CONFIG_PATH);
    PixelUndistorter::Config::SharedPtr pixel_config = std::make_shared<PixelUndistorter::FOVConfig>(PIXEL_CONFIG_PATH);
    Undistorter::Config::SharedPtr undis_config = std::make_shared<Undistorter::Config>(UNDIS_CONFIG_PATH);
    Frame::Config::SharedPtr frame_config = std::make_shared<Frame::Config>(FRAME_CONFIG_PATH);

    Undistorter::SharedPtr undistorter = std::make_shared<Undistorter>(pixel_config, photo_config, undis_config);

    cv::Mat distorted_image = cv::imread(TEST_IMG_PATH, cv::IMREAD_GRAYSCALE);
    cv::Mat only_pixel_undistorted_image;

    // 金字塔核心函数测试
    // cv::Mat pyrad_test, grad_x, grad_y;
    // distorted_image.convertTo(pyrad_test, CV_32FC1);
    // auto scale_image = prepro_image::MakePyrdOneLayer(pyrad_test);
    // prepro_image::MakeGradOneLayer(scale_image, grad_x, grad_y);
    // scale_image.convertTo(scale_image, CV_8U);

    // double max_value_grad_x, max_value_grad_y;
    // cv::minMaxLoc(grad_x, nullptr, &max_value_grad_x);
    // cv::minMaxLoc(grad_y, nullptr, &max_value_grad_y);

    // grad_x = grad_x / max_value_grad_x * 255;
    // grad_y = grad_y / max_value_grad_y * 255;

    // grad_x.convertTo(grad_x, CV_8U);
    // grad_y.convertTo(grad_y, CV_8U);
    // cv::imshow("scale image", scale_image);
    // cv::imshow("grad x", grad_x);
    // cv::imshow("grad y", grad_y);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    cv::Mat undistorted_image = timer_wrapper.ExecuteAndMeasure("Undistorter::Distort", [&]() -> cv::Mat
                                                                { return undistorter->Undistort(distorted_image, only_pixel_undistorted_image); });

    Frame::SharedPtr frame = timer_wrapper.ExecuteAndMeasure("Frame Constructor",
                                                             [&]() -> Frame::SharedPtr
                                                             {
                                                                 auto frame =
                                                                     std::make_shared<Frame>(frame_config, undistorted_image, only_pixel_undistorted_image);
                                                                 return frame;
                                                             });

    timer_wrapper.TimerShow();
    ShowPyraidImagesAndGrads(frame);
    return 0;
}
