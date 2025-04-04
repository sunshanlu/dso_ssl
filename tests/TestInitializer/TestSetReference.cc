/**
 * 初始化器的 SetReference 函数测试
 *  1. 测试 相同层级之间的相邻关系
 *  2. 测试 不同层级之间的父子关系
 */

#include <execution>
#include <random>
#include <vector>

#include <opencv2/opencv.hpp>

#include "dso/Initializer.hpp"
#include "utils/TimerWrapper.hpp"

using namespace dso_ssl;

/// 函数用于生成随机颜色
cv::Scalar GetRandomColor()
{
    int b = std::rand() % 256;
    int g = std::rand() % 256;
    int r = std::rand() % 256;
    return cv::Scalar(b, g, r);
}

/// 用于绘制包含points的最小外接圆
void DrawBoundingCircle(cv::Mat &image, const cv::Point2f &center, const std::vector<cv::Point2f> &points)
{
    auto color = GetRandomColor();

    if (points.empty())
        throw std::invalid_argument("points is empty");

    float radius = -1;
    for (const auto &point : points)
    {
        float distance = cv::norm(point - center);
        if (distance > radius)
            radius = distance;
    }

    cv::circle(image, center, static_cast<int>(radius), color, 1);
    for (const auto &point : points)
        cv::circle(image, point, 1, color, 1);

    cv::circle(image, center, 2, color, 2);
}

/// 用于绘制多对一的对应关系
void DrawCorrespondences(cv::Mat &image, const cv::Point2f &point1, const std::vector<cv::Point2f> &points2, const int &offset)
{
    // 绘制第二个图像上的点并连接
    auto color = GetRandomColor();
    for (const auto &point2 : points2)
    {
        cv::Point2f point1_offset(point1.x + offset, point1.y);
        cv::line(image, point1_offset, point2, color, 1);
    }
}

/// 展示参考帧
void ShowReferenceFrame(const std::vector<LayerFrame::SharedPtr> &layer_frames)
{
    for (int layer = 0; layer < layer_frames.size() - 1; ++layer)
    {
        auto curr_frame = layer_frames[layer];
        auto next_frame = layer_frames[layer + 1];

        cv::Mat curr_image, next_image;
        curr_frame->layer_image_.convertTo(curr_image, CV_8U);
        next_frame->layer_image_.convertTo(next_image, CV_8U);
        cv::cvtColor(curr_image, curr_image, cv::COLOR_GRAY2BGR);
        cv::cvtColor(next_image, next_image, cv::COLOR_GRAY2BGR);

        // 将两张图像拼接
        cv::Mat combine_image(curr_image.rows, curr_image.cols + next_image.cols, CV_8UC3, cv::Scalar(127, 127, 127));
        curr_image.copyTo(combine_image.rowRange(0, curr_image.rows).colRange(0, curr_image.cols));
        next_image.copyTo(combine_image.rowRange(0, next_image.rows).colRange(curr_image.cols, curr_image.cols + next_image.cols));

        auto current_frame_process = [&](const int &idx)
        {
            const auto &pc = curr_frame->pixel_points_[idx];
            cv::Point2f pc_cv(pc->pixel_position_[0], pc->pixel_position_[1]);
            cv::circle(curr_image, pc_cv, 1, cv::Scalar(0, 0, 255), 1);
            cv::circle(combine_image, pc_cv, 1, cv::Scalar(0, 0, 255), 1);
        };

        auto next_frame_process = [&](const int &idx)
        {
            const auto &pn = next_frame->pixel_points_[idx];
            cv::Point2f pn_cv(pn->pixel_position_[0], pn->pixel_position_[1]);
            cv::circle(combine_image, cv::Point2f(pn_cv.x + curr_image.cols, pn_cv.y), 1, cv::Scalar(0, 0, 255), 1);
        };

        // 在curr_image上绘制圆，在combine_image上绘制selected point
        auto circle_process = [&](const int &idx)
        {
            if (idx % 50 != 0)
                return;

            const auto &pc = curr_frame->pixel_points_[idx];
            cv::Point2f pc_cv(pc->pixel_position_[0], pc->pixel_position_[1]);

            std::vector<cv::Point2f> circle_data;
            for (const auto &neighbor_id : pc->neighbor_ids_)
            {
                const auto &neighbor_pc = curr_frame->pixel_points_[neighbor_id];

                cv::Point2f neighbor_cv(neighbor_pc->pixel_position_[0], neighbor_pc->pixel_position_[1]);
                circle_data.push_back(neighbor_cv);
            }

            // 在curr_image上绘制点和邻居的临域
            DrawBoundingCircle(curr_image, pc_cv, circle_data);
        };

        // 在combine_image上绘制点和line关系
        auto line_process = [&](const int &idx)
        {
            if (idx % 50 != 0)
                return;

            const auto &pn = next_frame->pixel_points_[idx];
            cv::Point2f pn_cv(pn->pixel_position_[0], pn->pixel_position_[1]);

            std::vector<cv::Point2f> line_data;
            for (const auto &child_id : pn->children_ids_)
            {
                const auto &child_pc = curr_frame->pixel_points_[child_id];

                cv::Point2f child_cv(child_pc->pixel_position_[0], child_pc->pixel_position_[1]);
                line_data.push_back(child_cv);
            }

            DrawCorrespondences(combine_image, pn_cv, line_data, curr_image.cols);
        };

        // 在图像上绘制点
        std::vector<int> curr_indices(curr_frame->pixel_points_.size(), 0);
        std::vector<int> next_indices(next_frame->pixel_points_.size(), 0);
        std::iota(curr_indices.begin(), curr_indices.end(), 0);
        std::iota(next_indices.begin(), next_indices.end(), 0);

        std::for_each(curr_indices.begin(), curr_indices.end(), current_frame_process);
        std::for_each(next_indices.begin(), next_indices.end(), next_frame_process);
        std::for_each(curr_indices.begin(), curr_indices.end(), circle_process);
        std::for_each(next_indices.begin(), next_indices.end(), line_process);

        cv::imshow("circle image", curr_image);
        cv::imshow("line image", combine_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

std::string TEST_IMG_PATH = "./tests/res/00011.jpg";
std::string PHOTO_CONFIG_PATH = "./tests/config/PhotoUndistorter.yaml";
std::string PIXEL_CONFIG_PATH = "./tests/config/FOVPixelUndistorter.yaml";
std::string UNDIS_CONFIG_PATH = "./tests/config/Undistorter.yaml";
std::string FRAME_CONFIG_PATH = "./tests/config/Frame.yaml";
std::string INIT_CONFIG_PATH = "./tests/config/Initializer.yaml";
std::string SELECT_CONFIG_PATH = "./tests/config/PixelSelector.yaml";

void NormFilePath()
{
    TEST_IMG_PATH = std::filesystem::absolute(TEST_IMG_PATH).lexically_normal();
    PHOTO_CONFIG_PATH = std::filesystem::absolute(PHOTO_CONFIG_PATH).lexically_normal();
    PIXEL_CONFIG_PATH = std::filesystem::absolute(PIXEL_CONFIG_PATH).lexically_normal();
    UNDIS_CONFIG_PATH = std::filesystem::absolute(UNDIS_CONFIG_PATH).lexically_normal();
    FRAME_CONFIG_PATH = std::filesystem::absolute(FRAME_CONFIG_PATH).lexically_normal();
    SELECT_CONFIG_PATH = std::filesystem::absolute(SELECT_CONFIG_PATH).lexically_normal();
    INIT_CONFIG_PATH = std::filesystem::absolute(INIT_CONFIG_PATH).lexically_normal();

    std::cout << "TEST_IMG_PATH: " << TEST_IMG_PATH << std::endl;
    std::cout << "PHOTO_CONFIG_PATH: " << PHOTO_CONFIG_PATH << std::endl;
    std::cout << "PIXEL_CONFIG_PATH: " << PIXEL_CONFIG_PATH << std::endl;
    std::cout << "UNDIS_CONFIG_PATH: " << UNDIS_CONFIG_PATH << std::endl;
    std::cout << "FRAME_CONFIG_PATH: " << FRAME_CONFIG_PATH << std::endl;
    std::cout << "SELECT_CONFIG_PATH: " << SELECT_CONFIG_PATH << std::endl;
    std::cout << "INIT_CONFIG_PATH: " << INIT_CONFIG_PATH << std::endl;
}

int main(int argc, char **argv)
{
    NormFilePath();
    timer::TimerWrapper timer_wrapper("Initializer SetReference Test");

    PhotoUndistorter::Options::SharedPtr photo_config = std::make_shared<PhotoUndistorter::Options>(PHOTO_CONFIG_PATH);
    PixelUndistorter::Options::SharedPtr pixel_config = std::make_shared<PixelUndistorter::FOVConfig>(PIXEL_CONFIG_PATH);
    Undistorter::Options::SharedPtr undis_config = std::make_shared<Undistorter::Options>(UNDIS_CONFIG_PATH);
    Frame::Options::SharedPtr frame_config = std::make_shared<Frame::Options>(FRAME_CONFIG_PATH);
    PixelSelector::Options::SharedPtr select_config = std::make_shared<PixelSelector::Options>(SELECT_CONFIG_PATH);
    Initializer::Options::SharedPtr init_config = std::make_shared<Initializer::Options>(INIT_CONFIG_PATH);

    Undistorter::SharedPtr undistorter = std::make_shared<Undistorter>(pixel_config, photo_config, undis_config);
    PixelSelector::SharedPtr pixel_selector = std::make_shared<PixelSelector>(select_config);

    cv::Mat distorted_image = cv::imread(TEST_IMG_PATH, cv::IMREAD_GRAYSCALE);
    cv::Mat only_pixel_undistorted_image, undistorted_image;

    auto Undistort = [&]() -> cv::Mat { return undistorter->Undistort(distorted_image, only_pixel_undistorted_image); };
    auto FrameConstruct = [&]() -> Frame::SharedPtr
    {
        auto frame = std::make_shared<Frame>(frame_config, undistorted_image, only_pixel_undistorted_image, 0, 0);
        return frame;
    };

    undistorted_image = timer_wrapper.ExecuteAndMeasure("Undistorter::Distort", Undistort);
    Frame::SharedPtr frame = timer_wrapper.ExecuteAndMeasure("Frame Constructor", FrameConstruct);

    // 构造初始化器
    Pattern::SharedPtr pattern = std::make_shared<Pattern>(8);
    Initializer::SharedPtr initializer = std::make_shared<Initializer>(init_config, pixel_selector, pattern, 0, 0, 0, 0);
    initializer->SetReferenceFrame(frame);

    ShowReferenceFrame(initializer->GetRefLayerInfo());

    return 0;
}
