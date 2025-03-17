/**
 * 测试 投影过程
 *  1. 非SSE投影过程
 *  2. SSE投影过程
 */
#include <iostream>

#include <Eigen/Geometry>
#include <sophus/se3.hpp>

#include "dso/Pattern.hpp"
#include "utils/Project.hpp"
#include "utils/TimerWrapper.hpp"

using namespace dso_ssl;
using namespace project;

void PrintVector3f(Eigen::Vector3f vector3f) { std::cout << vector3f[0] << " " << vector3f[1] << " " << vector3f[2] << std::endl; }
void PrintVector2f(Eigen::Vector2f vector2f) { std::cout << vector2f[0] << " " << vector2f[1] << std::endl; }

int main(int argc, char **argv)
{
    timer::TimerWrapper time_wrapper("Test Project");

    float fx = 640, fy = 640, cx = 320, cy = 230, dpi = 0.1;

    Eigen::AngleAxisf rji(M_PI / 8, Eigen::Vector3f(0, 0, 1));
    Eigen::Vector3f tji(1.2, 0, 0.3);
    Eigen::Vector2f pixel_position(256.3, 117.8);
    Sophus::SE3f Tji(rji.toRotationMatrix(), tji);

    Vector2fArray pixel_positions(4);
    Vector3fArray pj_temp_positions_not_sse(4);
    Vector2fArray pixel_not_sse(4);

    auto not_sse_operator = [&]()
    {
        for (int idx = 0; idx < 4; ++idx)
        {
            pixel_positions[idx] = pixel_position + Pattern::pattern8[idx];
            Pixel2Pixel(Tji, pixel_positions[idx], dpi, fx, fy, cx, cy, pj_temp_positions_not_sse[idx], pixel_not_sse[idx]);
        }
        return 0;
    };

    time_wrapper.ExecuteAndMeasure("No SSE", not_sse_operator);

    Vector2fArray pixel_sse;
    Vector3fArray pj_temp_positions_sse;

    auto sse_operator = [&]()
    {
        Pixel2PixelSSE(Tji, pixel_positions, dpi, fx, fy, cx, cy, pj_temp_positions_sse, pixel_sse);
        return 0;
    };
    time_wrapper.ExecuteAndMeasure("SSE", sse_operator);

    time_wrapper.TimerShow();

    for (int idx = 0; idx < 4; ++idx)
    {
        PrintVector2f(pixel_sse[idx]);
        PrintVector2f(pixel_not_sse[idx]);
    }

    for (int idx = 0; idx < 4; ++idx)
    {
        PrintVector3f(pj_temp_positions_sse[idx]);
        PrintVector3f(pj_temp_positions_not_sse[idx]);
    }

    return 0;
}