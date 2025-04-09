#include <ctime>
#include <execution>
#include <filesystem>
#include <numeric>
#include <yaml-cpp/yaml.h>

#include "dso/PixelSelector.hpp"
#include "utils/ParallelProcess.hpp"
#include "utils/PreProImage.hpp"
#include "utils/STLOperators.hpp"

namespace dso_ssl
{

/**
 * @brief 构建图像梯度阈值
 *
 * 以32px * 32px为图像块，统计图像块中的 config_->grad_thresh_percentile_ 分位数
 * 然后加上 config_->grad_thresh_bias_ 作为这个像素块里面的梯度阈值
 *
 * @param image_grad_squre  输入的图像像素梯度的平方
 */
void PixelSelector::BuildGradientThreshold(const cv::Mat &image_grad_squre)
{
    if (image_grad_squre.empty())
        throw std::runtime_error("image_grad_squre is empty");

    if (image_grad_squre.rows <= 3 * config_->patch_size_ || image_grad_squre.cols <= 3 * config_->patch_size_)
        throw std::runtime_error("image_grad_squre size is too small");

    row_start_ = (image_grad_squre.rows % config_->patch_size_) / 2;
    col_start_ = (image_grad_squre.cols % config_->patch_size_) / 2;
    row_end_ = image_grad_squre.rows - (image_grad_squre.rows % config_->patch_size_ - row_start_);
    col_end_ = image_grad_squre.cols - (image_grad_squre.cols % config_->patch_size_ - col_start_);

    grad_thresh_ = cv::Mat::zeros(image_grad_squre.rows / config_->patch_size_, image_grad_squre.cols / config_->patch_size_, CV_32F);

    auto patch_process = [&](const Eigen::Vector2i &start_position)
    {
        std::vector<float> hist(50, 0);

        for (int delta_row = 0; delta_row < config_->patch_size_; delta_row++)
            for (int delta_col = 0; delta_col < config_->patch_size_; delta_col++)
            {
                int row = start_position[1] + delta_row;
                int col = start_position[0] + delta_col;

                int hist_idx = std::sqrt(image_grad_squre.at<float>(row, col)) + 0.5;
                if (hist_idx > 49)
                    hist_idx = 49;

                ++hist[hist_idx];
            }

        int percentile_num = config_->patch_size_ * config_->patch_size_ * config_->grad_thresh_percentile_;
        int grad_thresh = 0;
        for (; grad_thresh < 50; ++grad_thresh)
        {
            percentile_num -= hist[grad_thresh];

            if (percentile_num <= 0)
                break;
        }

        int row = start_position[1] / config_->patch_size_;
        int col = start_position[0] / config_->patch_size_;

        grad_thresh_.at<float>(row, col) = grad_thresh + config_->grad_thresh_bias_;
    };

    prepro_image::SegmentImage(image_grad_squre, patch_process, config_->patch_size_, row_start_, col_start_);
}

/**
 * @brief 梯度阈值区域平滑
 *
 * 使用3x3的均值卷积实现整体的阈值平滑，重点处理了边界条件部分。
 * 对于四个角位置，使用2 * 2卷积
 * 对于四个边缘位置，使用2 * 3或者3 * 2卷积
 *
 */
void PixelSelector::SoomthGradientThreshold()
{
    cv::Mat grad_thresh_smooth(grad_thresh_.rows, grad_thresh_.cols, CV_32F);
    Vector2iArray pre_positions;

    for (int row = 1; row < grad_thresh_.rows - 1; ++row)
        for (int col = 1; col < grad_thresh_.cols - 1; ++col)
            pre_positions.emplace_back(col, row);

    auto simd_process = [&](const int &start)
    {
        alignas(16) float data00[4], data01[4], data02[4], data10[4], result[4];
        alignas(16) float data11[4], data12[4], data20[4], data21[4], data22[4];

        for (int idx = start; idx < start + 4; ++idx)
        {
            const int &row_center = pre_positions[idx][1];
            const int &col_center = pre_positions[idx][0];

            data00[idx - start] = grad_thresh_.at<float>(row_center - 1, col_center - 1);
            data01[idx - start] = grad_thresh_.at<float>(row_center - 1, col_center);
            data02[idx - start] = grad_thresh_.at<float>(row_center - 1, col_center + 1);
            data10[idx - start] = grad_thresh_.at<float>(row_center, col_center - 1);
            data11[idx - start] = grad_thresh_.at<float>(row_center, col_center);
            data12[idx - start] = grad_thresh_.at<float>(row_center, col_center + 1);
            data20[idx - start] = grad_thresh_.at<float>(row_center + 1, col_center - 1);
            data21[idx - start] = grad_thresh_.at<float>(row_center + 1, col_center);
            data22[idx - start] = grad_thresh_.at<float>(row_center + 1, col_center + 1);
        }

        auto data00_sse = _mm_load_ps(data00);
        auto data01_sse = _mm_load_ps(data01);
        auto data02_sse = _mm_load_ps(data02);
        auto data10_sse = _mm_load_ps(data10);
        auto data11_sse = _mm_load_ps(data11);
        auto data12_sse = _mm_load_ps(data12);
        auto data20_sse = _mm_load_ps(data20);
        auto data21_sse = _mm_load_ps(data21);
        auto data22_sse = _mm_load_ps(data22);
        auto multi_item = _mm_set1_ps(1.0f / 9.0f);

        auto sum_00_01_02 = _mm_add_ps(_mm_add_ps(data00_sse, data01_sse), data02_sse);
        auto sum_10_11_12 = _mm_add_ps(_mm_add_ps(data10_sse, data11_sse), data12_sse);
        auto sum_20_21_22 = _mm_add_ps(_mm_add_ps(data20_sse, data21_sse), data22_sse);

        auto result_sse = _mm_mul_ps(multi_item, _mm_add_ps(_mm_add_ps(sum_00_01_02, sum_10_11_12), sum_20_21_22));
        _mm_store_ps(result, result_sse);

        for (int idx = 0; idx < 4; ++idx)
        {
            const int &row_center = pre_positions[idx + start][1];
            const int &col_center = pre_positions[idx + start][0];
            grad_thresh_smooth.at<float>(row_center, col_center) = result[idx];
        }
    };

    auto single_process = [&](const int &idx)
    {
        const int &row_center = pre_positions[idx][1];
        const int &col_center = pre_positions[idx][0];

        const float &data00 = grad_thresh_.at<float>(row_center - 1, col_center - 1);
        const float &data01 = grad_thresh_.at<float>(row_center - 1, col_center);
        const float &data02 = grad_thresh_.at<float>(row_center - 1, col_center + 1);
        const float &data10 = grad_thresh_.at<float>(row_center, col_center - 1);
        const float &data11 = grad_thresh_.at<float>(row_center, col_center);
        const float &data12 = grad_thresh_.at<float>(row_center, col_center + 1);
        const float &data20 = grad_thresh_.at<float>(row_center + 1, col_center - 1);
        const float &data21 = grad_thresh_.at<float>(row_center + 1, col_center);
        const float &data22 = grad_thresh_.at<float>(row_center + 1, col_center + 1);

        grad_thresh_smooth.at<float>(row_center, col_center) = (data00 + data01 + data02 + data10 + data11 + data12 + data20 + data21 + data22) / 9.0f;
    };

    auto process_border = [&]()
    {
        // row = 0
        for (int col = 1; col < grad_thresh_.cols - 1; ++col)
        {
            const float &data10 = grad_thresh_.at<float>(0, col - 1);
            const float &data11 = grad_thresh_.at<float>(0, col);
            const float &data12 = grad_thresh_.at<float>(0, col + 1);
            const float &data20 = grad_thresh_.at<float>(1, col - 1);
            const float &data21 = grad_thresh_.at<float>(1, col);
            const float &data22 = grad_thresh_.at<float>(1, col + 1);

            grad_thresh_smooth.at<float>(0, col) = (data10 + data11 + data12 + data20 + data21 + data22) / 6.0f;
        }

        // row = -1
        for (int col = 1; col < grad_thresh_.cols - 1; ++col)
        {
            const float &data00 = grad_thresh_.at<float>(grad_thresh_.rows - 2, col - 1);
            const float &data01 = grad_thresh_.at<float>(grad_thresh_.rows - 2, col);
            const float &data02 = grad_thresh_.at<float>(grad_thresh_.rows - 2, col + 1);
            const float &data10 = grad_thresh_.at<float>(grad_thresh_.rows - 1, col - 1);
            const float &data11 = grad_thresh_.at<float>(grad_thresh_.rows - 1, col);
            const float &data12 = grad_thresh_.at<float>(grad_thresh_.rows - 1, col + 1);

            grad_thresh_smooth.at<float>(0, col) = (data10 + data11 + data12 + data00 + data01 + data02) / 6.0f;
        }

        // col = 0
        for (int row = 1; row < grad_thresh_.rows - 1; ++row)
        {
            const float &data01 = grad_thresh_.at<float>(row - 1, 0);
            const float &data02 = grad_thresh_.at<float>(row - 1, 1);
            const float &data11 = grad_thresh_.at<float>(row, 0);
            const float &data12 = grad_thresh_.at<float>(row, 1);
            const float &data21 = grad_thresh_.at<float>(row + 1, 0);
            const float &data22 = grad_thresh_.at<float>(row + 1, 1);

            grad_thresh_smooth.at<float>(row, 0) = (data01 + data02 + data11 + data12 + data21 + data22) / 6.0f;
        }

        // col = -1
        for (int row = 1; row < grad_thresh_.rows - 1; ++row)
        {
            const float &data00 = grad_thresh_.at<float>(row - 1, grad_thresh_.cols - 2);
            const float &data01 = grad_thresh_.at<float>(row - 1, grad_thresh_.cols - 1);
            const float &data10 = grad_thresh_.at<float>(row, grad_thresh_.cols - 2);
            const float &data11 = grad_thresh_.at<float>(row, grad_thresh_.cols - 1);
            const float &data20 = grad_thresh_.at<float>(row + 1, grad_thresh_.cols - 2);
            const float &data21 = grad_thresh_.at<float>(row + 1, grad_thresh_.cols - 1);

            grad_thresh_smooth.at<float>(row, 0) = (data00 + data01 + data10 + data11 + data20 + data21) / 6.0f;
        }

        // 左上角
        {
            const float &data11 = grad_thresh_.at<float>(0, 0);
            const float &data12 = grad_thresh_.at<float>(0, 1);
            const float &data21 = grad_thresh_.at<float>(1, 0);
            const float &data22 = grad_thresh_.at<float>(1, 1);

            grad_thresh_smooth.at<float>(0, 0) = (data11 + data12 + data21 + data22) / 4.0f;
        }

        // 右上角
        {
            const float &data10 = grad_thresh_.at<float>(0, grad_thresh_.cols - 2);
            const float &data11 = grad_thresh_.at<float>(0, grad_thresh_.cols - 1);
            const float &data20 = grad_thresh_.at<float>(1, grad_thresh_.cols - 2);
            const float &data21 = grad_thresh_.at<float>(1, grad_thresh_.cols - 1);

            grad_thresh_smooth.at<float>(0, 0) = (data10 + data11 + data20 + data21) / 4.0f;
        }

        // 左下角
        {
            const float &data01 = grad_thresh_.at<float>(grad_thresh_.rows - 2, 0);
            const float &data02 = grad_thresh_.at<float>(grad_thresh_.rows - 2, 1);
            const float &data11 = grad_thresh_.at<float>(grad_thresh_.rows - 1, 0);
            const float &data12 = grad_thresh_.at<float>(grad_thresh_.rows - 2, 1);

            grad_thresh_smooth.at<float>(0, 0) = (data01 + data02 + data11 + data12) / 4.0f;
        }

        // 右下角
        {
            const float &data00 = grad_thresh_.at<float>(grad_thresh_.rows - 2, grad_thresh_.cols - 2);
            const float &data01 = grad_thresh_.at<float>(grad_thresh_.rows - 2, grad_thresh_.cols - 1);
            const float &data10 = grad_thresh_.at<float>(grad_thresh_.rows - 1, grad_thresh_.cols - 2);
            const float &data11 = grad_thresh_.at<float>(grad_thresh_.rows - 1, grad_thresh_.cols - 1);

            grad_thresh_smooth.at<float>(0, 0) = (data00 + data01 + data10 + data11) / 4.0f;
        }
    };

    std::thread parall_process([&]() { parallel::ParallelWrapper(0, pre_positions.size(), 4, simd_process, single_process); });
    std::thread border_process(process_border);

    parall_process.join();
    border_process.join();
}

/**
 * @brief 在指定potinal大小的条件下，在金字塔第一层上选择点
 *
 * 分三个等级，对像素点进行选择：
 *  1. 如果pot * pot 区域的像素梯度良好，那么要求在pot * pot内选择一个点
 *  2. 如果2pot * 2pot 区域的像素梯度普遍不满足要求，那么要求在 2pot * 2pot内选择一个点
 *  3. 如果4pot * 4pot 区域的像素梯度普遍不满主要求，那么要求在 4pot * 4pot内选择一个点
 * 也就是说，最多可以选择 allpixels / (pot * pot)，小概率选择 allpixel / (2pot * 2pot)，最小概率选择allpixel / (4pot * 4pot)个点
 * 这种选择方式，可以保证像素分布的均匀性，而且像素梯度低的区域不至于选择不出像素点来。
 *
 * @param potinal                   输入的pot的大小
 * @param image_grad_squre          输入的图像梯度平方和
 * @param selected_points_out       输出的选择到的点
 * @param selected_points_conf_out  输出的选择点的置信度
 */
void PixelSelector::SelectFirstLayerInternal(const int &potinal, const cv::Mat &image_grad_squre, std::vector<int> &selected_points_conf_out,
                                             Vector2iArray &selected_points_out)
{
    const int &allpixels = image_grad_squre.rows * image_grad_squre.cols;

    Vector2iArray pre_positions;

    std::vector<bool> selected_points_flag(allpixels, false);
    Vector2iArray selected_points(allpixels);
    std::vector<int> selected_points_conf(allpixels, 0);

    int pot = potinal;
    int pot2 = 2 * pot;
    int pot4 = 4 * pot;

    int select_row_start = (image_grad_squre.rows % pot4) / 2;
    int select_col_start = (image_grad_squre.cols % pot4) / 2;

    for (int row = select_row_start; row < image_grad_squre.rows - pot4 + 1; row += pot4)
        for (int col = select_col_start; col < image_grad_squre.cols - pot4 + 1; col += pot4)
            pre_positions.emplace_back(col, row);

    // pot * pot 里面的处理函数
    auto pot_pot_process = [&](const int &pot_row_start, const int &pot_col_start, bool &have_flag_pot2, bool &have_flag_pot4, float &max_grad_value_pot2,
                               float &max_grad_value_pot4, Eigen::Vector2i &max_grad_position_pot2, Eigen::Vector2i &max_grad_position_pot4,
                               bool &pass_this_pot2, bool &pass_this_pot4)
    {
        // 遍历像素
        bool have_flag_pot = false;
        float max_grad_value_pot = -1;
        Eigen::Vector2i max_grad_position_pot;
        for (int row = pot_row_start; row < pot_row_start + pot; ++row)
        {
            for (int col = pot_col_start; col < pot_col_start + pot; ++col)
            {
                if (row < row_start_ || col < col_start_ || row >= row_end_ || col >= col_end_)
                    continue;

                const float &grad_value = std::sqrt(image_grad_squre.at<float>(row, col));
                const float &pot_grad_thresh = grad_thresh_.at<float>(row / config_->patch_size_, col / config_->patch_size_);
                const float &pot2_grad_thresh = config_->down_sacle_factor_ * pot_grad_thresh;
                const float &pot4_grad_thresh = config_->down_sacle_factor_ * pot2_grad_thresh;

                if (grad_value > pot_grad_thresh && grad_value > max_grad_value_pot)
                {
                    max_grad_value_pot = grad_value;
                    max_grad_position_pot = Eigen::Vector2i(col, row);
                    have_flag_pot = have_flag_pot2 = have_flag_pot4 = true;
                }
                else if (grad_value > pot2_grad_thresh && grad_value > max_grad_value_pot2)
                {
                    max_grad_value_pot2 = grad_value;
                    max_grad_position_pot2 = Eigen::Vector2i(col, row);
                    have_flag_pot2 = have_flag_pot4 = true;
                }
                else if (grad_value > pot4_grad_thresh && grad_value > max_grad_value_pot4)
                {
                    max_grad_value_pot4 = grad_value;
                    max_grad_position_pot4 = Eigen::Vector2i(col, row);
                    have_flag_pot4 = true;
                }
            }
        }

        if (have_flag_pot)
        {
            int idx = max_grad_position_pot[1] * image_grad_squre.cols + max_grad_position_pot[0];

            selected_points[idx] = max_grad_position_pot;
            selected_points_flag[idx] = true;
            pass_this_pot4 = pass_this_pot2 = true;
        }
    };

    // pot2 * pot2 里面的处理函数
    auto pot2_pot2_process = [&](const int &pot2_row_start, const int &pot2_col_start, bool &have_flag_pot4, float &max_grad_value_pot4,
                                 Eigen::Vector2i &max_grad_position_pot4, bool &pass_this_pot4)
    {
        bool have_flag_pot2 = false;
        bool pass_this_pot2 = false;
        float max_grad_value_pot2 = -1;
        Eigen::Vector2i max_grad_position_pot2;
        for (int pot_row_idx = 0; pot_row_idx < 2; ++pot_row_idx)
        {
            for (int pot_col_idx = 0; pot_col_idx < 2; ++pot_col_idx)
            {
                const int &pot_row_start = pot2_row_start + pot_row_idx * pot;
                const int &pot_col_start = pot2_col_start + pot_col_idx * pot;

                // pot * pot里面的处理函数
                pot_pot_process(pot_row_start, pot_col_start, have_flag_pot2, have_flag_pot4, max_grad_value_pot2, max_grad_value_pot4, max_grad_position_pot2,
                                max_grad_position_pot4, pass_this_pot2, pass_this_pot4);
            }
        }

        if (!pass_this_pot2 && have_flag_pot2)
        {
            int idx = max_grad_position_pot2[1] * image_grad_squre.cols + max_grad_position_pot2[0];

            selected_points[idx] = max_grad_position_pot2;
            selected_points_flag[idx] = true;
            selected_points_conf[idx] = 1;
            pass_this_pot4 = true;
        }
    };

    // pot4 * pot4 里面的处理函数
    auto pot4_pot4_process = [&](const int &idx)
    {
        const int &pot4_row_start = pre_positions[idx][1];
        const int &pot4_col_start = pre_positions[idx][0];

        // 遍历2pot * 2pot
        bool pass_this_pot4 = false;
        bool have_flag_pot4 = false;
        float max_grad_value_pot4 = -1;
        Eigen::Vector2i max_grad_position_pot4;
        for (int pot2_row_idx = 0; pot2_row_idx < 2; ++pot2_row_idx)
        {
            for (int pot2_col_idx = 0; pot2_col_idx < 2; ++pot2_col_idx)
            {
                const int &pot2_row_start = pot4_row_start + pot2_row_idx * pot2;
                const int &pot2_col_start = pot4_col_start + pot2_col_idx * pot2;

                // 遍历pot * pot
                pot2_pot2_process(pot2_row_start, pot2_col_start, have_flag_pot4, max_grad_value_pot4, max_grad_position_pot4, pass_this_pot4);
            }
        }

        if (!pass_this_pot4 && have_flag_pot4)
        {
            int idx = max_grad_position_pot4[1] * image_grad_squre.cols + max_grad_position_pot4[0];

            selected_points[idx] = max_grad_position_pot4;
            selected_points_flag[idx] = true;
            selected_points_conf[idx] = 2;
            pass_this_pot4 = true;
        }
    };

    std::vector<int> indices(pre_positions.size(), 0);
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(std::execution::par, indices.begin(), indices.end(), pot4_pot4_process);

    selected_points_out.clear();
    for (int idx = 0; idx < allpixels; ++idx)
        if (selected_points_flag[idx])
        {
            selected_points_out.push_back(selected_points[idx]);
            selected_points_conf_out.push_back(selected_points_conf[idx]);
        }
}

/**
 * @brief 选择图像金字塔第一层的特征点
 *
 * 该函数通过调整潜在特征点区域大小（potsize），在图像金字塔的第一层上选择特征点。
 * 它会递归地调整 potsize，直到选出的特征点数量满足期望值范围，或者达到最大递归深度。
 * 如果最终选出的特征点数量仍不满足要求，则可能对结果进行随机裁剪或输出警告信息。
 *
 * @param numwant               期望选择的特征点数量
 * @param image_grad_squre      输入的图像梯度平方和矩阵
 * @param selected_points       输出的选中特征点数组
 * @param selected_points_conf  输出的选中特征点置信度数组
 */
void PixelSelector::SelectFirstLayer(const int &numwant, const cv::Mat &image_grad_squre, Vector2iArray &selected_points,
                                     std::vector<int> &selected_points_conf)
{
    // 构建0层梯度阈值
    BuildGradientThreshold(image_grad_squre);
    SoomthGradientThreshold();

    int recursion_depth = 0;
    int allpixels = image_grad_squre.rows * image_grad_squre.cols;
    int current_potsize = std::sqrt(allpixels / static_cast<float>(numwant)) - 1;

    // 临界条件判断
    if (current_potsize < 1)
        current_potsize = 1;

    float ratio = 0;
    while (recursion_depth < config_->first_recu_depth_)
    {
        selected_points.clear();
        selected_points_conf.clear();

        SelectFirstLayerInternal(current_potsize, image_grad_squre, selected_points_conf, selected_points);

        ratio = static_cast<float>(numwant) / selected_points.size();

        if (ratio <= 1.25 && ratio >= 0.25)
            break;
        else if (ratio > 1.25)
        {
            current_potsize -= 1;
            if (current_potsize < 1)
                throw std::runtime_error("image select error, current_potsize < 1");
        }
        else
            current_potsize += 1;

        ++recursion_depth;
    }

    if (recursion_depth >= config_->first_recu_depth_)
        std::cerr << "达到最大迭代次数退出" << std::endl;

    if (ratio >= 0.25 && ratio < 0.95)
    {
        unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
        stl_opt::RemoveItemRandom(selected_points, numwant, seed);
        stl_opt::RemoveItemRandom(selected_points_conf, numwant, seed);
    }
}

/**
 * @brief 给定pot的条件下，在金字塔其他层上选择点
 *
 * 该函数用于在图像金字塔的其他层上选择特征点。在图像梯度大于给定梯度阈值的条件下，
 * pot * pot内区域内的x方向梯度最大位置，y方向梯度最大位置，x-y方向梯度最大位置和x+y方向梯度最大位置
 * 会被选择出来，当作该层的特征点位置。此外，该函数还通过多线程处理每个潜在的特征点区域，以加速计算过程。
 *
 * @param potinal               金字塔层的潜在特征点区域的大小
 * @param image_and_grads       包含图像和梯度信息的矩阵
 * @param selected_points_out   选中的特征点数组
 * @param factor_th             梯度阈值的因子，用于缩放梯度阈值
 */
void PixelSelector::SelectOtherLayerInternal(const int &potinal, const cv::Mat &image_and_grads, Vector2iArray &selected_points_out, float factor_th)
{
    Vector2iArray pre_postions;
    float other_layers_grad_th2 = config_->other_layers_grad_thresh_ * config_->other_layers_grad_thresh_ * factor_th * factor_th;

    int pot_row_start = (image_and_grads.rows % potinal) / 2;
    int pot_col_start = (image_and_grads.cols % potinal) / 2;

    for (int row = pot_row_start; row < image_and_grads.rows - potinal + 1; row += potinal)
        for (int col = pot_col_start; col < image_and_grads.cols - potinal + 1; col += potinal)
            pre_postions.push_back(Eigen::Vector2i(col, row));

    std::vector<bool> selected_points_flag(image_and_grads.rows * image_and_grads.cols, false);
    Vector2iArray selected_points(image_and_grads.rows * image_and_grads.cols);

    auto pot_process = [&](const int &idx)
    {
        const int &row_start = pre_postions[idx][1];
        const int &col_start = pre_postions[idx][0];

        // 逐像素遍历
        Eigen::Vector2i best_grad_xx_position, best_grad_yy_position, best_grad_xy_position, best_grad_yx_position;
        float best_grad_xx = -1, best_grad_yy = -1, best_grad_xy = -1, best_grad_yx = -1;
        for (int row_offset = 0; row_offset < potinal; ++row_offset)
        {
            for (int col_offset = 0; col_offset < potinal; ++col_offset)
            {
                int row = row_start + row_offset;
                int col = col_start + col_offset;

                const auto &image_and_grad = image_and_grads.at<cv::Vec3f>(row, col);
                const float &grad_x = image_and_grad[1];
                const float &grad_y = image_and_grad[2];

                if (grad_x * grad_x + grad_y * grad_y < other_layers_grad_th2)
                    continue;

                if (std::abs(grad_x) > best_grad_xx)
                {
                    best_grad_xx = std::abs(grad_x);
                    best_grad_xx_position = Eigen::Vector2i(col, row);
                }
                if (std::abs(grad_y) > best_grad_yy)
                {
                    best_grad_yy = std::abs(grad_y);
                    best_grad_yy_position = Eigen::Vector2i(col, row);
                }
                if (std::abs(grad_x - grad_y) > best_grad_xy)
                {
                    best_grad_xy = std::abs(grad_x - grad_y);
                    best_grad_xy_position = Eigen::Vector2i(col, row);
                }
                if (std::abs(grad_x - grad_y) > best_grad_yx)
                {
                    best_grad_yx = std::abs(grad_x - grad_y);
                    best_grad_yx_position = Eigen::Vector2i(col, row);
                }
            }
        }

        if (best_grad_xx > 0)
        {
            int idx = best_grad_xx_position[1] * image_and_grads.cols + best_grad_xx_position[0];
            selected_points[idx] = best_grad_xx_position;
            selected_points_flag[idx] = true;
        }
        if (best_grad_yy > 0)
        {
            int idx = best_grad_yy_position[1] * image_and_grads.cols + best_grad_yy_position[0];
            selected_points[idx] = best_grad_yy_position;
            selected_points_flag[idx] = true;
        }
        if (best_grad_xy > 0)
        {
            int idx = best_grad_xy_position[1] * image_and_grads.cols + best_grad_xy_position[0];
            selected_points[idx] = best_grad_xy_position;
            selected_points_flag[idx] = true;
        }
        if (best_grad_yx > 0)
        {
            int idx = best_grad_yx_position[1] * image_and_grads.cols + best_grad_yx_position[0];
            selected_points[idx] = best_grad_yx_position;
            selected_points_flag[idx] = true;
        }
    };

    std::vector<int> indices(pre_postions.size(), 0);
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par, indices.begin(), indices.end(), pot_process);

    selected_points_out.clear();
    for (int idx = 0; idx < selected_points.size(); ++idx)
    {
        if (!selected_points_flag[idx])
            continue;

        selected_points_out.push_back(selected_points[idx]);
    }
}

/**
 * @brief 在金字塔其他层上选点
 *
 * 该函数通过在图像的其他层上选择点来实现图像的特征点检测。它根据当前层的图像和梯度信息，
 * 以及配置参数，来确定在该层上应该选择多少个特征点，并调用内部函数来执行实际的选点操作。
 *
 * @param numwant           当前金字塔层上应该选择的特征点数量
 * @param image_and_grads   包含图像和梯度信息的矩阵
 * @param selected_points   选中的特征点数组，通过引用传递以便在函数内部修改
 * @param layer             当前处理的金字塔层索引
 */
void PixelSelector::SelectOtherLayer(const int &numwant, const cv::Mat &image_and_grads, Vector2iArray &selected_points)
{
    int current_pot = config_->other_layers_pot_size_;
    float current_factor_th = 1.0;

    if (current_pot < 1)
        current_pot = 1;
    SelectOtherLayerInternal(current_pot, image_and_grads, selected_points, current_factor_th);

    int allpixels = image_and_grads.rows * image_and_grads.cols;
    int recu_depth = 0;

    while (recu_depth < config_->other_recu_depth_)
    {
        int numhave = selected_points.size();
        float K = numhave * current_pot * current_pot;

        int desir_pot = std::sqrt(K / numwant) + 0.5f;
        float desir_factor_th = current_factor_th;
        if (desir_pot < 1)
        {
            std::cerr << "图像的整体梯度过低，目前会调整梯度阈值大小" << std::endl;
            desir_factor_th = current_factor_th * config_->other_layers_factor_th_;
            desir_pot = 1;
        }

        // 满住提取的阈值条件了
        if (numhave / (float)numwant > 0.8 && numhave / (float)numwant < 1 / 0.8)
            return;

        // 计算陷入死循环
        if (desir_pot == current_pot && current_factor_th == desir_factor_th)
            return;

        current_pot = desir_pot;
        current_factor_th = desir_factor_th;
        SelectOtherLayerInternal(current_pot, image_and_grads, selected_points, current_factor_th);
        ++recu_depth;
    }

    int numhave = selected_points.size();
    float ratio = numhave / (float)numwant;

    if (ratio < 0.8)
        std::cerr << "图像的梯度太低，导致提取的点过少，无法提取点" << std::endl;

    if (ratio > 1 / 0.8)
    {
        unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
        stl_opt::RemoveItemRandom(selected_points, numwant, seed);
        return;
    }
}

/**
 * @brief 根据yaml文件配置 构造像素选择器配置
 *
 * @param file_path 输入的配置yaml文件
 */
PixelSelector::Options::Options(std::string file_path)
{
    if (!std::filesystem::exists(file_path))
        throw std::runtime_error("配置文件不存在");

    auto info = YAML::LoadFile(file_path);
    first_layer_init_pot_ = info["FirstLayerConfig"]["InitPotinalSize"].as<int>();
    down_sacle_factor_ = info["FirstLayerConfig"]["DownScaleFactor"].as<float>();
    grad_thresh_bias_ = info["FirstLayerConfig"]["GradThreshBias"].as<float>();
    grad_thresh_percentile_ = info["FirstLayerConfig"]["GradThreshPercentile"].as<float>();
    patch_size_ = info["FirstLayerConfig"]["GradPatchSize"].as<int>();
    first_recu_depth_ = info["FirstLayerConfig"]["RecursionDepth"].as<int>();

    other_recu_depth_ = info["OtherLayerConfig"]["RecursionDepth"].as<int>();
    other_layers_pot_size_ = info["OtherLayerConfig"]["InitPotinalSize"].as<int>();
    other_layers_grad_thresh_ = info["OtherLayerConfig"]["GradThresh"].as<float>();
    other_layers_factor_th_ = info["OtherLayerConfig"]["GradFactorThresh"].as<float>();
}

} // namespace dso_ssl
