#include "optimize/IdepthVertex.hpp"
#include "dso/Initializer.hpp"

namespace dso_ssl
{

IdepthVertex::IdepthVertex(PixelPointPtr pixel_point)
    : pixel_point_(std::move(pixel_point))
    , accept_(false)
    , accepted_times_(0)
    , accumulate_times_(0)
{
    if (!pixel_point_)
        throw std::runtime_error("pixel_point is nullptr");

    _estimate = pixel_point_->idepth_;
    pixel_position_ = pixel_point_->pixel_position_;
    pixel_point_->idepth_vertex_ = this;
}

/**
 * @brief 更新逆深度状态
 *
 * 1. 要求逆深度状态的更新量要小于 maxstep，maxstep的值对应到j帧像素坐标系上为0.25px
 * 2. maxstep是由 InitializerPhotoResidual 计算雅可比矩阵的过程中得到的 0.25 / (dpj / ddpi).norm
 *
 * @param update 输入的更新量
 */
void IdepthVertex::oplusImpl(const double *update)
{
    double update_value = *update;

    if (update_value > maxstep_)
        update_value = maxstep_;

    _estimate += update_value;
    if (_estimate >= 50)
        _estimate = 50;
    if (_estimate <= 1e-3)
        _estimate = 1e-3;
}

/**
 * @brief 计算在这一次优化过程中，逆深度是否会被更新（lm的所有尝试过程中）
 *
 * 1. accept_ 信息的重置
 * 2. 判断accept_，在这次优化中是否被更新
 * 3. 如果更新，需要同步维护 accepted_times_ 在当前层的所有优化过程中，被接受的次数
 * 4. 如果更新，需要同步维护 accumulate_times_ 在这个初始化过程中 ，被接受的次数
 */
void IdepthVertex::ComputeAccept()
{
    accept_ = false;
    while (!lm_tries_accepts_.empty())
    {
        bool try_accept = lm_tries_accepts_.top();
        lm_tries_accepts_.pop();
        if (try_accept)
        {
            accept_ = true;
            ++accepted_times_;
            ++accumulate_times_;
            std::stack<bool> temp_tries;
            std::swap(lm_tries_accepts_, temp_tries);
            return;
        }
    }
}

} // namespace dso_ssl