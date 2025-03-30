#include "optimize/PoseVertex.hpp"

namespace dso_ssl
{

void LeftSE3Vertex::oplusImpl(const double *update)
{
    Vector6d update_e;
    update_e << update[0], update[1], update[2], update[3], update[4], update[5];

    /// 左扰动
    _estimate = SE3d::exp(update_e) * _estimate;
}

/**
 * @brief 计算在优化一轮中，lm尝试中，是否接受过优化更新
 *
 * 0. 在判断之前，需要将上一轮统计的结果给重置掉
 * 1. 计算了accetp_，用来判断优化的这一轮是否被接受了
 * 2. 计算了last_accept_, 用来判断最后一次lm尝试使用cmoputeError是否有意义
 * 3. 将 lm_tries_accepts_ 栈中维护的内容清空了
 */
void LeftSE3Vertex::ComputeAccept()
{
    accept_ = false;
    last_accept_ = false;

    last_accept_ = lm_tries_accepts_.top();
    while (!lm_tries_accepts_.empty())
    {
        bool try_accept = lm_tries_accepts_.top();
        lm_tries_accepts_.pop();

        if (try_accept)
        {
            accept_ = true;
            std::stack<bool> temp_stack;
            std::swap(lm_tries_accepts_, temp_stack);
            return;
        }
    }
}

} // namespace dso_ssl
