#pragma once

#include <stack>

#include <g2o/core/base_vertex.h>
#include <memory>
#include <sophus/se3.hpp>

namespace dso_ssl
{

/// 左扰动位姿顶点
class LeftSE3Vertex : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Vector6d = Eigen::Matrix<double, 6, 1>;
    using SharedPtr = std::shared_ptr<LeftSE3Vertex>;
    using SE3d = Sophus::SE3d;

    LeftSE3Vertex()
        : accept_(false)
        , last_accept_(false)
    {
    }

    bool read(std::istream &is) override { return false; }

    bool write(std::ostream &os) const override { return false; }

    /// 定义位姿部分的更新
    void oplusImpl(const double *update) override;

    void setToOriginImpl() override { _estimate = SE3d(); }

    /// 判断这一轮优化是否被接受了
    const bool &Accept() const { return accept_; }

    /// 判断lm最后一次尝试使用的ComputeError方法是否有意义
    const bool &LastAccept() const { return last_accept_; }

    void pop() override
    {
        BaseVertex::pop();
        lm_tries_accepts_.push(false);
    }

    void discardTop() override
    {
        BaseVertex::discardTop();
        lm_tries_accepts_.push(true);
    }

    /**
     * @brief 计算在优化一轮中，lm尝试中，是否接受过优化更新
     *
     * 0. 在判断之前，需要将上一轮统计的结果给重置掉
     * 1. 计算了accetp_，用来判断优化的这一轮是否被接受了
     * 2. 计算了last_accept_, 用来判断最后一次lm尝试使用cmoputeError是否有意义
     * 3. 将 lm_tries_accepts_ 栈中维护的内容清空了
     */
    void ComputeAccept();

private:
    bool last_accept_;                  ///< 在lm优化过程中，最后一次的尝试是否接受更新
    bool accept_;                       ///< 在一轮lm优化过程中，是否产生的接受当前更新条件
    std::stack<bool> lm_tries_accepts_; ///< 在一轮lm尝试过程中，接受是否接受更新的判断
};

/// 右扰动位姿顶点
class RightSE3Vertex : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Vector6d = Eigen::Matrix<double, 6, 1>;
    using SharedPtr = std::shared_ptr<RightSE3Vertex>;
    using SE3d = Sophus::SE3d;

    RightSE3Vertex(SE3d init_estimate) { _estimate = std::move(init_estimate); }

    /// 定义位姿部分的更新
    void oplusImpl(const double *update) override
    {
        Vector6d update_e;
        update_e << update[0], update[1], update[2], update[3], update[4], update[5];

        /// 右扰动
        _estimate = _estimate * SE3d::exp(update_e);
    }

    void setToOriginImpl() override { _estimate = SE3d(); }
};

} // namespace dso_ssl