#pragma once
#include <memory>
#include <stack>

#include <g2o/core/base_vertex.h>

namespace dso_ssl
{
class PixelPoint;

/// 逆深度顶点
class IdepthVertex : public g2o::BaseVertex<1, double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Vector2f = Eigen::Vector2f;
    using PixelPointPtr = std::shared_ptr<PixelPoint>;

    IdepthVertex(PixelPointPtr pixel_point);

    void oplusImpl(const double *update) override;

    bool read(std::istream &is) override { return false; }

    bool write(std::ostream &os) const override { return false; }

    void setToOriginImpl() override { _estimate = 0; }

    const bool &Accept() const { return accept_; }

    void SetMaxstep(const double &maxstep)
    {
        maxstep_ = maxstep;
    }

    /// 获取某一帧接受更新的次数
    const int &AcceptTimes() const { return accepted_times_; }

    /// 获取整个初始化过程中的累计接受更新次数
    const int &AccumalateTimes() const { return accumulate_times_; }

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
     * @brief 计算在这一次优化过程中，逆深度是否会被更新（lm的所有尝试过程中）
     *
     * 1. accept_ 信息的重置
     * 2. 判断accept_，在这次优化中是否被更新
     * 3. 如果更新，需要同步维护 accepted_times_ 在当前层的所有优化过程中，被接受的次数
     * 4. 如果更新，需要同步维护 accumulate_times_ 在这个初始化过程中 ，被接受的次数
     */
    void ComputeAccept();

    Vector2f pixel_position_;           ///< host帧像素位置
    PixelPointPtr pixel_point_;         ///< 维护的像素点信息
    double maxstep_;                    ///< 维护0.25的pj像素位置对应的逆深度更新量
    int accepted_times_;                ///< 针对某个普通帧的某一层逆深度更新被接受的次数，新帧来到会更新
    bool accept_;                       ///< 是否接受了某个普通帧的某一层最新的逆深度更新，新帧来到会更新
    std::stack<bool> lm_tries_accepts_; ///< LM算法的尝试中，是否接受当前更新
    int accumulate_times_;              ///< 针对初始化过程中，逆深度更新被接受的次数
};

} // namespace dso_ssl
