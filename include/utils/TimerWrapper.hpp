#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include <tabulate/table.hpp>

using namespace std::chrono_literals;

namespace timer
{

class TimerWrapper
{
public:
    using SharedPtr = std::shared_ptr<TimerWrapper>;

    TimerWrapper(std::string wrapper_name)
        : wrapper_name_(std::move(wrapper_name))
    {
    }

    /**
     * 执行函数并测量其执行时间
     *
     * 本函数用于执行给定的函数并测量其执行时间，同时记录函数名称和执行时间
     * 主要用于性能分析和调试目的
     *
     * @tparam F 要执行的函数的类型
     * @tparam Args 函数参数的类型
     * @param name 函数的名称，用于在性能分析时标识函数
     * @param func 要执行的函数对象
     * @param args 函数的参数，使用完美转发保持原始类型
     * @return 执行函数的结果
     */
    template <typename F, typename... Args>
    auto ExecuteAndMeasure(const std::string &name, F &&func, Args &&...args)
    {
        auto duration = 999999ms;
        decltype(std::forward<F>(func)(std::forward<Args>(args)...)) result;

        /// todo 异常捕获部分在debug的时候不好用，先注释掉，以后在放开。
        // try
        // {
        //     auto start = std::chrono::high_resolution_clock::now();
        //     result = std::forward<F>(func)(std::forward<Args>(args)...);
        //     auto end = std::chrono::high_resolution_clock::now();
        //     duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // }
        // catch (const std::exception &e)
        // {
        //     throw;
        // }

        auto start = std::chrono::high_resolution_clock::now();
        result = std::forward<F>(func)(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            names_.push_back(name);
            durations_.push_back(duration);
        }

        return result;
    }

    /// 生成时间消耗表格
    void TimerShow(std::ostream &stream = std::cout) const;

private:
    std::string wrapper_name_;                         ///< 时间测试名称
    std::vector<std::string> names_;                   ///< 维护函数名称
    std::vector<std::chrono::milliseconds> durations_; ///< 维护函数耗时
    std::mutex mutex_;                                 ///< 维护类的线程安全
};

} // namespace timer
