#pragma once

#include <tbb/parallel_for.h>

namespace parallel
{

/**
 * @brief TBB和SIMD 并行化封装
 * 
 * 要求SIMD函数接收一个起始start,内部负责计算SSE，并做结果存储（一般会处理[start, start + 3]）
 * 要求Single函数接收一个索引，处理单个数据的函数
 * 
 * @tparam SIMD         SIMD函数模板类型
 * @tparam Single       不足4个时，使用单线程处理
 * @param begin         处理索引的起始点
 * @param end           处理索引的终止点
 * @param gain_size     每个块的最小的索引数量（TBB会经过负载计算一个合理的块数量）
 * @param ProcessSIMD   处理SIMD函数
 * @param ProcessSingle 处理单线程函数
 */
template <typename SIMD, typename Single>
void ParallelWrapper(const int &begin, const int &end, const int &gain_size, SIMD &&ProcessSIMD, Single &&ProcessSingle)
{
    auto process_func = [&](const tbb::blocked_range<int> &range)
    {
        int remains = (range.end() - range.begin()) % 4;
        for (int start = range.begin(); start < range.end() - 3; start += 4)
            ProcessSIMD(start);

        // 处理
        for (int idx = 1; idx <= remains; ++idx)
            ProcessSingle(range.end() - idx);
    };

    tbb::parallel_for(tbb::blocked_range<int>(begin, end, gain_size), process_func);
}

} // namespace parallel