#pragma once

#include <algorithm>
#include <random>

namespace stl_opt
{
/**
 * @brief 通过随机打乱索引的方式从向量中随机移除指定数量的元素
 *
 * 该函数通过随机种子生成一个随机数生成器，用于打乱元素索引，
 * 然后根据需要保留的元素数量从打乱后的索引中选择元素，达到随机移除元素的效果。
 * 通过使用相同的seed可以满住有索引对应关系的所个vector实现相同位置的元素剔除
 *
 * @tparam T            向量中元素的类型
 * @tparam Alloc        向量使用的内存分配器，默认为std::allocator<T>
 * @param vec           待操作的向量
 * @param num_remains   指定需要保留的元素数量，必须小于向量的当前大小
 * @param seed          随机数生成器的种子，用于产生随机性
 * @throws std::runtime_error 如果num_remains大于等于向量的大小，则抛出运行时错误
 */
template <typename T, typename Alloc = std::allocator<T>>
void RemoveItemRandom(std::vector<T, Alloc> &vec, const int &num_remains, unsigned int seed)
{
    if (num_remains >= vec.size())
        throw std::runtime_error("num_remains >= vec.size()");

    std::vector<int> indices(vec.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    // 计算要删除点的数量，并进行随机删除
    int remove_num = vec.size() - num_remains;

    std::vector<T, Alloc> vec_temp;

    for (int idx = remove_num; idx < vec.size(); ++idx)
    {
        vec_temp.push_back(vec[indices[idx]]);
    }

    std::swap(vec_temp, vec);
}
} // namespace stl_opt