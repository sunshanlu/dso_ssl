#pragma once
#include <memory>
#include <vector>

#include <Eigen/Core>

namespace dso_ssl
{
class Pattern
{
public:
    using SharedPtr = std::shared_ptr<Pattern>;
    using Vector2f = Eigen::Matrix<float, 2, 1>;
    using Vector2fArray = std::vector<Vector2f, Eigen::aligned_allocator<Vector2f>>;

    Pattern(const int &pattern_id);

    // 获取pattern
    const Vector2fArray &GetPattern() const { return pattern_; }

    // 获取pattern的半尺寸
    const int &GetHalfPatternSize() const { return half_pattern_size_; }

    static Vector2fArray pattern1; ///< dso pattern1
    static Vector2fArray pattern2; ///< dso pattern2
    static Vector2fArray pattern3; ///< dso pattern3
    static Vector2fArray pattern4; ///< dso pattern4
    static Vector2fArray pattern5; ///< dso pattern5
    static Vector2fArray pattern6; ///< dso pattern6
    static Vector2fArray pattern7; ///< dso pattern7
    static Vector2fArray pattern8; ///< dso pattern8
    static Vector2fArray pattern9; ///< dso pattern9

private:
    Vector2fArray pattern_; ///< 算法使用的pattern
    int half_pattern_size_; ///< pattern的半尺寸，作用到点选器上
};
} // namespace dso_ssl
