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
    using Vector2Array = std::vector<Vector2f, Eigen::aligned_allocator<Vector2f>>;

    Pattern(const int &pattern_id);

    const Vector2Array &GetPattern() const { return pattern_; }

    static Vector2Array pattern1; ///< dso pattern1
    static Vector2Array pattern2; ///< dso pattern2
    static Vector2Array pattern3; ///< dso pattern3
    static Vector2Array pattern4; ///< dso pattern4
    static Vector2Array pattern5; ///< dso pattern5
    static Vector2Array pattern6; ///< dso pattern6
    static Vector2Array pattern7; ///< dso pattern7
    static Vector2Array pattern8; ///< dso pattern8
    static Vector2Array pattern9; ///< dso pattern9

private:
    Vector2Array pattern_; ///< 算法使用的pattern
};
} // namespace dso_ssl
