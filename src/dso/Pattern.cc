#include "dso/Pattern.hpp"

namespace dso_ssl
{

Pattern::Pattern(const int &pattern_id)
{
    switch (pattern_id)
    {
    case 1:
        pattern_ = pattern1;
        half_pattern_size_ = 1;
        break;
    case 2:
        pattern_ = pattern2;
        half_pattern_size_ = 1;
        break;
    case 3:
        pattern_ = pattern3;
        half_pattern_size_ = 1;
        break;
    case 4:
        pattern_ = pattern4;
        half_pattern_size_ = 2;
        break;
    case 5:
        pattern_ = pattern5;
        half_pattern_size_ = 2;
        break;
    case 6:
        pattern_ = pattern6;
        half_pattern_size_ = 2;
        break;
    case 7:
        pattern_ = pattern7;
        half_pattern_size_ = 3;
        break;
    case 8:
        pattern_ = pattern8;
        half_pattern_size_ = 2;
        break;
    case 9:
        pattern_ = pattern9;
        half_pattern_size_ = 4;
        break;
    default:
        throw std::runtime_error("Pattern id is not supported");
    }
}

// clang-format off
Pattern::Vector2fArray Pattern::pattern1 = {
    Eigen::Vector2f( 0, -1),
    Eigen::Vector2f(-1,  0), Eigen::Vector2f( 0,  0), Eigen::Vector2f( 1,  0), 
    Eigen::Vector2f( 0,  1)
};
Pattern::Vector2fArray Pattern::pattern2 = {
    Eigen::Vector2f(-1, -1), Eigen::Vector2f( 1, -1), 
    Eigen::Vector2f( 0,  0),
    Eigen::Vector2f(-1,  1), Eigen::Vector2f( 1,  1)
};
Pattern::Vector2fArray Pattern::pattern3 = {
    Eigen::Vector2f(-1, -1), Eigen::Vector2f( 0, -1), Eigen::Vector2f( 1, -1),
    Eigen::Vector2f(-1,  0), Eigen::Vector2f( 0,  0), Eigen::Vector2f( 1,  0),  
    Eigen::Vector2f(-1,  1), Eigen::Vector2f( 0,  1), Eigen::Vector2f( 1,  1)
};
Pattern::Vector2fArray Pattern::pattern4 = {
    Eigen::Vector2f( 0, -2),
    Eigen::Vector2f(-1, -1), Eigen::Vector2f( 1, -1),
    Eigen::Vector2f(-2,  0), Eigen::Vector2f( 0,  0), Eigen::Vector2f( 2,  0),
    Eigen::Vector2f(-1,  1), Eigen::Vector2f( 1,  1),
    Eigen::Vector2f( 0,  2)
};
Pattern::Vector2fArray Pattern::pattern5 = {
    Eigen::Vector2f(-2, -2), Eigen::Vector2f( 0, -2), Eigen::Vector2f( 2, -2), 
    Eigen::Vector2f(-1, -1), Eigen::Vector2f( 1, -1),   
    Eigen::Vector2f(-2,  0), Eigen::Vector2f( 0,  0), Eigen::Vector2f( 2,  0),  
    Eigen::Vector2f(-1,  1), Eigen::Vector2f( 1,  1),    
    Eigen::Vector2f(-2,  2), Eigen::Vector2f( 0,  2), Eigen::Vector2f( 2,  2),  
};
Pattern::Vector2fArray Pattern::pattern6 = {
    Eigen::Vector2f(-2, -2), Eigen::Vector2f(-1, -2), Eigen::Vector2f( 0, -2), Eigen::Vector2f( 1, -2), Eigen::Vector2f( 2, -2),
    Eigen::Vector2f(-2, -1), Eigen::Vector2f(-1, -1), Eigen::Vector2f( 0, -1), Eigen::Vector2f( 1, -1), Eigen::Vector2f( 2, -1),
    Eigen::Vector2f(-2,  0), Eigen::Vector2f(-1,  0), Eigen::Vector2f( 0,  0), Eigen::Vector2f( 1,  0), Eigen::Vector2f( 2,  0),
    Eigen::Vector2f(-2,  1), Eigen::Vector2f(-1,  1), Eigen::Vector2f( 0,  1), Eigen::Vector2f( 1,  1), Eigen::Vector2f( 2,  1),
    Eigen::Vector2f(-2,  2), Eigen::Vector2f(-1,  2), Eigen::Vector2f( 0,  2), Eigen::Vector2f( 1,  2), Eigen::Vector2f( 2,  2)
};
Pattern::Vector2fArray Pattern::pattern7 = {
    Eigen::Vector2f(-1, -3), Eigen::Vector2f( 1, -3),
    Eigen::Vector2f(-2, -2), Eigen::Vector2f( 0, -2), Eigen::Vector2f( 2, -2),
    Eigen::Vector2f(-3, -1), Eigen::Vector2f(-1, -1), Eigen::Vector2f( 1, -1), Eigen::Vector2f( 3, -1),
    Eigen::Vector2f(-2,  0), Eigen::Vector2f( 0,  0), Eigen::Vector2f( 2,  0),
    Eigen::Vector2f(-3,  1), Eigen::Vector2f(-1,  1), Eigen::Vector2f( 1,  1), Eigen::Vector2f( 3,  1),
    Eigen::Vector2f(-2,  2), Eigen::Vector2f( 0,  2), Eigen::Vector2f( 2,  2),
    Eigen::Vector2f(-1,  3), Eigen::Vector2f( 1,  3)
};
Pattern::Vector2fArray Pattern::pattern8 = {
    Eigen::Vector2f( 0, -2),
    Eigen::Vector2f(-1, -1), Eigen::Vector2f( 1, -1),
    Eigen::Vector2f(-2,  0), Eigen::Vector2f( 0,  0), Eigen::Vector2f( 2,  0),
    Eigen::Vector2f(-1,  1),
    Eigen::Vector2f( 0,  2)
};  
Pattern::Vector2fArray Pattern::pattern9 = {
    Eigen::Vector2f(-4, -4), Eigen::Vector2f(-2, -4), Eigen::Vector2f( 0, -4), Eigen::Vector2f( 2, -4), Eigen::Vector2f( 4, -4),
    Eigen::Vector2f(-4, -2), Eigen::Vector2f(-2, -2), Eigen::Vector2f( 0, -2), Eigen::Vector2f( 2, -2), Eigen::Vector2f( 4, -2),
    Eigen::Vector2f(-4,  0), Eigen::Vector2f(-2,  0), Eigen::Vector2f( 0,  0), Eigen::Vector2f( 2,  0), Eigen::Vector2f( 4,  0),
    Eigen::Vector2f(-4,  2), Eigen::Vector2f(-2,  2), Eigen::Vector2f( 0,  2), Eigen::Vector2f( 2,  2), Eigen::Vector2f( 4,  2),
    Eigen::Vector2f(-4,  4), Eigen::Vector2f(-2,  4), Eigen::Vector2f( 0,  4), Eigen::Vector2f( 2,  4), Eigen::Vector2f( 4,  4),
};

// clang-format on

} // namespace dso_ssl