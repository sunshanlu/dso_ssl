#include "utils/Project.hpp"

namespace project
{

/**
 * @brief 以SSE指令集类型为输入输出表示 像素坐标到归一化坐标系
 *
 * @param u_sse         输入的像素坐标x
 * @param v_sse         输入的像素坐标y
 * @param fx_inv_sse    输入的 1.0 / fx
 * @param fy_inv_sse    输入的 1.0 / fy
 * @param cx_inv_sse    输入的 cx / fx
 * @param cy_inv_sse    输入的 cy / fy
 */
void Pixel2NormSSEInternal(const __m128 &u_sse, const __m128 &v_sse, const __m128 &fx_inv_sse, const __m128 &fy_inv_sse, const __m128 &cx_inv_sse,
                           const __m128 &cy_inv_sse, __m128 &result_x_sse, __m128 &result_y_sse)
{
    result_x_sse = _mm_sub_ps(_mm_mul_ps(u_sse, fx_inv_sse), cx_inv_sse);
    result_y_sse = _mm_sub_ps(_mm_mul_ps(v_sse, fy_inv_sse), cy_inv_sse);
}

/**
 * @brief 以SSE指令集类型为输入输出表示 归一化坐标系到像素坐标
 *
 * @param x_norm_sse    输入的归一化坐标x
 * @param y_norm_sse    输入的归一化坐标y
 * @param fx_sse        输入的 fx
 * @param fy_sse        输入的 fy
 * @param cx_sse        输入的 cx
 * @param cy_sse        输入的 cy
 */
void Norm2PixelSSEInternal(const __m128 &x_norm_sse, const __m128 &y_norm_sse, const __m128 &fx_sse, const __m128 &fy_sse, const __m128 &cx_sse,
                           const __m128 &cy_sse, __m128 &u_sse, __m128 &v_sse)
{
    u_sse = _mm_add_ps(_mm_mul_ps(x_norm_sse, fx_sse), cx_sse);
    v_sse = _mm_add_ps(_mm_mul_ps(y_norm_sse, fy_sse), cy_sse);
}

/**
 * @brief 以SSE指令集类型为输入输出表示 3d坐标到归一化坐标
 *
 * @param x_data_sse 输入的 3d坐标x
 * @param y_data_sse 输入的 3d坐标y
 * @param z_data_sse 输入的 3d坐标z
 */
void Point2NormSSEInternal(const __m128 &x_data_sse, const __m128 &y_data_sse, const __m128 &z_data_sse, __m128 &result_x_sse, __m128 &result_y_sse)
{
    result_x_sse = _mm_div_ps(x_data_sse, z_data_sse);
    result_y_sse = _mm_div_ps(y_data_sse, z_data_sse);
}

/**
 * @brief 使用SSE加速，将4个像素坐标系反向投影到归一化坐标系下
 *
 * @param pixels    输入的待投影的像素坐标
 * @param fx_inv    输入的 1.0 / fx
 * @param fy_inv    输入的 1.0 / fy
 * @param cx_inv    输入的 cx / fx
 * @param cy_inv    输入的 cy / fy
 * @return Vector3fArray    输出的投影完成的归一化坐标
 *
 * @see Pixel2NormSSEInternal()
 */
Vector3fArray Pixel2NormSSE(const Vector3fArray &pixels, const float &fx_inv, const float &fy_inv, const float &cx_inv, const float &cy_inv)
{
    if (pixels.size() != 4)
        throw std::runtime_error("pixels.size() != 4");

    alignas(16) float u[4], v[4];
    alignas(16) float result_x[4], result_y[4];

    __m128 fx_inv_sse = _mm_set1_ps(fx_inv);
    __m128 fy_inv_sse = _mm_set1_ps(fy_inv);
    __m128 cx_inv_sse = _mm_set1_ps(cx_inv);
    __m128 cy_inv_sse = _mm_set1_ps(cy_inv);

    for (int idx = 0; idx < 4; ++idx)
    {
        u[idx] = pixels[idx][0];
        v[idx] = pixels[idx][1];
    }

    __m128 u_sse = _mm_load_ps(u);
    __m128 v_sse = _mm_load_ps(v);

    __m128 result_x_sse, result_y_sse;
    Pixel2NormSSEInternal(u_sse, v_sse, fx_inv_sse, fy_inv_sse, cx_inv_sse, cy_inv_sse, result_x_sse, result_y_sse);

    _mm_store_ps(result_x, result_x_sse);
    _mm_store_ps(result_y, result_y_sse);

    Vector3fArray result(4);
    for (int idx = 0; idx < 4; ++idx)
        result[idx] = Eigen::Vector3f(result_x[idx], result_y[idx], 1.0);

    return result;
}

/**
 * @brief 该函数使用SSE指令集加速计算，将4个归一化坐标系坐标转换为图像空间的像素坐标。
 *
 * @param norms 归一化坐标系坐标vec，包含4个
 * @param fx    输入的fx
 * @param fy    输入的fy
 * @param cx    输入的cx
 * @param cy    输入的cy
 * @return Vector2fArray 返回一个包含4个像素坐标的数组
 *
 * @see Norm2PixelSSEInternal()
 */
Vector2fArray Norm2PixelSSE(const Vector3fArray &norms, const float &fx, const float &fy, const float &cx, const float &cy)
{
    if (norms.size() != 4)
        throw std::runtime_error("norms.size() != 4");

    __m128 fx_sse = _mm_set1_ps(fx);
    __m128 fy_sse = _mm_set1_ps(fy);
    __m128 cx_sse = _mm_set1_ps(cx);
    __m128 cy_sse = _mm_set1_ps(cy);

    alignas(16) float x_norm[4], y_norm[4];
    alignas(16) float result_u[4], result_v[4];
    for (int idx = 0; idx < 4; ++idx)
    {
        x_norm[idx] = norms[idx][0];
        y_norm[idx] = norms[idx][1];
    }

    __m128 x_norm_sse = _mm_load_ps(x_norm);
    __m128 y_norm_sse = _mm_load_ps(y_norm);

    __m128 u_sse, v_sse;
    Norm2PixelSSEInternal(x_norm_sse, y_norm_sse, fx_sse, fy_sse, cx_sse, cy_sse, u_sse, v_sse);

    _mm_store_ps(result_u, u_sse);
    _mm_store_ps(result_v, v_sse);

    Vector2fArray result(4);
    for (int idx = 0; idx < 4; ++idx)
        result[idx] = Eigen::Vector2f(result_u[idx], result_v[idx]);

    return result;
}

/**
 * @brief 使用SSE指令集加速计算，将4个点转换为归一化坐标系下的坐标。
 *
 * @param points    输入的点vec，包含4个点
 * @return Vector3fArray 输出的归一化坐标vec，包含4个点
 *
 * @see Point2NormSSEInternal()
 */
Vector3fArray Point2NormSSE(const Vector3fArray &points)
{
    if (points.size() != 4)
        throw std::runtime_error("points.size() != 4");

    alignas(16) float x_data[4], y_data[4], z_data[4];
    alignas(16) float result_x[4], result_y[4];

    for (int idx = 0; idx < 4; ++idx)
    {
        x_data[idx] = points[idx][0];
        y_data[idx] = points[idx][1];
        z_data[idx] = points[idx][2];
    }

    __m128 x_data_sse = _mm_load_ps(x_data);
    __m128 y_data_sse = _mm_load_ps(y_data);
    __m128 z_data_sse = _mm_load_ps(z_data);

    __m128 x_norm_sse, y_norm_sse;
    Point2NormSSEInternal(x_data_sse, y_data_sse, z_data_sse, x_norm_sse, y_norm_sse);

    _mm_store_ps(result_x, x_norm_sse);
    _mm_store_ps(result_y, y_norm_sse);

    Vector3fArray result(4);
    for (int idx = 0; idx < 4; ++idx)
        result[idx] = Eigen::Vector3f(result_x[idx], result_y[idx], 1.0);

    return result;
}

/**
 * @brief 使用SSE指令集加速计算，将i下的像素坐标系到j像素坐标系的转换
 *
 * @param Tji       输入的Tji
 * @param pixel_in  输入的i下的像素坐标vec，包含4个像素坐标
 * @param dpi       输入的dpi，一个pattern共享逆深度值
 * @param fx        输入的fx
 * @param fy        输入的fy
 * @param cx        输入的cx
 * @param cy        输入的cy
 * @param p         输出的Pj' 的vec，作为中间变量，包含4个坐标，Rji * pi_norm + tji * dpi
 * @param pixel_out 输出的j下的像素坐标vec，包含4个像素坐标
 */
void Pixel2PixelSSE(const Sophus::SE3f &Tji, const Vector2fArray &pixel_in, const float &dpi, const float &fx, const float &fy, const float &cx,
                    const float &cy, Vector3fArray &p, Vector2fArray &pixel_out)
{
    if (pixel_in.size() != 4)
        throw std::runtime_error("pixel_in.size() != 4");

    if (fx == 0 || fy == 0)
        throw std::runtime_error("fx == 0 || fy == 0");

    // 构造相机内参相关的SSE向量
    __m128 fx_inv_sse = _mm_set1_ps(1.0 / fx);
    __m128 fy_inv_sse = _mm_set1_ps(1.0 / fy);
    __m128 cx_inv_sse = _mm_set1_ps(cx / fx);
    __m128 cy_inv_sse = _mm_set1_ps(cy / fy);
    __m128 fx_sse = _mm_set1_ps(fx);
    __m128 fy_sse = _mm_set1_ps(fy);
    __m128 cx_sse = _mm_set1_ps(cx);
    __m128 cy_sse = _mm_set1_ps(cy);

    alignas(16) float u_data[4], v_data[4];
    alignas(16) float x_data[4], y_data[4], z_data[4];
    for (int idx = 0; idx < 4; ++idx)
    {
        u_data[idx] = pixel_in[idx][0];
        v_data[idx] = pixel_in[idx][1];
    }

    __m128 u_sse = _mm_load_ps(u_data);
    __m128 v_sse = _mm_load_ps(v_data);

    // pi向归一化坐标系下转换
    __m128 x_norm_i_sse, y_norm_i_sse;
    Pixel2NormSSEInternal(u_sse, v_sse, fx_inv_sse, fy_inv_sse, cx_inv_sse, cy_inv_sse, x_norm_i_sse, y_norm_i_sse);

    // pi的归一化考虑相对位姿影响到pj坐标系下
    __m128 temp_data_x, temp_data_y, temp_data_z;
    Point2PointTempSSEInternal(Tji, x_norm_i_sse, y_norm_i_sse, dpi, temp_data_x, temp_data_y, temp_data_z);
    _mm_store_ps(x_data, temp_data_x);
    _mm_store_ps(y_data, temp_data_y);
    _mm_store_ps(z_data, temp_data_z);

    // 保存中间变量 Pj'
    p = Vector3fArray(4);
    for (int idx = 0; idx < 4; ++idx)
        p[idx] = Eigen::Vector3f(x_data[idx], y_data[idx], z_data[idx]);

    // Pj'向归一化坐标系变换
    __m128 x_norm_j_sse, y_norm_j_sse;
    Point2NormSSEInternal(temp_data_x, temp_data_y, temp_data_z, x_norm_j_sse, y_norm_j_sse);

    // Pj'归一化坐标系向pj像素坐标系变换
    __m128 result_pixel_u_j, result_pixel_v_j;
    Norm2PixelSSEInternal(x_norm_j_sse, y_norm_j_sse, fx_sse, fy_sse, cx_sse, cy_sse, result_pixel_u_j, result_pixel_v_j);

    // 保存输出pj像素坐标
    alignas(16) float result_uj[4], result_vj[4];
    _mm_store_ps(result_uj, result_pixel_u_j);
    _mm_store_ps(result_vj, result_pixel_v_j);
    pixel_out = Vector2fArray(4);
    for (int idx = 0; idx < 4; ++idx)
        pixel_out[idx] = Eigen::Vector2f(result_uj[idx], result_vj[idx]);
}

/**
 * @brief 使用SSE加速，计算点到临时点的变换
 *
 * @param Tji           输入的ji之间的相对变换
 * @param x_norm_i_sse  输入的pi归一化坐标x
 * @param y_norm_i_sse  输入的pi归一化坐标y
 * @param dpi           输入的逆深度dpi
 */
void Point2PointTempSSEInternal(const Sophus::SE3f &Tji, const __m128 &x_norm_i_sse, const __m128 &y_norm_i_sse, const float &dpi, __m128 &x_data_sse,
                                __m128 &y_data_sse, __m128 &z_data_sse)
{
    // 构造旋转矩阵和平移向量相关的SSE向量
    auto Rji = Tji.so3().matrix();
    auto tji = Tji.translation();
    __m128 r00_sse = _mm_set1_ps(Rji(0, 0)), r01_sse = _mm_set1_ps(Rji(0, 1)), r02_sse = _mm_set1_ps(Rji(0, 2));
    __m128 r10_sse = _mm_set1_ps(Rji(1, 0)), r11_sse = _mm_set1_ps(Rji(1, 1)), r12_sse = _mm_set1_ps(Rji(1, 2));
    __m128 r20_sse = _mm_set1_ps(Rji(2, 0)), r21_sse = _mm_set1_ps(Rji(2, 1)), r22_sse = _mm_set1_ps(Rji(2, 2));
    __m128 t0_sse = _mm_set1_ps(tji[0]), t1_sse = _mm_set1_ps(tji[1]), t2_sse = _mm_set1_ps(tji[2]);
    __m128 dpi_sse = _mm_set1_ps(dpi);
    __m128 one_sse = _mm_set1_ps(1.0);

    // Pi^{norm}转换到Pj'
    x_data_sse = _mm_add_ps(_mm_add_ps(_mm_mul_ps(r00_sse, x_norm_i_sse), _mm_mul_ps(r01_sse, y_norm_i_sse)), _mm_mul_ps(r02_sse, one_sse));
    y_data_sse = _mm_add_ps(_mm_add_ps(_mm_mul_ps(r10_sse, x_norm_i_sse), _mm_mul_ps(r11_sse, y_norm_i_sse)), _mm_mul_ps(r12_sse, one_sse));
    z_data_sse = _mm_add_ps(_mm_add_ps(_mm_mul_ps(r20_sse, x_norm_i_sse), _mm_mul_ps(r21_sse, y_norm_i_sse)), _mm_mul_ps(r22_sse, one_sse));
    x_data_sse = _mm_add_ps(x_data_sse, _mm_mul_ps(t0_sse, dpi_sse));
    y_data_sse = _mm_add_ps(y_data_sse, _mm_mul_ps(t1_sse, dpi_sse));
    z_data_sse = _mm_add_ps(z_data_sse, _mm_mul_ps(t2_sse, dpi_sse));
}

/**
 * @brief 使用SSE加速，计算点到临时点的变换
 *
 * @param pnorm_i   输入的pi的归一化坐标系，4个
 * @param Tji       输入的ji之间的相对位姿
 * @param dpi       输入的pi归一化点对应的逆深度
 * @return Vector3fArray 输出的临时变量点Pj'
 *
 * @see Point2PointTempSSEInternal()
 */
Vector3fArray Point2PointTempSSE(const Vector3fArray &pnorm_i, const Sophus::SE3f &Tji, const float &dpi)
{
    if (pnorm_i.size() != 4)
        throw std::runtime_error("pnorm_i.size() != 4");

    alignas(16) float x_norm_i_data[4], y_norm_i_data[4];
    alignas(16) float x_data[4], y_data[4], z_data[4];

    for (int idx = 0; idx < 4; ++idx)
    {
        x_norm_i_data[idx] = pnorm_i[idx][0];
        y_norm_i_data[idx] = pnorm_i[idx][1];
    }

    __m128 x_norm_i_sse = _mm_load_ps(x_norm_i_data);
    __m128 y_norm_i_sse = _mm_load_ps(y_norm_i_data);

    __m128 temp_data_x, temp_data_y, temp_data_z;
    Point2PointTempSSEInternal(Tji, x_norm_i_sse, y_norm_i_sse, dpi, temp_data_x, temp_data_y, temp_data_z);

    _mm_store_ps(x_data, temp_data_x);
    _mm_store_ps(y_data, temp_data_y);
    _mm_store_ps(z_data, temp_data_z);

    Vector3fArray result(4);
    for (int idx = 0; idx < 4; ++idx)
        result[idx] = Eigen::Vector3f(x_data[idx], y_data[idx], z_data[idx]);

    return result;
}

} // namespace project
