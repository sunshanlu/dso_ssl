#pragma once
#include <memory>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "dso/Frame.hpp"
#include "dso/Keyframe.hpp"

namespace dso_ssl
{

struct TrackIdpethPoint
{
  using SharedPtr = std::shared_ptr<TrackIdpethPoint>;
  using Vec2f = Eigen::Vector2f;

  Vec2f ref_pixel_point_; ///< 参考帧的像素坐标
  float ref_idepth_;      ///< 参考帧的逆深度点
};

class Tracker
{
public:
  using SharedPtr = std::shared_ptr<Tracker>;
  using SE3f = Sophus::SE3f;
  using Mat3f = Eigen::Matrix3f;
  using Vec3f = Eigen::Vector3f;
  using Vec2i = Eigen::Vector2i;
  using Vec2f = Eigen::Vector2f;
  using Vec8d = Eigen::Vector<double, 8>;
  using Vec8f = Eigen::Vector<float, 8>;
  using Mat8d = Eigen::Matrix<double, 8, 8>;
  using Mat8f = Eigen::Matrix<float, 8, 8>;
  using Vec6f = Eigen::Vector<float, 6>;
  using TrackerIdepthPoints = std::vector<std::vector<TrackIdpethPoint::SharedPtr>>;

  struct Options
  {
    using SharedPtr = std::shared_ptr<Options>;

    Options(const std::string &filepath);

    int pyra_levels_;                 ///< 使用的图像金字塔层数
    float huber_threshold_;           ///< 优化过程中规定的huber阈值
    std::vector<int> max_iterations_; ///< 定义每层最大迭代次数
    bool verbose_;                    ///< 定义是否verbose输出
    float outlier_threshold_;         ///< 外点阈值
    float time_interval_;             ///< 创建关键帧的时间间隔
    float shift_weight_t_;            ///< 仅平移光流权重
    float shift_weight_rt_;           ///< 位姿光流权重
    float shift_treshold_;            ///< 光流阈值，关键帧条件
    float max_exp_aji_threshold_;     ///< 相对exp_aji的最大阈值
    float min_exp_aji_threshold_;     ///< 相对exp_aji的最小阈值
    float rmse_threshold_factor_;     ///< 相对于参考第一帧的rmse阈值系数
  };

  struct RefFrameState
  {
    RefFrameState()
        : ai(0.f)
        , bi(0.f)
        , ti(1.f)
    {
      Mat3f R = Mat3f::Identity();
      Vec3f t = Vec3f::Zero();
      Trw_ = SE3f(R, t);
    }

    SE3f Trw_;    ///< 参考帧位姿
    float ai, bi; ///< 参考帧绝对光度仿射参数
    float ti;     ///< 参考帧曝光时间
  };

  Tracker(Options::SharedPtr options)
      : options_(std::move(options))
      , curr_frame_(nullptr)
      , ref_keyframe_(nullptr)
      , track_idepth_points_(options_->pyra_levels_)
      , track_fx_(options_->pyra_levels_)
      , track_fy_(options_->pyra_levels_)
      , track_cx_(options_->pyra_levels_)
      , track_cy_(options_->pyra_levels_)
      , ref_keyframe_self_(nullptr)
      , last_rmse_(options_->pyra_levels_, 0.f)
      , abort_rmse_(options_->pyra_levels_, std::numeric_limits<float>::max())
      , first_ref_rmse_(options_->pyra_levels_, std::numeric_limits<float>::max())
      , mapper_changed_(true)
      , refframe_changed_(true)
      , last_ref_keyframe_(nullptr)
  {

    logger_ = spdlog::stdout_color_mt("Tracker");

    // 初始化恒速、倍速、半速、静止和相对参考帧静止
    for (int idx = 0; idx < 5; ++idx)
      velocity_tries_.push_back(Vec6f::Zero());

    // 生成小转角尝试
    Vec3f try_out;
    GenerateTries(try_out, 0);
  }

  /**
   * @brief 当后端优化完成后，更新Tracker跟踪器信息，以匹配最新优化状态，后端线程调用
   *
   * 1. 更新参考关键帧、滑动窗口状态等信息
   * 2. 更新相机的内参矩阵
   *
   * @param sliding_window 输入的滑动窗口，用于向最新关键帧投影
   * @param fx             后端优化后的fx
   * @param fy             后端优化后的fy
   * @param cx             后端优化后的cx
   * @param cy             后端优化后的cy
   */
  void UpdateTracker(const std::vector<KeyFrame::SharedPtr> &sliding_window, const float &fx, const float &fy, const float &cx, const float &cy);

  /**
   * @brief 构建参考点，将构造的参考点放到 track_idepth_points中
   *
   * 1. 将滑动窗口中的关键帧逆深度点，投影到最新的关键帧上，得到第0层逆深度点分布状态
   *  1.1 投影过程需要考虑四舍五入，因此存在多个投影点对应一个投影点的情况，使用高斯归一化积
   *  1.2 逆深度点，向上投影，得到金字塔层级上的逆深度点分布情况
   * 2. Tracker中弃用pattern,使用逆深度点膨胀的方式实现跟踪点的扩充
   *  2.1 0层和1层使用斜侧膨胀方式，对于多个逆深度情况，使用高斯归一化积
   *  2.2 2层及以上使用上下左右膨胀方式，对于多个逆深度情况，使用高斯归一化积
   *
   * @param sliding_window       输入的滑动窗口，用于向最新关键帧投影
   */
  void BuildTrackerPoints(const std::vector<KeyFrame::SharedPtr> &sliding_window);

  /**
   * @brief 更新Tracker跟踪器的内参矩阵
   *
   * @param fx 输入的后端更新后的fx
   * @param fy 输入的后端更新后的fy
   * @param cx 输入的后端更新后的cx
   * @param cy 输入的后端更新后的cy
   */
  void UpdateCalib(const float &fx, const float &fy, const float &cx, const float &cy);

  /**
   * @brief 将滑动窗口中的关键帧向最新关键帧进行投影
   *
   * 1. 要求关键帧的逆深度点状态里面没有外点
   * 2. 使用的逆深度点时、Tcw等滑动窗口状态时，不能在逆深度更新过程中
   * 3. 由于使用四舍五入，因此存在多对一的情况，使用高斯归一化积
   *
   * @param sliding_window          输入的滑动窗口，用于逆深度点投影
   * @param ref_idepth_sum          输出的逆深度点高斯归一化积的分布情况，第0层
   * @param ref_hessian_sum         输出的逆深度点hessian的分布情况，第0层
   */
  void ProjectWindow2Ref(const std::vector<KeyFrame::SharedPtr> &sliding_window, cv::Mat &ref_idepth_sum, cv::Mat &ref_hessian_sum);

  /**
   * @brief 基于当前层的idpeth_sum和hessian_sum信息，向上一层进行投影
   *
   * @param curr_idepth_sum   输入的当前金字塔idepth_sum
   * @param curr_hessian_sum  输入的当前金字塔hessian_sum
   * @param prev_idepth_sum   输出的上一层金字塔idepth_sum
   * @param prev_hessian_sum  输出的上一层金字塔hessian_sum
   */
  void PropagateUp(const cv::Mat &curr_idepth_sum, const cv::Mat &curr_hessian_sum, cv::Mat &prev_idepth_sum, cv::Mat &prev_hessian_sum);

  /**
   * @brief 逆深度点的膨胀，用于在track中取代pattern
   *
   *  1. 0层和1层使用斜侧膨胀方式，对于多个逆深度情况，使用高斯归一化积
   *  2. 2层及以上使用上下左右膨胀方式，对于多个逆深度情况，使用高斯归一化积
   *
   * @param level             输入金字塔的层数
   * @param input_idepth_sum  输入输出的金字塔level层上的逆深度高斯归一化积
   * @param input_hessian_sum 输入输出的金字塔level层上的hessian和
   */
  void IdpethExpansion(const int &level, cv::Mat &input_idepth_sum, cv::Mat &input_hessian_sum);

  /**
   * Tracker中使用的像素到像素的变换
   *
   * @param Tji     输入的参考帧到当前帧的变换
   * @param pi      输入的参考帧上的像素点pi
   * @param level   输入的金字塔层级
   * @param p_temp  输出的临时temp点
   * @param pj      输出的当前帧上的像素点pj
   */
  void Pixel2pixel(const SE3f &Tji, const TrackIdpethPoint::SharedPtr &pi, const int &level, Vec3f &p_temp, Vec2f &pj);

  /**
   * 设置当前帧，用于测试
   *
   * @param frame 输入的当前待跟踪的帧
   */
  void SetCurrFrame(Frame::SharedPtr frame) { curr_frame_ = std::move(frame); }

  /**
   * 计算jacobian和error误差，并构建正规方程，使用huber核函数
   *
   * 1. 获取level层的参考帧的逆深度点 lvl_pts
   * 2. 将lvl_pts上的点pi投影到当前帧pj点上（投影过程+差值过程）
   * 3. 使用Tji、ai、bi、aj、bj和上述得到的投影点和图像差值构建残差
   * 4. 根据残差和options_中的huber_threshold_计算hw值
   * 5. 计算雅可比矩阵
   * 6. 根据hw、雅可比矩阵和残差构建正规方程
   *
   * @param level       输入的金字塔层级
   * @param Tji         输入的参考帧到当前帧的变换
   * @param ai          输入的参考帧的绝对仿射参数ai
   * @param bi          输入的参考帧的绝对仿射参数bi
   * @param aj          输入的当前帧的绝对仿射参数aj
   * @param bj          输入的当前帧的绝对仿射参数bj
   * @param H           输出的正规方程中的H矩阵
   * @param b           输出的正规方程中的b向量
   * @param adjust_flag 输出的是否调整了外点阈值
   * @return Vec2f  [energy_photo, ngood]
   */
  Vec2f ComputeJacobianAndError(const int &level, const SE3f &Tji, const float &ai, const float &bi, const float &aj, const float &bj, Mat8f &H,
                                Vec8f &b, bool &adjust_flag);

  /**
   * 根据Tji,尝试一次优化
   *
   * 1. 从粗到精遍历金字塔层级
   * 2. 使用lm方法，进行优化
   *
   * 优化失败触发条件，可以做到提前退出
   *  1. ComputeJacobianAndError函数中，内点数量始终不能大于0.6，则认定失败
   *  2. 层级优化完成后，如果发现优化的RMSE结果，大于1.5倍的中断abort_res，则认定失败
   *
   * @param Tji_estimate  输入输出的Tji，参考帧和普通帧之间的位姿变换
   * @param aj_estimate   输入输出的aj，当前帧的绝对仿射参数aj
   * @param bj_estimate   输入输出的bj，当前帧的绝对仿射参数bj
   * @param ai            输入的ai，参考帧绝对仿射参数
   * @param bi            输入的bi，参考帧绝对仿射参数bj
   * @param abort_res     输入的各层级中优化的RMSE中断阈值
   * @return true         优化成功
   * @return false        优化失败
   */
  bool TryOnce(SE3f &Tji_estimate, float &aj_estimate, float &bj_estimate, const float &ai, const float &bi, const std::vector<float> &abort_res);

  /**
   * 从后端线程中同步数据，然后才能开始跟踪
   *
   * 1. 同步参考帧状态
   *  1.1 同步参考帧指针到ref_keyframe_self_中，保证能够获取最新的位姿状态
   *  1.2 同步参考帧状态到ref_frame_state_中，与后端产生数据分离，保证线程安全
   *  1.3 同步上一帧参考关键帧的位姿，与后端产生数据分离
   * 2. 同步逆深度点指针，与后端线程保证数据隔离
   * 3. 同步相机内参，与后端线程保证数据隔离
   */
  void SyncFromLocalMapper();

  /**
   * 根据Tracker跟踪器的状态，跟踪当前帧
   *
   * 1. 有 last frame 的跟踪尝试
   * 2. 没有 last frame 的跟踪尝试
   *
   * @param frame   输入的待跟踪的帧
   * @return true   跟踪成功
   * @return false  跟踪失败
   */
  bool TrackActivateFrame(Frame::SharedPtr frame);


  /**
   * 根据Tracker跟踪器的状态，跟踪当前帧
   *
   * 1. 做线程同步，保证线程安全，取后端线程更新数据，做数据隔离和数据关联
   * 2. 尝试不同的优化假设初值，进行TryOnce，实现提前退出
   * 3. 对 velocity 等参数进行初始化
   *
   * @return true   跟踪成功
   * @return false  跟踪失败
   */
  bool TrackFrameWithLast();

  /**
   * 对于没有上一帧条件下的跟踪frame
   *
   * 1. 做线程同步，保证线程安全，取后端线程更新数据，做数据隔离和数据关联
   * 2. 仅使用相对参考关键帧静止策略进行跟踪尝试
   * 3. 对 velocity 等参数进行初始化
   *
   * @return true   跟踪成功
   * @return false  跟踪失败
   */
  bool TrackFrameWithoutLast();

  /**
   * 使用回溯算法，构建跟踪器的小转角尝试生成（不包含静止条件）
   *
   * @param try_out 输入的尝试向量
   * @param idx     输入的更改索引idx
   *
   */
  void GenerateTries(Vec3f &try_out, const int &idx);

  /**
   * 判断当前跟踪帧是否是关键帧
   *
   * 1. 时间间隔考虑，当间隔超过规定的间隔时间阈值时，插入关键帧
   * 2. 空间间隔考虑，使用[Rji,tji]、[Rji,-tji]、[I,tji]、[I,-tji]计算平均光流，与阈值进行判断
   * 3. 从光度仿射参数考虑，当相对仿射参数exp_aji变化较大时，说明光照条件发生改变，要创建关键帧了
   * 4. 从优化角度考虑，如果当前帧跟踪跟踪的能量值超过当前参考帧第一次跟踪的两倍，则创建关键帧
   *
   * @return true   创建关键帧
   * @return false  不创建关键帧
   */
  bool IsNeedKeyFrame();


private:
  std::shared_ptr<spdlog::logger> logger_; ///< 基于spdlog的Tracker的日志记录器

  Options::SharedPtr options_;  ///< 跟踪器的配置参数
  Frame::SharedPtr curr_frame_; ///< 当前待跟踪帧

  KeyFrame::SharedPtr ref_keyframe_;        ///< 跟踪器参考关键帧，后端可更新
  TrackerIdepthPoints track_idepth_points_; ///< 参考关键帧的逆深度点，后端可更新
  std::vector<float> track_fx_;             ///< 图像金字塔的fx，后端可更新
  std::vector<float> track_fy_;             ///< 图像金字塔的fy，后端可更新
  std::vector<float> track_cx_;             ///< 图像金字塔的cx，后端可更新
  std::vector<float> track_cy_;             ///< 图像金字塔的cy，后端可更新

  KeyFrame::SharedPtr ref_keyframe_self_;        ///< 跟踪器参考关键帧，与后端有关，可能更新
  RefFrameState ref_frame_state_;                ///< 跟踪器参考关键帧状态，与后端无关
  TrackerIdepthPoints track_idepth_points_self_; ///< 参考关键帧的逆深度点，与后端无关
  std::vector<float> track_fx_self_;             ///< 图像金字塔的fx，与后端无关
  std::vector<float> track_fy_self_;             ///< 图像金字塔的fy，与后端无关
  std::vector<float> track_cx_self_;             ///< 图像金字塔的cx，与后端无关
  std::vector<float> track_cy_self_;             ///< 图像金字塔的cy，与后端无关

  std::vector<Vec6f, Eigen::aligned_allocator<Vec6f>> velocity_tries_; ///< 速度尝试

  std::vector<float> last_rmse_;      ///< tracker最新的优化均方根误差
  std::vector<float> abort_rmse_;     ///< tracker的优化均方根误差中断阈值
  std::vector<float> first_ref_rmse_; ///< 第一次参考关键帧跟踪的RMSE

  std::mutex tracker_mutex_;        ///< 跟踪器互斥量
  std::atomic_bool mapper_changed_; ///< 后端是否更新
  bool refframe_changed_;           ///< 参考关键帧是否改变

  SE3f velocity_;                         ///< Tracker维护的速度信息Tcl
  SE3f Tlr_;                              ///< 上一帧相对参考帧位姿
  KeyFrame::SharedPtr last_ref_keyframe_; ///< 上一帧的参考关键帧，与后端有关，可能更新
  RefFrameState last_ref_frame_state_;    ///< 上一帧的参考关键帧状态，与后端无关
};

} // namespace dso_ssl
