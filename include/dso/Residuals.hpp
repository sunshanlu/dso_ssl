#pragma once

#include <Eigen/Core>

namespace dso_ssl
{
class Residual
{
public:
private:
};

/// 初始化部分需要构建的残差
class InitiallizerResidual
{
public:
    void ComputeError();

    void ComputeJacobian();

private:
};
} // namespace dso_ssl
