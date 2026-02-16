#ifndef QK_FDM_PARAMS_H
#define QK_FDM_PARAMS_H

#include <cstdint>

namespace qk::fdm {

struct ADIHestonParams {
    double v0;
    double kappa;
    double theta_v;
    double sigma;
    double rho;
    int32_t s_steps;
    int32_t v_steps;
    int32_t time_steps;
};

struct PsorConfig {
    double omega;
    double tol;
    int32_t max_iter;
};

} // namespace qk::fdm

#endif /* QK_FDM_PARAMS_H */
