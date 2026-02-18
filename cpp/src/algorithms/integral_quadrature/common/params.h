#ifndef QK_IQM_PARAMS_H
#define QK_IQM_PARAMS_H

#include <cstdint>

namespace qk::iqm {

struct GaussHermiteParams {
    int32_t n_points;
};

struct GaussLaguerreParams {
    int32_t n_points;
};

struct GaussLegendreParams {
    int32_t n_points;
    double integration_limit;
};

struct AdaptiveQuadratureParams {
    double abs_tol;
    double rel_tol;
    int32_t max_depth;
    double integration_limit;
};

} // namespace qk::iqm

#endif /* QK_IQM_PARAMS_H */
