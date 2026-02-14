#ifndef QK_CFA_PARAMS_H
#define QK_CFA_PARAMS_H

#include <cstdint>

namespace qk::cfa {

struct HestonParams {
    double v0;
    double kappa;
    double theta;
    double sigma;
    double rho;
};

struct MertonJumpDiffusionParams {
    double jump_intensity;
    double jump_mean;
    double jump_vol;
    int32_t max_terms;
};

struct VarianceGammaParams {
    double sigma;
    double theta;
    double nu;
};

struct SABRParams {
    double alpha;
    double beta;
    double rho;
    double nu;
};

} // namespace qk::cfa

#endif /* QK_CFA_PARAMS_H */
