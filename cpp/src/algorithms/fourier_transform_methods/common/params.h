#ifndef QK_FTM_PARAMS_H
#define QK_FTM_PARAMS_H

#include <cstdint>

namespace qk::ftm {

struct CarrMadanFFTParams {
    int32_t grid_size;
    double eta;
    double alpha;
};

struct COSMethodParams {
    int32_t n_terms;
    double truncation_width;
};

struct FractionalFFTParams {
    int32_t grid_size;
    double eta;
    double lambda;
    double alpha;
};

struct LewisFourierInversionParams {
    int32_t integration_steps;
    double integration_limit;
};

struct HilbertTransformParams {
    int32_t integration_steps;
    double integration_limit;
};

} // namespace qk::ftm

#endif /* QK_FTM_PARAMS_H */
