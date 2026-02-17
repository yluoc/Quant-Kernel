#ifndef QK_FTM_HILBERT_TRANSFORM_H
#define QK_FTM_HILBERT_TRANSFORM_H

#include "algorithms/fourier_transform_methods/common/params.h"

namespace qk::ftm {

double hilbert_transform_price(double spot, double strike, double t, double vol,
                               double r, double q, int32_t option_type,
                               const HilbertTransformParams& params = {4096, 300.0});

} // namespace qk::ftm

#endif /* QK_FTM_HILBERT_TRANSFORM_H */
