#ifndef QK_FTM_LEWIS_FOURIER_INVERSION_H
#define QK_FTM_LEWIS_FOURIER_INVERSION_H

#include "algorithms/fourier_transform_methods/common/params.h"

namespace qk::ftm {

double lewis_fourier_inversion_price(double spot, double strike, double t, double vol,
                                     double r, double q, int32_t option_type,
                                     const LewisFourierInversionParams& params = {4096, 300.0});

} // namespace qk::ftm

#endif /* QK_FTM_LEWIS_FOURIER_INVERSION_H */
