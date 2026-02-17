#ifndef QK_FTM_FRACTIONAL_FFT_H
#define QK_FTM_FRACTIONAL_FFT_H

#include "algorithms/fourier_transform_methods/common/params.h"

namespace qk::ftm {

double fractional_fft_price(double spot, double strike, double t, double vol,
                            double r, double q, int32_t option_type,
                            const FractionalFFTParams& params = {256, 0.25, 0.05, 1.5});

} // namespace qk::ftm

#endif /* QK_FTM_FRACTIONAL_FFT_H */
