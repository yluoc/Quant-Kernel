#ifndef QK_FTM_CARR_MADAN_FFT_H
#define QK_FTM_CARR_MADAN_FFT_H

#include "algorithms/fourier_transform_methods/common/params.h"

namespace qk::ftm {

double carr_madan_fft_price(double spot, double strike, double t, double vol,
                            double r, double q, int32_t option_type,
                            const CarrMadanFFTParams& params = {4096, 0.25, 1.5});

} // namespace qk::ftm

#endif /* QK_FTM_CARR_MADAN_FFT_H */
