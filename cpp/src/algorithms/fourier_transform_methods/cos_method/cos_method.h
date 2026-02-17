#ifndef QK_FTM_COS_METHOD_H
#define QK_FTM_COS_METHOD_H

#include "algorithms/fourier_transform_methods/common/params.h"

namespace qk::ftm {

double cos_method_fang_oosterlee_price(double spot, double strike, double t, double vol,
                                       double r, double q, int32_t option_type,
                                       const COSMethodParams& params = {256, 10.0});

} // namespace qk::ftm

#endif /* QK_FTM_COS_METHOD_H */
