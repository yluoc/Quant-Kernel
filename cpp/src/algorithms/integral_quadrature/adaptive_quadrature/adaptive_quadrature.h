#ifndef QK_IQM_ADAPTIVE_QUADRATURE_H
#define QK_IQM_ADAPTIVE_QUADRATURE_H

#include "algorithms/integral_quadrature/common/params.h"

namespace qk::iqm {

double adaptive_quadrature_price(double spot, double strike, double t, double vol,
                                 double r, double q, int32_t option_type,
                                 const AdaptiveQuadratureParams& params = {1e-9, 1e-8, 14, 200.0});

} // namespace qk::iqm

#endif /* QK_IQM_ADAPTIVE_QUADRATURE_H */
