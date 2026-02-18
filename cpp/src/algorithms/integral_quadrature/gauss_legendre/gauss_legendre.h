#ifndef QK_IQM_GAUSS_LEGENDRE_H
#define QK_IQM_GAUSS_LEGENDRE_H

#include "algorithms/integral_quadrature/common/params.h"

namespace qk::iqm {

double gauss_legendre_price(double spot, double strike, double t, double vol,
                            double r, double q, int32_t option_type,
                            const GaussLegendreParams& params = {128, 200.0});

} // namespace qk::iqm

#endif /* QK_IQM_GAUSS_LEGENDRE_H */
