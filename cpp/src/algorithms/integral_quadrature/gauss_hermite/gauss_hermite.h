#ifndef QK_IQM_GAUSS_HERMITE_H
#define QK_IQM_GAUSS_HERMITE_H

#include "algorithms/integral_quadrature/common/params.h"

namespace qk::iqm {

double gauss_hermite_price(double spot, double strike, double t, double vol,
                           double r, double q, int32_t option_type,
                           const GaussHermiteParams& params = {128});

} // namespace qk::iqm

#endif /* QK_IQM_GAUSS_HERMITE_H */
