#ifndef QK_CFA_VARIANCE_GAMMA_H
#define QK_CFA_VARIANCE_GAMMA_H

#include "algorithms/closed_form_semi_analytical/common/params.h"

namespace qk::cfa {

double variance_gamma_price_cf(double spot, double strike, double t, double r, double q,
                               const VarianceGammaParams& params, int32_t option_type,
                               int32_t integration_steps = 1024,
                               double integration_limit = 120.0);

} // namespace qk::cfa

#endif /* QK_CFA_VARIANCE_GAMMA_H */
