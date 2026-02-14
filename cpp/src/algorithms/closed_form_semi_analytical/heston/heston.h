#ifndef QK_CFA_HESTON_H
#define QK_CFA_HESTON_H

#include "algorithms/closed_form_semi_analytical/common/params.h"

namespace qk::cfa {

double heston_price_cf(double spot, double strike, double t, double r, double q,
                       const HestonParams& params, int32_t option_type,
                       int32_t integration_steps = 1024,
                       double integration_limit = 120.0);

} // namespace qk::cfa

#endif /* QK_CFA_HESTON_H */
