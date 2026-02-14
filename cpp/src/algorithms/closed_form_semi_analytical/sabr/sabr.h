#ifndef QK_CFA_SABR_H
#define QK_CFA_SABR_H

#include "algorithms/closed_form_semi_analytical/common/params.h"

namespace qk::cfa {

double sabr_hagan_lognormal_iv(double forward, double strike, double t,
                               const SABRParams& params);

double sabr_hagan_black76_price(double forward, double strike, double t, double r,
                                const SABRParams& params, int32_t option_type);

} // namespace qk::cfa

#endif /* QK_CFA_SABR_H */
