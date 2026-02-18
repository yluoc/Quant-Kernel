#ifndef QK_AGM_LIKELIHOOD_RATIO_H
#define QK_AGM_LIKELIHOOD_RATIO_H

#include "algorithms/adjoint_greeks/common/params.h"

namespace qk::agm {

double likelihood_ratio_delta(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const LikelihoodRatioParams& params = {20000, 42, 6.0}
);

} // namespace qk::agm

#endif /* QK_AGM_LIKELIHOOD_RATIO_H */
