#ifndef QK_AGM_PATHWISE_DERIVATIVE_H
#define QK_AGM_PATHWISE_DERIVATIVE_H

#include "algorithms/adjoint_greeks/common/params.h"

namespace qk::agm {

double pathwise_derivative_delta(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const PathwiseDerivativeParams& params = {20000, 42}
);

} // namespace qk::agm

#endif /* QK_AGM_PATHWISE_DERIVATIVE_H */
