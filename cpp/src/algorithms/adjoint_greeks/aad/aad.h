#ifndef QK_AGM_AAD_H
#define QK_AGM_AAD_H

#include "algorithms/adjoint_greeks/common/params.h"

namespace qk::agm {

double aad_delta(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const AadParams& params = {64, 1e-6}
);

} // namespace qk::agm

#endif /* QK_AGM_AAD_H */
