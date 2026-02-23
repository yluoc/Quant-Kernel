#ifndef QK_MCM_LOCAL_VOL_MONTE_CARLO_H
#define QK_MCM_LOCAL_VOL_MONTE_CARLO_H

#include <cstdint>

namespace qk::mcm {

// Constant local volatility MC pricer — matches BSM Euler when vol is constant.
// General lambda-based local vol is available via the C++ template in
// model_concepts.h (make_local_vol_euler_step).
double local_vol_monte_carlo_price(double spot, double strike, double t,
                                   double vol, double r, double q,
                                   int32_t option_type,
                                   int32_t paths, int32_t steps, uint64_t seed);

} // namespace qk::mcm

#endif /* QK_MCM_LOCAL_VOL_MONTE_CARLO_H */
