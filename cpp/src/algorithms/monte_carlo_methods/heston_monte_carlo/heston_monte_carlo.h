#ifndef QK_MCM_HESTON_MONTE_CARLO_H
#define QK_MCM_HESTON_MONTE_CARLO_H

#include <cstdint>

namespace qk::mcm {

double heston_monte_carlo_price(double spot, double strike, double t,
                                double r, double q,
                                double v0, double kappa, double theta,
                                double sigma, double rho,
                                int32_t option_type,
                                int32_t paths, int32_t steps, uint64_t seed);

} // namespace qk::mcm

#endif /* QK_MCM_HESTON_MONTE_CARLO_H */
