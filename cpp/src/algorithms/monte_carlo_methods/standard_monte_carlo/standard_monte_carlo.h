#ifndef QK_MCM_STANDARD_MONTE_CARLO_H
#define QK_MCM_STANDARD_MONTE_CARLO_H

#include <cstdint>

namespace qk::mcm {

double standard_monte_carlo_price(double spot, double strike, double t, double vol,
                                  double r, double q, int32_t option_type,
                                  int32_t paths, uint64_t seed);

} // namespace qk::mcm

#endif /* QK_MCM_STANDARD_MONTE_CARLO_H */
