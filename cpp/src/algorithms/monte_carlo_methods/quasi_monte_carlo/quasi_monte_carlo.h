#ifndef QK_MCM_QUASI_MONTE_CARLO_H
#define QK_MCM_QUASI_MONTE_CARLO_H

#include <cstdint>

namespace qk::mcm {

double quasi_monte_carlo_sobol_price(double spot, double strike, double t, double vol,
                                     double r, double q, int32_t option_type,
                                     int32_t paths);

double quasi_monte_carlo_halton_price(double spot, double strike, double t, double vol,
                                      double r, double q, int32_t option_type,
                                      int32_t paths);

} // namespace qk::mcm

#endif /* QK_MCM_QUASI_MONTE_CARLO_H */
