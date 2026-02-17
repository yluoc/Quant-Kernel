#ifndef QK_MCM_ANTITHETIC_VARIATES_H
#define QK_MCM_ANTITHETIC_VARIATES_H

#include <cstdint>

namespace qk::mcm {

double antithetic_variates_price(double spot, double strike, double t, double vol,
                                 double r, double q, int32_t option_type,
                                 int32_t paths, uint64_t seed);

} // namespace qk::mcm

#endif /* QK_MCM_ANTITHETIC_VARIATES_H */
