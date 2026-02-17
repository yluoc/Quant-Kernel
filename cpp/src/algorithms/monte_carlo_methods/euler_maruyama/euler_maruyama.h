#ifndef QK_MCM_EULER_MARUYAMA_H
#define QK_MCM_EULER_MARUYAMA_H

#include <cstdint>

namespace qk::mcm {

double euler_maruyama_price(double spot, double strike, double t, double vol,
                            double r, double q, int32_t option_type,
                            int32_t paths, int32_t steps, uint64_t seed);

} // namespace qk::mcm

#endif /* QK_MCM_EULER_MARUYAMA_H */
