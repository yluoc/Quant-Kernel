#ifndef QK_MCM_IMPORTANCE_SAMPLING_H
#define QK_MCM_IMPORTANCE_SAMPLING_H

#include <cstdint>

namespace qk::mcm {

double importance_sampling_price(double spot, double strike, double t, double vol,
                                 double r, double q, int32_t option_type,
                                 int32_t paths, double shift, uint64_t seed);

} // namespace qk::mcm

#endif /* QK_MCM_IMPORTANCE_SAMPLING_H */
