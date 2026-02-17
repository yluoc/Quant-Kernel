#ifndef QK_MCM_STRATIFIED_SAMPLING_H
#define QK_MCM_STRATIFIED_SAMPLING_H

#include <cstdint>

namespace qk::mcm {

double stratified_sampling_price(double spot, double strike, double t, double vol,
                                 double r, double q, int32_t option_type,
                                 int32_t paths, uint64_t seed);

} // namespace qk::mcm

#endif /* QK_MCM_STRATIFIED_SAMPLING_H */
