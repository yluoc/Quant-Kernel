#ifndef QK_MCM_PARAMS_H
#define QK_MCM_PARAMS_H

#include <cstdint>

namespace qk::mcm {

struct MonteCarloParams {
    int32_t paths;
    int32_t steps;
    uint64_t seed;
};

} // namespace qk::mcm

#endif /* QK_MCM_PARAMS_H */
