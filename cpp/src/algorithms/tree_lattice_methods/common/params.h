#ifndef QK_TLM_PARAMS_H
#define QK_TLM_PARAMS_H

#include <cstdint>

namespace qk::tlm {

struct ImpliedTreeConfig {
    int32_t steps;
    bool american_style;
};

} // namespace qk::tlm

#endif /* QK_TLM_PARAMS_H */
