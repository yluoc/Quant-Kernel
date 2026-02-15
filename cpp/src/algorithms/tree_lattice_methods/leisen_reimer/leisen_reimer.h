#ifndef QK_TLM_LEISEN_REIMER_H
#define QK_TLM_LEISEN_REIMER_H

#include <cstdint>

namespace qk::tlm {

double leisen_reimer_price(double spot, double strike, double t, double vol, double r, double q,
                           int32_t option_type, int32_t steps, bool american_style = false);

} // namespace qk::tlm

#endif /* QK_TLM_LEISEN_REIMER_H */
