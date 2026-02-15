#ifndef QK_TLM_JARROW_RUDD_H
#define QK_TLM_JARROW_RUDD_H

#include <cstdint>

namespace qk::tlm {

double jarrow_rudd_price(double spot, double strike, double t, double vol, double r, double q,
                         int32_t option_type, int32_t steps, bool american_style = false);

} // namespace qk::tlm

#endif /* QK_TLM_JARROW_RUDD_H */
