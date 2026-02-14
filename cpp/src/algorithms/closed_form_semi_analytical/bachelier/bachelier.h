#ifndef QK_CFA_BACHELIER_H
#define QK_CFA_BACHELIER_H

#include <cstdint>

namespace qk::cfa {

double bachelier_price(double forward, double strike, double t, double normal_vol,
                       double r, int32_t option_type);

} // namespace qk::cfa

#endif /* QK_CFA_BACHELIER_H */
