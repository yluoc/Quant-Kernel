#ifndef QK_CFA_BLACK_1976_H
#define QK_CFA_BLACK_1976_H

#include <cstdint>

namespace qk::cfa {

double black76_price(double forward, double strike, double t, double vol,
                     double r, int32_t option_type);

} // namespace qk::cfa

#endif /* QK_CFA_BLACK_1976_H */
