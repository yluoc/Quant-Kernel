#ifndef QK_CFA_BLACK_SCHOLES_MERTON_H
#define QK_CFA_BLACK_SCHOLES_MERTON_H

#include <cstdint>

namespace qk::cfa {

double black_scholes_merton_price(double spot, double strike, double t, double vol,
                                  double r, double q, int32_t option_type);

} // namespace qk::cfa

#endif /* QK_CFA_BLACK_SCHOLES_MERTON_H */
