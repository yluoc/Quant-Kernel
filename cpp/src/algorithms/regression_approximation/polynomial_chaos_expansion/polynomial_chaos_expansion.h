#ifndef QK_RAM_POLYNOMIAL_CHAOS_EXPANSION_H
#define QK_RAM_POLYNOMIAL_CHAOS_EXPANSION_H

#include "algorithms/regression_approximation/common/params.h"

namespace qk::ram {

double polynomial_chaos_expansion_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const PolynomialChaosExpansionParams& params = {4, 32}
);

} // namespace qk::ram

#endif /* QK_RAM_POLYNOMIAL_CHAOS_EXPANSION_H */
