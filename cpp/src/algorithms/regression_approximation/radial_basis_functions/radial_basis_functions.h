#ifndef QK_RAM_RADIAL_BASIS_FUNCTIONS_H
#define QK_RAM_RADIAL_BASIS_FUNCTIONS_H

#include "algorithms/regression_approximation/common/params.h"

namespace qk::ram {

double radial_basis_function_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const RadialBasisFunctionParams& params = {24, 1.0, 1e-4}
);

} // namespace qk::ram

#endif /* QK_RAM_RADIAL_BASIS_FUNCTIONS_H */
