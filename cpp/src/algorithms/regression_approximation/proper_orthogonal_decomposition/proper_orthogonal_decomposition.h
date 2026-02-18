#ifndef QK_RAM_PROPER_ORTHOGONAL_DECOMPOSITION_H
#define QK_RAM_PROPER_ORTHOGONAL_DECOMPOSITION_H

#include "algorithms/regression_approximation/common/params.h"

namespace qk::ram {

double proper_orthogonal_decomposition_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const ProperOrthogonalDecompositionParams& params = {8, 64}
);

} // namespace qk::ram

#endif /* QK_RAM_PROPER_ORTHOGONAL_DECOMPOSITION_H */
