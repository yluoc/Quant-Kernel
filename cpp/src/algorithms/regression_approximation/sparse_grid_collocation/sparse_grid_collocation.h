#ifndef QK_RAM_SPARSE_GRID_COLLOCATION_H
#define QK_RAM_SPARSE_GRID_COLLOCATION_H

#include "algorithms/regression_approximation/common/params.h"

namespace qk::ram {

double sparse_grid_collocation_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const SparseGridCollocationParams& params = {3, 9}
);

} // namespace qk::ram

#endif /* QK_RAM_SPARSE_GRID_COLLOCATION_H */
