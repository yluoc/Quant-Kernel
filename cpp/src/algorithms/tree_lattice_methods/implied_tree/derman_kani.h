#ifndef QK_TLM_DERMAN_KANI_H
#define QK_TLM_DERMAN_KANI_H

#include "algorithms/tree_lattice_methods/common/params.h"

#include <cstdint>
#include <functional>

namespace qk::tlm {

double derman_kani_implied_tree_price(
    double spot, double strike, double t, double r, double q,
    int32_t option_type,
    const std::function<double(double, double)>& local_vol_surface,
    ImpliedTreeConfig config
);

} // namespace qk::tlm

#endif /* QK_TLM_DERMAN_KANI_H */
