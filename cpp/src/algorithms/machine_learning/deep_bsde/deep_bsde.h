#ifndef QK_MLM_DEEP_BSDE_H
#define QK_MLM_DEEP_BSDE_H

#include "algorithms/machine_learning/common/params.h"

namespace qk::mlm {

double deep_bsde_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const DeepBsdeParams& params = {50, 64, 400, 5e-3}
);

} // namespace qk::mlm

#endif /* QK_MLM_DEEP_BSDE_H */
