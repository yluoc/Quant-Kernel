#ifndef QK_MLM_PINNS_H
#define QK_MLM_PINNS_H

#include "algorithms/machine_learning/common/params.h"

namespace qk::mlm {

double pinns_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const PinnsParams& params = {5000, 400, 300, 1.0}
);

} // namespace qk::mlm

#endif /* QK_MLM_PINNS_H */
