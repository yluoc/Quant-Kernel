#ifndef QK_MLM_DEEP_HEDGING_H
#define QK_MLM_DEEP_HEDGING_H

#include "algorithms/machine_learning/common/params.h"

namespace qk::mlm {

double deep_hedging_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const DeepHedgingParams& params = {26, 0.5, 20000, 42}
);

} // namespace qk::mlm

#endif /* QK_MLM_DEEP_HEDGING_H */
