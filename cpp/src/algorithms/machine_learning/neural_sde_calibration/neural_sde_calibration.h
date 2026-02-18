#ifndef QK_MLM_NEURAL_SDE_CALIBRATION_H
#define QK_MLM_NEURAL_SDE_CALIBRATION_H

#include "algorithms/machine_learning/common/params.h"

namespace qk::mlm {

double neural_sde_calibration_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const NeuralSdeCalibrationParams& params = {0.2, 200, 1e-3}
);

} // namespace qk::mlm

#endif /* QK_MLM_NEURAL_SDE_CALIBRATION_H */
