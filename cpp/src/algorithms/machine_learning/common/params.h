#ifndef QK_MLM_PARAMS_H
#define QK_MLM_PARAMS_H

#include <cstdint>

namespace qk::mlm {

struct DeepBsdeParams {
    int32_t time_steps;
    int32_t hidden_width;
    int32_t training_epochs;
    double learning_rate;
};

struct PinnsParams {
    int32_t collocation_points;
    int32_t boundary_points;
    int32_t epochs;
    double loss_balance;
};

struct DeepHedgingParams {
    int32_t rehedge_steps;
    double risk_aversion;
    int32_t scenarios;
    uint64_t seed;
};

struct NeuralSdeCalibrationParams {
    double target_implied_vol;
    int32_t calibration_steps;
    double regularization;
};

} // namespace qk::mlm

#endif /* QK_MLM_PARAMS_H */
