#ifndef QK_FDM_CRANK_NICOLSON_H
#define QK_FDM_CRANK_NICOLSON_H

#include <cstdint>

namespace qk::fdm {

double crank_nicolson_price(double spot, double strike, double t, double vol,
                            double r, double q, int32_t option_type,
                            int32_t time_steps, int32_t spot_steps,
                            bool american_style = false);

} // namespace qk::fdm

#endif /* QK_FDM_CRANK_NICOLSON_H */
