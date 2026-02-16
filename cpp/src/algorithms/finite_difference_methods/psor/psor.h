#ifndef QK_FDM_PSOR_H
#define QK_FDM_PSOR_H

#include <cstdint>

namespace qk::fdm {

double psor_price(double spot, double strike, double t, double vol,
                  double r, double q, int32_t option_type,
                  int32_t time_steps, int32_t spot_steps,
                  double omega = 1.2, double tol = 1e-8,
                  int32_t max_iter = 10000);

} // namespace qk::fdm

#endif /* QK_FDM_PSOR_H */
