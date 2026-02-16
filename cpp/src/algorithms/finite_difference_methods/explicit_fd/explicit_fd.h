#ifndef QK_FDM_EXPLICIT_FD_H
#define QK_FDM_EXPLICIT_FD_H

#include <cstdint>

namespace qk::fdm {

double explicit_fd_price(double spot, double strike, double t, double vol,
                         double r, double q, int32_t option_type,
                         int32_t time_steps, int32_t spot_steps,
                         bool american_style = false);

} // namespace qk::fdm

#endif /* QK_FDM_EXPLICIT_FD_H */
