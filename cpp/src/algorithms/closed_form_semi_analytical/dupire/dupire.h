#ifndef QK_CFA_DUPIRE_H
#define QK_CFA_DUPIRE_H

#include <functional>

namespace qk::cfa {

double dupire_local_vol(double strike, double t, double call_price,
                        double dC_dT, double dC_dK, double d2C_dK2,
                        double r, double q);

double dupire_local_vol_fd(const std::function<double(double, double)>& call_price_fn,
                           double strike, double t, double r, double q,
                           double dK = 0.25, double dT = 1e-3);

} // namespace qk::cfa

#endif /* QK_CFA_DUPIRE_H */
