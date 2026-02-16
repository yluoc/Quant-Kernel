#ifndef QK_FDM_ADI_H
#define QK_FDM_ADI_H

#include "algorithms/finite_difference_methods/common/params.h"
#include <cstdint>

namespace qk::fdm {

double adi_douglas_price(double spot, double strike, double t, double r, double q,
                         const ADIHestonParams& params, int32_t option_type);

double adi_craig_sneyd_price(double spot, double strike, double t, double r, double q,
                             const ADIHestonParams& params, int32_t option_type);

double adi_hundsdorfer_verwer_price(double spot, double strike, double t, double r, double q,
                                   const ADIHestonParams& params, int32_t option_type);

} // namespace qk::fdm

#endif /* QK_FDM_ADI_H */
