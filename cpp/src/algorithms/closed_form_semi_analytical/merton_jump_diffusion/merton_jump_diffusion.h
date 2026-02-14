#ifndef QK_CFA_MERTON_JUMP_DIFFUSION_H
#define QK_CFA_MERTON_JUMP_DIFFUSION_H

#include "algorithms/closed_form_semi_analytical/common/params.h"

namespace qk::cfa {

double merton_jump_diffusion_price(double spot, double strike, double t, double vol,
                                   double r, double q,
                                   const MertonJumpDiffusionParams& params,
                                   int32_t option_type);

} // namespace qk::cfa

#endif /* QK_CFA_MERTON_JUMP_DIFFUSION_H */
