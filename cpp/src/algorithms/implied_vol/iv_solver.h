#ifndef QK_IV_SOLVER_H
#define QK_IV_SOLVER_H

#include <quantkernel/qk_abi.h>

namespace qk {

/* Batch implied-vol solver — assumes non-null pointers (checked by API layer) */
void iv_solve_batch(const QKIVInput& input, QKIVOutput& output);

} // namespace qk

#endif /* QK_IV_SOLVER_H */
