#ifndef QK_BS_PRICER_H
#define QK_BS_PRICER_H

#include <quantkernel/qk_abi.h>

namespace qk {

/* Batch Black-Scholes pricer — assumes input/output pointers are non-null.
   Null-checking is done in the API layer. */
void bs_price_batch(const QKBSInput& input, QKBSOutput& output);

/* Single-row BS price only (used internally by IV solver).
   Returns price; does NOT write to any output struct.
   Caller must validate inputs beforehand. */
double bs_price_single(double spot, double strike, double t, double vol,
                       double r, double q, int32_t opt_type);

} // namespace qk

#endif /* QK_BS_PRICER_H */
