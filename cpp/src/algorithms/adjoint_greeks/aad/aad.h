#ifndef QK_AGM_AAD_H
#define QK_AGM_AAD_H

#include "algorithms/adjoint_greeks/common/params.h"

namespace qk::agm {

// ---------------------------------------------------------------------------
// IMPLEMENTATION NOTE (January 2026)
//
// Despite its name, aad_delta() does NOT implement general-purpose Adjoint
// Algorithmic Differentiation (AAD) with a computational tape.  The actual
// implementation is:
//
//   1. Closed-form Black-Scholes-Merton call pricing.
//   2. Hand-written reverse-mode differentiation of that closed-form
//      expression (manual adjoint propagation through ~12 operations).
//   3. Tikhonov regularization toward the ATM delta prior, scaled by
//      1/sqrt(tape_steps).
//
// The AadParams.tape_steps field controls regularization decay, not an
// actual tape size.  AadParams.regularization is the Tikhonov lambda.
//
// This function is BSM-specific and cannot be extended to other models
// without a full rewrite.
//
// The name "aad_delta" is preserved for C ABI and Python API compatibility.
// New internal code should prefer bsm_adjoint_delta() (alias below).
// ---------------------------------------------------------------------------

// Legacy name — preserved for ABI/API backward compatibility.
double aad_delta(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const AadParams& params = {64, 1e-6}
);

// Preferred name — truthfully describes the implementation.
// Calls aad_delta(); identical behavior, zero overhead.
inline double bsm_adjoint_delta(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const AadParams& params = {64, 1e-6}
) {
    return aad_delta(spot, strike, t, vol, r, q, option_type, params);
}

} // namespace qk::agm

#endif /* QK_AGM_AAD_H */
