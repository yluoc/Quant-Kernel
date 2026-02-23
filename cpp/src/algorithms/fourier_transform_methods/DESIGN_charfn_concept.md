# Design: Pluggable Characteristic Function (CharFn) Concept

**Status:** Design only (no implementation yet)
**Scope:** Fourier pricing methods — Carr-Madan FFT, COS, Lewis, Fractional FFT, Hilbert Transform

---

## 1. Proposed CharFn Callable Signature

```cpp
// Primary interface: characteristic function of log-price
//   phi(u, t) -> complex<double>
// where u is a complex frequency argument and t is maturity.
//
// For BSM: phi(u, t) = exp(i*u*mu*t - 0.5*vol^2*t*u^2)
// For Heston: phi(u, t) = exp(C(u,t) + D(u,t)*v0 + i*u*log(S))
//
// The callable captures all model parameters (vol, kappa, theta, etc.)
// at construction time, so the hot-loop call is phi(u, t) only.

template<typename F>
inline constexpr bool is_charfn_v =
    std::is_invocable_r_v<std::complex<double>, F,
                          std::complex<double> /*u*/, double /*t*/>;
```

### Why `phi(u, t)` and not `phi(u, t, spot, ...)`?

The five existing Fourier methods use two variants of the characteristic function:

| Method | Current function | Spot in CF? | Complex arg pattern |
|--------|-----------------|-------------|---------------------|
| Carr-Madan FFT | `bs_log_cf(u, spot, t, vol, r, q)` | Yes | `(v, -(alpha+1))` |
| COS (Fang-Oosterlee) | `bs_log_cf(u, spot, t, vol, r, q)` | Yes | `(u, 0)` |
| Fractional FFT | `bs_log_cf(u, spot, t, vol, r, q)` | Yes | `(v, -(alpha+1))` |
| Lewis inversion | `bs_log_return_cf(u, t, vol, r, q)` | No | `(u, -0.5)` |
| Hilbert transform | `bs_log_return_cf(u, t, vol, r, q)` | No | `(u, 0)` and `(u, -1)` |

The distinction is whether `spot` enters the characteristic function itself or is
handled by the pricing formula externally. For a unified interface:

- **Capture `spot` at construction time** (in the lambda closure), so the
  callable signature is simply `phi(u, t)`.
- This works for both `bs_log_cf` (spot enters via `log(spot)` additive term)
  and `bs_log_return_cf` (spot handled externally by the pricing formula).
- For Lewis/Hilbert, construct the CharFn **without** the spot term and let
  the pricing formula handle `spot` via `log(S/K)` in the integrand.

Two factory variants:

```cpp
// Full log-price CF (for Carr-Madan, COS, Fractional FFT)
auto make_bsm_log_charfn(double spot, double vol, double r, double q);

// Log-return CF (for Lewis, Hilbert — no spot dependence)
auto make_bsm_logreturn_charfn(double vol, double r, double q);

// Heston log-price CF
auto make_heston_log_charfn(double spot, double r, double q,
                            double v0, double kappa, double theta,
                            double sigma, double rho);

// Heston log-return CF
auto make_heston_logreturn_charfn(double r, double q,
                                  double v0, double kappa, double theta,
                                  double sigma, double rho);
```

Each returns a lightweight lambda satisfying `is_charfn_v`.

---

## 2. Refactor Plan

### 2.1 Carr-Madan FFT

**Current call site** (`carr_madan_fft.cpp:45`):
```cpp
detail::bs_log_cf(arg, spot, t, vol, r, q) / den;
```

**Refactored (template):**
```cpp
template<typename CharFn>
double carr_madan_fft_price_impl(CharFn&& phi, double spot, double strike,
                                  double t, double r,
                                  int32_t option_type,
                                  const CarrMadanFFTParams& params)
{
    // ... setup ...
    for (int32_t j = 0; j < n; ++j) {
        // ...
        std::complex<double> psi = std::exp(-r * t) * phi(arg, t) / den;
        // ... (rest unchanged)
    }
}
```

**C ABI wrapper (unchanged signature):**
```cpp
double carr_madan_fft_price(double spot, double strike, double t, double vol,
                            double r, double q, int32_t option_type,
                            const CarrMadanFFTParams& params) {
    auto phi = make_bsm_log_charfn(spot, vol, r, q);
    return carr_madan_fft_price_impl(phi, spot, strike, t, r, option_type, params);
}

// New Heston variant:
double carr_madan_fft_heston_price(double spot, double strike, double t,
                                   double r, double q,
                                   double v0, double kappa, double theta,
                                   double sigma, double rho,
                                   int32_t option_type,
                                   const CarrMadanFFTParams& params) {
    auto phi = make_heston_log_charfn(spot, r, q, v0, kappa, theta, sigma, rho);
    return carr_madan_fft_price_impl(phi, spot, strike, t, r, option_type, params);
}
```

### 2.2 COS Method (Fang-Oosterlee)

**Current call site** (`cos_method.cpp:59`):
```cpp
detail::bs_log_cf(std::complex<double>(u, 0.0), spot, t, vol, r, q);
```

**Refactored:**
```cpp
template<typename CharFn>
double cos_method_price_impl(CharFn&& phi, double spot, double strike,
                              double t, double r,
                              int32_t option_type,
                              const COSMethodParams& params)
{
    // ... setup ...
    std::complex<double> phi_val = phi(std::complex<double>(u, 0.0), t);
    // ...
}
```

### 2.3 Lewis Fourier Inversion

**Current call site** (`lewis_fourier_inversion.cpp:27`):
```cpp
detail::bs_log_return_cf(arg, t, vol, r, q);
```

**Refactored:**
```cpp
template<typename CharFn>
double lewis_fourier_inversion_price_impl(CharFn&& phi, double spot, double strike,
                                           double t, double r,
                                           int32_t option_type,
                                           const LewisFourierInversionParams& params)
{
    // phi here is the log-return CF (no spot baked in)
    auto integrand = [&](double u) -> double {
        std::complex<double> arg(u, -0.5);
        std::complex<double> cf = phi(arg, t);
        // ...
    };
}
```

### 2.4 Fractional FFT and Hilbert Transform

Follow the same pattern as Carr-Madan (log-price CF) and Lewis (log-return CF)
respectively.

---

## 3. Minimal Change Strategy

### Principle: Template internals, preserve C ABI

1. **Extract the inner loop** of each Fourier method into a `_impl` template
   function parameterized on `CharFn`.

2. **Keep the existing public functions** as thin wrappers that construct
   `make_bsm_log_charfn(...)` and forward to `_impl`.

3. **Add new Heston Fourier functions** as additional thin wrappers that
   construct `make_heston_log_charfn(...)`.

4. **No changes to existing C ABI signatures** — BSM Fourier functions retain
   their `(spot, strike, t, vol, r, q, ...)` signatures.

5. **New C ABI functions** for Heston Fourier:
   ```
   qk_ftm_carr_madan_fft_heston_price(...)
   qk_ftm_cos_fang_oosterlee_heston_price(...)
   qk_ftm_lewis_fourier_inversion_heston_price(...)
   ```

### Performance Guarantee

Since `CharFn` is a template parameter (not `std::function`), the compiler
can inline the characteristic function evaluation into the integration loop.
Under LTO, the generated code is identical to the current hardcoded version.
No vtable, no heap, no indirection.

---

## 4. Migration Plan: Hardcoded BSM CF to Pluggable CF

### Step 1: Add CharFn trait and BSM/Heston factories

**File:** `cpp/src/common/charfn_concepts.h` (new)

```
- is_charfn_v<F> compile-time trait
- make_bsm_log_charfn(spot, vol, r, q)
- make_bsm_logreturn_charfn(vol, r, q)
- make_heston_log_charfn(spot, r, q, v0, kappa, theta, sigma, rho)
- make_heston_logreturn_charfn(r, q, v0, kappa, theta, sigma, rho)
```

### Step 2: Create `_impl` template functions (one per method)

For each of the five Fourier methods:
- Extract the pricing logic into `method_price_impl<CharFn>(phi, ...)`.
- The existing public function becomes a one-liner calling `_impl` with BSM CF.

**Files modified:**
```
carr_madan_fft.cpp        -> extract _impl
cos_method.cpp            -> extract _impl
lewis_fourier_inversion.cpp -> extract _impl
fractional_fft.cpp        -> extract _impl
hilbert_transform.cpp     -> extract _impl
```

No functional change — existing tests pass unchanged.

### Step 3: Add Heston Fourier pricing functions

New thin wrappers that call `_impl` with `make_heston_log_charfn(...)`:
```
cpp/src/algorithms/fourier_transform_methods/carr_madan_fft/carr_madan_fft_heston.cpp
cpp/src/algorithms/fourier_transform_methods/cos_method/cos_method_heston.cpp
cpp/src/algorithms/fourier_transform_methods/lewis_fourier_inversion/lewis_heston.cpp
```

### Step 4: Wire up C ABI, Python bindings, tests

Follow the standard pattern:
- Add to `qk_api.h` / `qk_api.cpp`
- Add to `_loader.py`, `engine.py`, `__init__.pyi`
- Add to `accelerator.py` native batch map
- Add comparison tests: Heston Fourier vs Heston CF (should match closely)

### Step 5: (Future) Additional models

With the `CharFn` concept in place, adding new models to Fourier pricing
requires only:
1. Write `make_MODEL_log_charfn(...)` factory returning a lambda.
2. Call existing `_impl` template functions.

Models that naturally fit:
- Variance Gamma: `make_vg_log_charfn(sigma, theta, nu, r, q)`
- Merton Jump-Diffusion: `make_mjd_log_charfn(vol, lambda, mu_j, sigma_j, r, q)`
- CGMY: `make_cgmy_log_charfn(C, G, M, Y, r, q)`

---

## 5. Heston Characteristic Function Reference

For the Heston model, the log-price characteristic function is:

```
phi(u, t) = exp(i*u*log(S) + i*u*(r-q)*t + C(u,t) + D(u,t)*v0)
```

where:
```
d     = sqrt((rho*sigma*i*u - kappa)^2 + sigma^2*(i*u + u^2))
g     = (kappa - rho*sigma*i*u - d) / (kappa - rho*sigma*i*u + d)
C(u,t) = (kappa*theta/sigma^2) * ((kappa - rho*sigma*i*u - d)*t
          - 2*log((1 - g*exp(-d*t)) / (1 - g)))
D(u,t) = ((kappa - rho*sigma*i*u - d) / sigma^2)
          * (1 - exp(-d*t)) / (1 - g*exp(-d*t))
```

The log-return variant omits the `i*u*log(S)` term.

---

## 6. Confirmation Checklist

- [x] Proposed callable signature: `std::complex<double> phi(complex<double> u, double t)`
- [x] Compile-time trait: `is_charfn_v<F>`
- [x] Refactor plan for all 5 Fourier methods
- [x] Minimal change strategy: template internals, preserve C ABI
- [x] Migration plan: 5 steps, incremental, backwards-compatible
- [x] Performance guarantee: template parameter = zero overhead
- [x] No runtime polymorphism (no std::function, no vtable)
- [x] Heston CF reference formula provided
- [x] Future model extensibility path documented
