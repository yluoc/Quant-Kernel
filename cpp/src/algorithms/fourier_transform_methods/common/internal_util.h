#ifndef QK_FTM_INTERNAL_UTIL_H
#define QK_FTM_INTERNAL_UTIL_H

#include "common/option_util.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace qk::ftm::detail {

constexpr double kPi = 3.1415926535897932384626433832795;
constexpr double kEps = 1e-12;
const std::complex<double> kI(0.0, 1.0);

inline double nan_value() { return qk::nan_value(); }

inline bool valid_option_type(int32_t option_type) { return qk::valid_option_type(option_type); }

inline double intrinsic_value(double x, double y, int32_t option_type) {
    return qk::intrinsic_value(x, y, option_type);
}

inline double clamp01(double x) {
    return std::min(1.0, std::max(0.0, x));
}

inline bool is_power_of_two(int32_t n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

inline double deterministic_price(double spot, double strike, double t,
                                  double r, double q, int32_t option_type) {
    double forward = spot * std::exp((r - q) * t);
    return std::exp(-r * t) * intrinsic_value(forward, strike, option_type);
}

inline double call_put_from_call_parity(double call_price, double spot, double strike,
                                        double t, double r, double q, int32_t option_type) {
    if (option_type == QK_CALL) return call_price;
    if (option_type == QK_PUT) {
        return call_price - spot * std::exp(-q * t) + strike * std::exp(-r * t);
    }
    return nan_value();
}

inline std::complex<double> bs_log_cf(std::complex<double> u, double spot,
                                      double t, double vol, double r, double q) {
    double mu = std::log(spot) + (r - q - 0.5 * vol * vol) * t;
    return std::exp(kI * u * mu - 0.5 * vol * vol * t * u * u);
}

inline std::complex<double> bs_log_return_cf(std::complex<double> u,
                                             double t, double vol, double r, double q) {
    double mu = (r - q - 0.5 * vol * vol) * t;
    return std::exp(kI * u * mu - 0.5 * vol * vol * t * u * u);
}

inline double linear_interpolate(const std::vector<double>& x,
                                 const std::vector<double>& y,
                                 double x0) {
    if (x.empty() || y.empty() || x.size() != y.size()) return nan_value();
    if (x0 <= x.front()) return y.front();
    if (x0 >= x.back()) return y.back();

    auto it = std::lower_bound(x.begin(), x.end(), x0);
    std::size_t idx = static_cast<std::size_t>(std::distance(x.begin(), it));
    if (idx == 0) return y[0];

    double x1 = x[idx - 1];
    double x2 = x[idx];
    double y1 = y[idx - 1];
    double y2 = y[idx];

    double w = (x0 - x1) / (x2 - x1);
    return (1.0 - w) * y1 + w * y2;
}

template <typename F>
inline double integrate_trapezoid(const F& f, double a, double b, int32_t steps) {
    if (!(b > a)) return 0.0;
    if (steps < 2) steps = 2;

    double h = (b - a) / static_cast<double>(steps - 1);
    double total = 0.5 * (f(a) + f(b));
    for (int32_t i = 1; i < steps - 1; ++i) {
        total += f(a + static_cast<double>(i) * h);
    }
    return h * total;
}

inline std::size_t next_power_of_two(std::size_t n) {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

inline void fft_inplace(std::vector<std::complex<double>>& a) {
    std::size_t n = a.size();
    std::size_t j = 0;
    for (std::size_t i = 1; i < n; ++i) {
        std::size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    for (std::size_t len = 2; len <= n; len <<= 1) {
        double angle = -2.0 * kPi / static_cast<double>(len);
        std::complex<double> wlen(std::cos(angle), std::sin(angle));
        for (std::size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            std::size_t half = len >> 1;
            for (std::size_t j2 = 0; j2 < half; ++j2) {
                std::complex<double> u = a[i + j2];
                std::complex<double> v = a[i + j2 + half] * w;
                a[i + j2] = u + v;
                a[i + j2 + half] = u - v;
                w *= wlen;
            }
        }
    }
}

inline void ifft_inplace(std::vector<std::complex<double>>& a) {
    std::size_t n = a.size();
    for (auto& v : a) v = std::conj(v);
    fft_inplace(a);
    double inv = 1.0 / static_cast<double>(n);
    for (auto& v : a) v = std::conj(v) * inv;
}

/// Bluestein chirp-z transform: y[m] = sum_j x[j] * exp(-2 pi i theta j m)
inline void bluestein_fractional_dft(const std::vector<std::complex<double>>& x,
                                     std::vector<std::complex<double>>& y,
                                     double theta) {
    std::size_t n = x.size();
    std::size_t m = next_power_of_two(2 * n - 1);

    // Chirp sequence: chirp[k] = exp(-pi i theta k^2)
    std::vector<std::complex<double>> chirp(n);
    for (std::size_t k = 0; k < n; ++k) {
        double phase = -kPi * theta * static_cast<double>(k) * static_cast<double>(k);
        chirp[k] = std::complex<double>(std::cos(phase), std::sin(phase));
    }

    // a[j] = x[j] * chirp[j], zero-padded to length m
    std::vector<std::complex<double>> a(m, std::complex<double>(0.0, 0.0));
    for (std::size_t j = 0; j < n; ++j) {
        a[j] = x[j] * chirp[j];
    }

    // b[k] = conj(chirp[|k|]), with circular wrapping for negative indices
    std::vector<std::complex<double>> b(m, std::complex<double>(0.0, 0.0));
    b[0] = std::conj(chirp[0]);
    for (std::size_t k = 1; k < n; ++k) {
        b[k] = std::conj(chirp[k]);
        b[m - k] = std::conj(chirp[k]);
    }

    // Convolve a and b via FFT
    fft_inplace(a);
    fft_inplace(b);
    for (std::size_t i = 0; i < m; ++i) {
        a[i] *= b[i];
    }
    ifft_inplace(a);

    // y[k] = chirp[k] * conv[k]
    y.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        y[k] = chirp[k] * a[k];
    }
}

} // namespace qk::ftm::detail

#endif /* QK_FTM_INTERNAL_UTIL_H */
