#ifndef QK_IQM_INTERNAL_UTIL_H
#define QK_IQM_INTERNAL_UTIL_H

#include "common/math_util.h"
#include <quantkernel/qk_abi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace qk::iqm::detail {

constexpr double kPi = 3.1415926535897932384626433832795;
constexpr double kEps = 1e-12;
inline const std::complex<double> kI(0.0, 1.0);

inline double nan_value() {
    double out = 0.0;
    write_nan(&out);
    return out;
}

inline bool valid_option_type(int32_t option_type) {
    return option_type == QK_CALL || option_type == QK_PUT;
}

inline double intrinsic_value(double x, double y, int32_t option_type) {
    if (option_type == QK_CALL) return std::max(0.0, x - y);
    if (option_type == QK_PUT) return std::max(0.0, y - x);
    return nan_value();
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

inline std::complex<double> bs_log_return_cf(std::complex<double> u,
                                             double t, double vol, double r, double q) {
    double mu = (r - q - 0.5 * vol * vol) * t;
    return std::exp(kI * u * mu - 0.5 * vol * vol * t * u * u);
}

inline double lewis_integrand(double u, double log_moneyness,
                              double t, double vol, double r, double q) {
    std::complex<double> arg(u, -0.5);
    std::complex<double> cf = bs_log_return_cf(arg, t, vol, r, q);
    std::complex<double> numerator = std::exp(kI * (u * log_moneyness)) * cf;
    return (numerator / (u * u + 0.25)).real();
}

inline bool jacobi_eigendecompose(std::vector<double>& a,
                                  std::vector<double>& eigvals,
                                  std::vector<double>& eigvecs,
                                  int32_t n,
                                  int32_t max_iter = 200,
                                  double tol = 1e-14) {
    eigvecs.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int32_t i = 0; i < n; ++i) {
        eigvecs[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)] = 1.0;
    }

    for (int32_t iter = 0; iter < max_iter * n * n; ++iter) {
        int32_t p = 0;
        int32_t q = 1;
        double max_off = 0.0;
        for (int32_t i = 0; i < n; ++i) {
            for (int32_t j = i + 1; j < n; ++j) {
                double v = std::fabs(a[static_cast<std::size_t>(i) * n + j]);
                if (v > max_off) {
                    max_off = v;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off < tol) break;

        double app = a[static_cast<std::size_t>(p) * n + p];
        double aqq = a[static_cast<std::size_t>(q) * n + q];
        double apq = a[static_cast<std::size_t>(p) * n + q];

        double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        double c = std::cos(phi);
        double s = std::sin(phi);

        for (int32_t i = 0; i < n; ++i) {
            if (i == p || i == q) continue;
            double aip = a[static_cast<std::size_t>(i) * n + p];
            double aiq = a[static_cast<std::size_t>(i) * n + q];
            double new_ip = c * aip - s * aiq;
            double new_iq = s * aip + c * aiq;
            a[static_cast<std::size_t>(i) * n + p] = new_ip;
            a[static_cast<std::size_t>(p) * n + i] = new_ip;
            a[static_cast<std::size_t>(i) * n + q] = new_iq;
            a[static_cast<std::size_t>(q) * n + i] = new_iq;
        }

        a[static_cast<std::size_t>(p) * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[static_cast<std::size_t>(q) * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[static_cast<std::size_t>(p) * n + q] = 0.0;
        a[static_cast<std::size_t>(q) * n + p] = 0.0;

        for (int32_t i = 0; i < n; ++i) {
            double vip = eigvecs[static_cast<std::size_t>(i) * n + p];
            double viq = eigvecs[static_cast<std::size_t>(i) * n + q];
            eigvecs[static_cast<std::size_t>(i) * n + p] = c * vip - s * viq;
            eigvecs[static_cast<std::size_t>(i) * n + q] = s * vip + c * viq;
        }
    }

    eigvals.resize(static_cast<std::size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        eigvals[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i) * n + i];
    }

    std::vector<int32_t> idx(static_cast<std::size_t>(n));
    for (int32_t i = 0; i < n; ++i) idx[static_cast<std::size_t>(i)] = i;
    std::sort(idx.begin(), idx.end(), [&](int32_t lhs, int32_t rhs) {
        return eigvals[static_cast<std::size_t>(lhs)] < eigvals[static_cast<std::size_t>(rhs)];
    });

    std::vector<double> eigvals_sorted(static_cast<std::size_t>(n));
    std::vector<double> eigvecs_sorted(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (int32_t c = 0; c < n; ++c) {
        int32_t old_c = idx[static_cast<std::size_t>(c)];
        eigvals_sorted[static_cast<std::size_t>(c)] = eigvals[static_cast<std::size_t>(old_c)];
        for (int32_t r = 0; r < n; ++r) {
            eigvecs_sorted[static_cast<std::size_t>(r) * n + c] =
                eigvecs[static_cast<std::size_t>(r) * n + old_c];
        }
    }

    eigvals.swap(eigvals_sorted);
    eigvecs.swap(eigvecs_sorted);
    return true;
}

inline bool gauss_rule_from_jacobi_matrix(const std::vector<double>& diag,
                                          const std::vector<double>& offdiag,
                                          double mu0,
                                          std::vector<double>& nodes,
                                          std::vector<double>& weights) {
    int32_t n = static_cast<int32_t>(diag.size());
    if (n <= 0 || static_cast<int32_t>(offdiag.size()) != n - 1) return false;

    std::vector<double> jac(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int32_t i = 0; i < n; ++i) {
        jac[static_cast<std::size_t>(i) * n + i] = diag[static_cast<std::size_t>(i)];
        if (i + 1 < n) {
            double b = offdiag[static_cast<std::size_t>(i)];
            jac[static_cast<std::size_t>(i) * n + (i + 1)] = b;
            jac[static_cast<std::size_t>(i + 1) * n + i] = b;
        }
    }

    std::vector<double> eigvals;
    std::vector<double> eigvecs;
    if (!jacobi_eigendecompose(jac, eigvals, eigvecs, n)) return false;

    nodes = eigvals;
    weights.resize(static_cast<std::size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        double v0 = eigvecs[static_cast<std::size_t>(0) * n + i];
        weights[static_cast<std::size_t>(i)] = mu0 * v0 * v0;
    }
    return true;
}

inline bool gauss_hermite_rule(int32_t n, std::vector<double>& nodes, std::vector<double>& weights) {
    static std::mutex mtx;
    static std::map<int32_t, std::pair<std::vector<double>, std::vector<double>>> cache;
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = cache.find(n);
        if (it != cache.end()) {
            nodes = it->second.first;
            weights = it->second.second;
            return true;
        }
    }

    std::vector<double> diag(static_cast<std::size_t>(n), 0.0);
    std::vector<double> offdiag(static_cast<std::size_t>(n > 0 ? n - 1 : 0));
    for (int32_t i = 0; i < n - 1; ++i) {
        offdiag[static_cast<std::size_t>(i)] = std::sqrt((static_cast<double>(i) + 1.0) / 2.0);
    }

    bool ok = gauss_rule_from_jacobi_matrix(diag, offdiag, std::sqrt(kPi), nodes, weights);
    if (ok) {
        std::lock_guard<std::mutex> lock(mtx);
        cache[n] = {nodes, weights};
    }
    return ok;
}

inline bool gauss_laguerre_rule(int32_t n, std::vector<double>& nodes, std::vector<double>& weights) {
    static std::mutex mtx;
    static std::map<int32_t, std::pair<std::vector<double>, std::vector<double>>> cache;
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = cache.find(n);
        if (it != cache.end()) {
            nodes = it->second.first;
            weights = it->second.second;
            return true;
        }
    }

    std::vector<double> diag(static_cast<std::size_t>(n));
    std::vector<double> offdiag(static_cast<std::size_t>(n > 0 ? n - 1 : 0));
    for (int32_t i = 0; i < n; ++i) {
        diag[static_cast<std::size_t>(i)] = 2.0 * static_cast<double>(i) + 1.0;
        if (i + 1 < n) {
            offdiag[static_cast<std::size_t>(i)] = static_cast<double>(i) + 1.0;
        }
    }

    bool ok = gauss_rule_from_jacobi_matrix(diag, offdiag, 1.0, nodes, weights);
    if (ok) {
        std::lock_guard<std::mutex> lock(mtx);
        cache[n] = {nodes, weights};
    }
    return ok;
}

inline bool gauss_legendre_rule(int32_t n, std::vector<double>& nodes, std::vector<double>& weights) {
    static std::mutex mtx;
    static std::map<int32_t, std::pair<std::vector<double>, std::vector<double>>> cache;
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = cache.find(n);
        if (it != cache.end()) {
            nodes = it->second.first;
            weights = it->second.second;
            return true;
        }
    }

    std::vector<double> diag(static_cast<std::size_t>(n), 0.0);
    std::vector<double> offdiag(static_cast<std::size_t>(n > 0 ? n - 1 : 0));
    for (int32_t i = 0; i < n - 1; ++i) {
        double k = static_cast<double>(i) + 1.0;
        offdiag[static_cast<std::size_t>(i)] = k / std::sqrt(4.0 * k * k - 1.0);
    }

    bool ok = gauss_rule_from_jacobi_matrix(diag, offdiag, 2.0, nodes, weights);
    if (ok) {
        std::lock_guard<std::mutex> lock(mtx);
        cache[n] = {nodes, weights};
    }
    return ok;
}

template <typename F>
inline double adaptive_simpson_recursive(const F& f,
                                         double a,
                                         double b,
                                         double fa,
                                         double fm,
                                         double fb,
                                         double whole,
                                         double tol,
                                         int32_t depth) {
    double mid = 0.5 * (a + b);
    double left_mid = 0.5 * (a + mid);
    double right_mid = 0.5 * (mid + b);

    double flm = f(left_mid);
    double frm = f(right_mid);

    double left = (mid - a) * (fa + 4.0 * flm + fm) / 6.0;
    double right = (b - mid) * (fm + 4.0 * frm + fb) / 6.0;
    double delta = left + right - whole;

    if (depth <= 0 || std::fabs(delta) <= 15.0 * tol) {
        return left + right + delta / 15.0;
    }

    return adaptive_simpson_recursive(f, a, mid, fa, flm, fm, left, tol * 0.5, depth - 1)
         + adaptive_simpson_recursive(f, mid, b, fm, frm, fb, right, tol * 0.5, depth - 1);
}

template <typename F>
inline double integrate_adaptive_simpson(const F& f,
                                         double a,
                                         double b,
                                         double abs_tol,
                                         double rel_tol,
                                         int32_t max_depth) {
    if (!(b > a)) return 0.0;
    if (abs_tol <= 0.0) abs_tol = 1e-9;
    if (rel_tol <= 0.0) rel_tol = 1e-8;
    if (max_depth < 2) max_depth = 2;

    double fa = f(a);
    double fb = f(b);
    double mid = 0.5 * (a + b);
    double fm = f(mid);
    double whole = (b - a) * (fa + 4.0 * fm + fb) / 6.0;
    double tol = std::max(abs_tol, rel_tol * std::fabs(whole));

    return adaptive_simpson_recursive(f, a, b, fa, fm, fb, whole, tol, max_depth);
}

} // namespace qk::iqm::detail

#endif /* QK_IQM_INTERNAL_UTIL_H */
