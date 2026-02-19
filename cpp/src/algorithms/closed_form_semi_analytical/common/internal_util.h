#ifndef QK_CFA_INTERNAL_UTIL_H
#define QK_CFA_INTERNAL_UTIL_H

#include "common/option_util.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>

namespace qk::cfa::detail {

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

namespace gl64 {
    static constexpr int N = 64;
    static constexpr double x[] = {
        0.02435029266342443, 0.07299312178779904, 0.12146281929612056, 0.16964442042399283,
        0.21742364374720903, 0.26468716220876742, 0.31132287199021097, 0.35722015833766813,
        0.40227015796399163, 0.44636601725346409, 0.48940314570705296, 0.53127946401989457,
        0.57189564620263400, 0.61115535517239328, 0.64896547125465731, 0.68523631305423327,
        0.71988185017161088, 0.75281990726053194, 0.78397235894334139, 0.81326531512279754,
        0.84062929625258032, 0.86599939815409282, 0.88931544599511414, 0.91052213707850280,
        0.92956917213193957, 0.94641137485840282, 0.96100879965205372, 0.97332682778991096,
        0.98333625388462596, 0.99101337147674668, 0.99634011677195528, 0.99930504173577214
    };
    static constexpr double w[] = {
        0.04869095700913972, 0.04857546744150343, 0.04834476223480296, 0.04799938859645831,
        0.04754016571483031, 0.04696818281621002, 0.04628479658131442, 0.04549162792741814,
        0.04459055816375656, 0.04358372452932345, 0.04247351512365359, 0.04126256324262353,
        0.03995374113272034, 0.03855015317861563, 0.03705512854024005, 0.03547221325688239,
        0.03380516183714161, 0.03205792835485155, 0.03023465707240248, 0.02833967261425948,
        0.02637746971505466, 0.02435270256871087, 0.02227017380838325, 0.02013482315353021,
        0.01795171577569734, 0.01572603047602472, 0.01346304789671864, 0.01116813946013113,
        0.00884675982636395, 0.00650445796897836, 0.00414703326056247, 0.00178328072169643
    };
} // namespace gl64

template <typename F>
inline double integrate_gauss_legendre(const F& f, double a, double b) {
    if (!(b > a)) return 0.0;
    double mid = 0.5 * (a + b);
    double half = 0.5 * (b - a);
    double sum = 0.0;
    for (int i = 0; i < 32; ++i) {
        double xi = gl64::x[i];
        double wi = gl64::w[i];
        sum += wi * (f(mid - half * xi) + f(mid + half * xi));
    }
    return half * sum;
}

template <typename F>
inline double integrate_gl_panels(const F& f, double a, double b, int32_t panels) {
    if (panels < 1) panels = 1;
    double panel_width = (b - a) / static_cast<double>(panels);
    double total = 0.0;
    for (int32_t p = 0; p < panels; ++p) {
        double pa = a + static_cast<double>(p) * panel_width;
        double pb = pa + panel_width;
        total += integrate_gauss_legendre(f, pa, pb);
    }
    return total;
}

template <typename CfFn>
inline void probability_p1p2(const CfFn& cf, double log_strike,
                             int32_t /* steps */, double integration_limit,
                             double& p1_out, double& p2_out) {
    std::complex<double> phi_minus_i = cf(std::complex<double>(0.0, -1.0));
    double phi_mi_abs = std::abs(phi_minus_i);

    int32_t panels = std::max(2, static_cast<int32_t>(integration_limit / 15.0));

    double a = 1e-8;
    double b = integration_limit;
    double sum_p1 = 0.0;
    double sum_p2 = 0.0;

    double panel_width = (b - a) / static_cast<double>(panels);
    for (int32_t p = 0; p < panels; ++p) {
        double pa = a + static_cast<double>(p) * panel_width;
        double pb = pa + panel_width;
        double mid = 0.5 * (pa + pb);
        double half = 0.5 * (pb - pa);

        for (int i = 0; i < 32; ++i) {
            double xi = gl64::x[i];
            double wi = gl64::w[i];

            for (int side = 0; side < 2; ++side) {
                double u = (side == 0) ? (mid - half * xi) : (mid + half * xi);
                double inv_u = 1.0 / u;
                std::complex<double> cf_val2 = cf(std::complex<double>(u, 0.0));
                double cos_term = std::cos(u * log_strike);
                double sin_term = std::sin(u * log_strike);

                double re_cf = cf_val2.real();
                double im_cf = cf_val2.imag();
                double val_p2 = (cos_term * im_cf - sin_term * re_cf) * inv_u;

                sum_p2 += half * wi * val_p2;

                if (phi_mi_abs >= 1e-14) {
                    std::complex<double> cf_val1 = cf(std::complex<double>(u, -1.0));
                    double re1 = cf_val1.real();
                    double im1 = cf_val1.imag();
                    double n1_re = cos_term * re1 + sin_term * im1;
                    double n1_im = cos_term * im1 - sin_term * re1;
                    double d_re = n1_im * inv_u;
                    double d_im = -n1_re * inv_u;
                    double phi_re = phi_minus_i.real();
                    double phi_im = phi_minus_i.imag();
                    double phi_abs2 = phi_re * phi_re + phi_im * phi_im;
                    double val_p1 = (d_re * phi_re + d_im * phi_im) / phi_abs2;

                    sum_p1 += half * wi * val_p1;
                }
            }
        }
    }

    p2_out = clamp01(0.5 + sum_p2 / kPi);
    p1_out = (phi_mi_abs >= 1e-14) ? clamp01(0.5 + sum_p1 / kPi) : nan_value();
}

template <typename CfFn>
inline double probability_p2(const CfFn& cf, double log_strike,
                             int32_t steps, double integration_limit) {
    double p1, p2;
    probability_p1p2(cf, log_strike, steps, integration_limit, p1, p2);
    return p2;
}

template <typename CfFn>
inline double probability_p1(const CfFn& cf, double log_strike,
                             int32_t steps, double integration_limit) {
    double p1, p2;
    probability_p1p2(cf, log_strike, steps, integration_limit, p1, p2);
    return p1;
}

inline double call_put_from_call_parity(double call_price, double spot, double strike,
                                        double t, double r, double q, int32_t option_type) {
    if (option_type == QK_CALL) return call_price;
    if (option_type == QK_PUT) {
        return call_price - spot * std::exp(-q * t) + strike * std::exp(-r * t);
    }
    return nan_value();
}

} // namespace qk::cfa::detail

#endif /* QK_CFA_INTERNAL_UTIL_H */
