#include "algorithms/finite_difference_methods/adi/adi.h"

#include "algorithms/finite_difference_methods/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace qk::fdm {

namespace {

inline int32_t idx(int32_t i, int32_t j, int32_t Ns1) {
    return j * Ns1 + i;
}

struct ADIGrid {
    std::vector<double> x;
    std::vector<double> v;
    double dx, dv;
    int32_t Ns, Nv;
};

ADIGrid build_2d_grid(double spot, double t, const ADIHestonParams& p) {
    ADIGrid g;
    g.Ns = p.s_steps;
    g.Nv = p.v_steps;

    double x0 = std::log(spot);
    double x_spread = 4.0 * std::sqrt(p.v0) * std::sqrt(t);
    if (x_spread < 1.0) x_spread = 1.0;
    g.dx = (2.0 * x_spread) / static_cast<double>(g.Ns);
    g.x.resize(g.Ns + 1);
    for (int32_t i = 0; i <= g.Ns; ++i) {
        g.x[i] = (x0 - x_spread) + i * g.dx;
    }

    double v_max = std::max(p.v0 * 5.0, p.theta_v * 5.0);
    if (v_max < 0.5) v_max = 0.5;
    g.dv = v_max / static_cast<double>(g.Nv);
    g.v.resize(g.Nv + 1);
    for (int32_t j = 0; j <= g.Nv; ++j) {
        g.v[j] = j * g.dv;
    }

    return g;
}

void apply_boundaries(std::vector<double>& U, const ADIGrid& g,
                      double strike, double r, double q, double tau,
                      int32_t option_type) {
    int32_t Ns1 = g.Ns + 1;

    for (int32_t j = 0; j <= g.Nv; ++j) {
        double s_min = std::exp(g.x[0]);
        if (option_type == QK_PUT) {
            U[idx(0, j, Ns1)] = std::max(0.0, strike * std::exp(-r * tau) - s_min * std::exp(-q * tau));
        } else {
            U[idx(0, j, Ns1)] = 0.0;
        }

        double s_max = std::exp(g.x[g.Ns]);
        if (option_type == QK_CALL) {
            U[idx(g.Ns, j, Ns1)] = std::max(0.0, s_max * std::exp(-q * tau) - strike * std::exp(-r * tau));
        } else {
            U[idx(g.Ns, j, Ns1)] = 0.0;
        }
    }

    for (int32_t i = 0; i <= g.Ns; ++i) {
        double s = std::exp(g.x[i]);
        double fwd = s * std::exp((r - q) * tau);
        U[idx(i, 0, Ns1)] = std::exp(-r * tau) * detail::intrinsic_value(fwd, strike, option_type);
    }

    for (int32_t i = 0; i <= g.Ns; ++i) {
        U[idx(i, g.Nv, Ns1)] = U[idx(i, g.Nv - 1, Ns1)];
    }
}

void compute_operators(const std::vector<double>& U,
                       std::vector<double>& FU,
                       std::vector<double>& A1U,
                       std::vector<double>& A2U,
                       const ADIGrid& g, double r, double q, const ADIHestonParams& p) {
    int32_t Ns1 = g.Ns + 1;
    int32_t total = Ns1 * (g.Nv + 1);
    FU.assign(total, 0.0);
    A1U.assign(total, 0.0);
    A2U.assign(total, 0.0);

    double dx = g.dx, dv = g.dv;
    double dx2 = dx * dx, dv2 = dv * dv;

    for (int32_t j = 1; j < g.Nv; ++j) {
        double vj = g.v[j];
        for (int32_t i = 1; i < g.Ns; ++i) {
            int32_t k = idx(i, j, Ns1);

            double diff_x = 0.5 * vj / dx2;
            double conv_x = (r - q - 0.5 * vj) / (2.0 * dx);
            double a1 = (diff_x - conv_x) * U[idx(i - 1, j, Ns1)]
                      + (-2.0 * diff_x - r) * U[k]
                      + (diff_x + conv_x) * U[idx(i + 1, j, Ns1)];

            double diff_v = 0.5 * p.sigma * p.sigma * vj / dv2;
            double conv_v = p.kappa * (p.theta_v - vj) / (2.0 * dv);
            double a2 = (diff_v - conv_v) * U[idx(i, j - 1, Ns1)]
                      + (-2.0 * diff_v) * U[k]
                      + (diff_v + conv_v) * U[idx(i, j + 1, Ns1)];

            double a0 = p.rho * p.sigma * vj / (4.0 * dx * dv)
                       * (U[idx(i + 1, j + 1, Ns1)] - U[idx(i - 1, j + 1, Ns1)]
                        - U[idx(i + 1, j - 1, Ns1)] + U[idx(i - 1, j - 1, Ns1)]);

            A1U[k] = a1;
            A2U[k] = a2;
            FU[k] = a1 + a2 + a0;
        }
    }
}

void solve_x_implicit(std::vector<double>& Y, const std::vector<double>& rhs,
                      const ADIGrid& g, double r, double q,
                      double theta_dt, int32_t j) {
    int32_t Ns1 = g.Ns + 1;
    int32_t n = g.Ns - 1;
    if (n <= 0) return;

    double vj = g.v[j];
    double dx = g.dx;
    double dx2 = dx * dx;

    std::vector<double> a(n), b(n), c(n), d(n);

    for (int32_t ii = 0; ii < n; ++ii) {
        double diff = 0.5 * vj / dx2;
        double conv = (r - q - 0.5 * vj) / (2.0 * dx);

        double lower = diff - conv;
        double upper = diff + conv;
        double diag  = -2.0 * diff - r;

        a[ii] = -theta_dt * lower;
        b[ii] = 1.0 - theta_dt * diag;
        c[ii] = -theta_dt * upper;
        d[ii] = rhs[idx(ii + 1, j, Ns1)];
    }

    {
        double diff = 0.5 * vj / dx2;
        double conv = (r - q - 0.5 * vj) / (2.0 * dx);
        d[0] += theta_dt * (diff - conv) * Y[idx(0, j, Ns1)];
    }
    {
        double diff = 0.5 * vj / dx2;
        double conv = (r - q - 0.5 * vj) / (2.0 * dx);
        d[n - 1] += theta_dt * (diff + conv) * Y[idx(g.Ns, j, Ns1)];
    }

    detail::thomas_solve(a.data(), b.data(), c.data(), d.data(), n);

    for (int32_t ii = 0; ii < n; ++ii) {
        Y[idx(ii + 1, j, Ns1)] = d[ii];
    }
}

void solve_v_implicit(std::vector<double>& Y, const std::vector<double>& rhs,
                      const ADIGrid& g, const ADIHestonParams& p,
                      double theta_dt, int32_t i) {
    int32_t Ns1 = g.Ns + 1;
    int32_t n = g.Nv - 1;
    if (n <= 0) return;

    double dv = g.dv;
    double dv2 = dv * dv;

    std::vector<double> a(n), b(n), c(n), d(n);

    for (int32_t jj = 0; jj < n; ++jj) {
        int32_t j = jj + 1;
        double vj = g.v[j];
        double diff = 0.5 * p.sigma * p.sigma * vj / dv2;
        double conv = p.kappa * (p.theta_v - vj) / (2.0 * dv);

        a[jj] = -theta_dt * (diff - conv);
        b[jj] = 1.0 - theta_dt * (-2.0 * diff);
        c[jj] = -theta_dt * (diff + conv);
        d[jj] = rhs[idx(i, j, Ns1)];
    }

    {
        double v1 = g.v[1];
        double diff = 0.5 * p.sigma * p.sigma * v1 / dv2;
        double conv = p.kappa * (p.theta_v - v1) / (2.0 * dv);
        d[0] += theta_dt * (diff - conv) * Y[idx(i, 0, Ns1)];
    }
    {
        double vn = g.v[g.Nv - 1];
        double diff = 0.5 * p.sigma * p.sigma * vn / dv2;
        double conv = p.kappa * (p.theta_v - vn) / (2.0 * dv);
        d[n - 1] += theta_dt * (diff + conv) * Y[idx(i, g.Nv, Ns1)];
    }

    detail::thomas_solve(a.data(), b.data(), c.data(), d.data(), n);

    for (int32_t jj = 0; jj < n; ++jj) {
        Y[idx(i, jj + 1, Ns1)] = d[jj];
    }
}

bool valid_adi_inputs(double spot, double strike, double t, double r, double q,
                      const ADIHestonParams& p, int32_t option_type) {
    if (!detail::valid_option_type(option_type)) return false;
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(r) || !is_finite_safe(q))
        return false;
    if (spot <= 0.0 || strike <= 0.0 || t <= 0.0) return false;
    if (p.v0 < 0.0 || p.kappa < 0.0 || p.theta_v < 0.0 || p.sigma < 0.0) return false;
    if (p.rho < -1.0 || p.rho > 1.0) return false;
    if (p.s_steps < 2 || p.v_steps < 2 || p.time_steps < 1) return false;
    return true;
}

double extract_price_2d(const std::vector<double>& U, const ADIGrid& g,
                        double spot, double v0) {
    int32_t Ns1 = g.Ns + 1;
    double x_target = std::log(spot);

    int32_t ix = 0;
    for (int32_t i = 0; i < g.Ns; ++i) {
        if (g.x[i] <= x_target && x_target <= g.x[i + 1]) {
            ix = i;
            break;
        }
    }
    double wx = (x_target - g.x[ix]) / g.dx;

    int32_t jv = 0;
    for (int32_t j = 0; j < g.Nv; ++j) {
        if (g.v[j] <= v0 && v0 <= g.v[j + 1]) {
            jv = j;
            break;
        }
    }
    double wv = (v0 - g.v[jv]) / g.dv;

    double v00 = U[idx(ix, jv, Ns1)];
    double v10 = U[idx(ix + 1, jv, Ns1)];
    double v01 = U[idx(ix, jv + 1, Ns1)];
    double v11 = U[idx(ix + 1, jv + 1, Ns1)];

    return (1.0 - wx) * (1.0 - wv) * v00 + wx * (1.0 - wv) * v10
         + (1.0 - wx) * wv * v01 + wx * wv * v11;
}

} // anonymous namespace

// Douglas scheme:
// Y0 = U + dt*F(U)
// (I - θ·dt·A1) Y1 = Y0 - θ·dt·A1·U
// (I - θ·dt·A2) U_new = Y1 - θ·dt·A2·U
double adi_douglas_price(double spot, double strike, double t, double r, double q,
                         const ADIHestonParams& params, int32_t option_type) {
    if (!valid_adi_inputs(spot, strike, t, r, q, params, option_type))
        return detail::nan_value();

    ADIGrid g = build_2d_grid(spot, t, params);
    int32_t Ns1 = g.Ns + 1;
    int32_t total = Ns1 * (g.Nv + 1);
    double dt = t / static_cast<double>(params.time_steps);
    double theta = 0.5;

    std::vector<double> U(total);
    for (int32_t j = 0; j <= g.Nv; ++j) {
        for (int32_t i = 0; i <= g.Ns; ++i) {
            double s = std::exp(g.x[i]);
            U[idx(i, j, Ns1)] = detail::intrinsic_value(s, strike, option_type);
        }
    }

    std::vector<double> FU(total), A1U(total), A2U(total);
    std::vector<double> Y0(total), RHS(total), Y1(total);

    for (int32_t step = params.time_steps - 1; step >= 0; --step) {
        double tau = (params.time_steps - step) * dt;

        compute_operators(U, FU, A1U, A2U, g, r, q, params);

        for (int32_t k = 0; k < total; ++k) {
            Y0[k] = U[k] + dt * FU[k];
        }

        for (int32_t k = 0; k < total; ++k) {
            RHS[k] = Y0[k] - theta * dt * A1U[k];
        }
        Y1 = RHS;
        apply_boundaries(Y1, g, strike, r, q, tau, option_type);
        for (int32_t j = 1; j < g.Nv; ++j) {
            solve_x_implicit(Y1, RHS, g, r, q, theta * dt, j);
        }

        for (int32_t k = 0; k < total; ++k) {
            RHS[k] = Y1[k] - theta * dt * A2U[k];
        }
        U = RHS;
        apply_boundaries(U, g, strike, r, q, tau, option_type);
        for (int32_t i = 1; i < g.Ns; ++i) {
            solve_v_implicit(U, RHS, g, params, theta * dt, i);
        }

        apply_boundaries(U, g, strike, r, q, tau, option_type);
    }

    return extract_price_2d(U, g, spot, params.v0);
}

// Craig-Sneyd scheme:
// Same as Douglas but with mixed-derivative correction after predictor step
double adi_craig_sneyd_price(double spot, double strike, double t, double r, double q,
                             const ADIHestonParams& params, int32_t option_type) {
    if (!valid_adi_inputs(spot, strike, t, r, q, params, option_type))
        return detail::nan_value();

    ADIGrid g = build_2d_grid(spot, t, params);
    int32_t Ns1 = g.Ns + 1;
    int32_t total = Ns1 * (g.Nv + 1);
    double dt = t / static_cast<double>(params.time_steps);
    double theta = 0.5;

    std::vector<double> U(total);
    for (int32_t j = 0; j <= g.Nv; ++j) {
        for (int32_t i = 0; i <= g.Ns; ++i) {
            double s = std::exp(g.x[i]);
            U[idx(i, j, Ns1)] = detail::intrinsic_value(s, strike, option_type);
        }
    }

    std::vector<double> FU(total), A1U(total), A2U(total);
    std::vector<double> FYtilde(total), A1Yt(total), A2Yt(total);
    std::vector<double> Y0(total), RHS(total), Y1(total), Ytilde(total), Yhat(total);

    for (int32_t step = params.time_steps - 1; step >= 0; --step) {
        double tau = (params.time_steps - step) * dt;

        compute_operators(U, FU, A1U, A2U, g, r, q, params);

        for (int32_t k = 0; k < total; ++k) {
            Y0[k] = U[k] + dt * FU[k];
        }

        for (int32_t k = 0; k < total; ++k) {
            RHS[k] = Y0[k] - theta * dt * A1U[k];
        }
        Y1 = RHS;
        apply_boundaries(Y1, g, strike, r, q, tau, option_type);
        for (int32_t j = 1; j < g.Nv; ++j) {
            solve_x_implicit(Y1, RHS, g, r, q, theta * dt, j);
        }

        for (int32_t k = 0; k < total; ++k) {
            RHS[k] = Y1[k] - theta * dt * A2U[k];
        }
        Ytilde = RHS;
        apply_boundaries(Ytilde, g, strike, r, q, tau, option_type);
        for (int32_t i = 1; i < g.Ns; ++i) {
            solve_v_implicit(Ytilde, RHS, g, params, theta * dt, i);
        }

        compute_operators(Ytilde, FYtilde, A1Yt, A2Yt, g, r, q, params);
        for (int32_t k = 0; k < total; ++k) {
            Yhat[k] = Y0[k] + 0.5 * dt * (FYtilde[k] - FU[k]);
        }

        for (int32_t k = 0; k < total; ++k) {
            RHS[k] = Yhat[k] - theta * dt * A1U[k];
        }
        Y1 = RHS;
        apply_boundaries(Y1, g, strike, r, q, tau, option_type);
        for (int32_t j = 1; j < g.Nv; ++j) {
            solve_x_implicit(Y1, RHS, g, r, q, theta * dt, j);
        }

        for (int32_t k = 0; k < total; ++k) {
            RHS[k] = Y1[k] - theta * dt * A2U[k];
        }
        U = RHS;
        apply_boundaries(U, g, strike, r, q, tau, option_type);
        for (int32_t i = 1; i < g.Ns; ++i) {
            solve_v_implicit(U, RHS, g, params, theta * dt, i);
        }

        apply_boundaries(U, g, strike, r, q, tau, option_type);
    }

    return extract_price_2d(U, g, spot, params.v0);
}

// Hundsdorfer-Verwer scheme: two-stage predictor-corrector
double adi_hundsdorfer_verwer_price(double spot, double strike, double t, double r, double q,
                                   const ADIHestonParams& params, int32_t option_type) {
    if (!valid_adi_inputs(spot, strike, t, r, q, params, option_type))
        return detail::nan_value();

    ADIGrid g = build_2d_grid(spot, t, params);
    int32_t Ns1 = g.Ns + 1;
    int32_t total = Ns1 * (g.Nv + 1);
    double dt = t / static_cast<double>(params.time_steps);
    double theta = 0.5;

    std::vector<double> U(total);
    for (int32_t j = 0; j <= g.Nv; ++j) {
        for (int32_t i = 0; i <= g.Ns; ++i) {
            double s = std::exp(g.x[i]);
            U[idx(i, j, Ns1)] = detail::intrinsic_value(s, strike, option_type);
        }
    }

    std::vector<double> FU(total), A1U(total), A2U(total);
    std::vector<double> FYt(total), A1Yt(total), A2Yt(total);
    std::vector<double> Y0(total), RHS(total), Y1(total), Ytilde(total);

    for (int32_t step = params.time_steps - 1; step >= 0; --step) {
        double tau = (params.time_steps - step) * dt;

        compute_operators(U, FU, A1U, A2U, g, r, q, params);

        for (int32_t k = 0; k < total; ++k) {
            Y0[k] = U[k] + dt * FU[k];
        }

        for (int32_t k = 0; k < total; ++k) {
            RHS[k] = Y0[k] - theta * dt * A1U[k];
        }
        Y1 = RHS;
        apply_boundaries(Y1, g, strike, r, q, tau, option_type);
        for (int32_t j = 1; j < g.Nv; ++j) {
            solve_x_implicit(Y1, RHS, g, r, q, theta * dt, j);
        }

        for (int32_t k = 0; k < total; ++k) {
            RHS[k] = Y1[k] - theta * dt * A2U[k];
        }
        Ytilde = RHS;
        apply_boundaries(Ytilde, g, strike, r, q, tau, option_type);
        for (int32_t i = 1; i < g.Ns; ++i) {
            solve_v_implicit(Ytilde, RHS, g, params, theta * dt, i);
        }

        compute_operators(Ytilde, FYt, A1Yt, A2Yt, g, r, q, params);
        for (int32_t k = 0; k < total; ++k) {
            Y0[k] = Ytilde[k] + 0.5 * dt * (FYt[k] - FU[k]);
        }

        for (int32_t k = 0; k < total; ++k) {
            RHS[k] = Y0[k] - theta * dt * A1Yt[k];
        }
        Y1 = RHS;
        apply_boundaries(Y1, g, strike, r, q, tau, option_type);
        for (int32_t j = 1; j < g.Nv; ++j) {
            solve_x_implicit(Y1, RHS, g, r, q, theta * dt, j);
        }

        for (int32_t k = 0; k < total; ++k) {
            RHS[k] = Y1[k] - theta * dt * A2Yt[k];
        }
        U = RHS;
        apply_boundaries(U, g, strike, r, q, tau, option_type);
        for (int32_t i = 1; i < g.Ns; ++i) {
            solve_v_implicit(U, RHS, g, params, theta * dt, i);
        }

        apply_boundaries(U, g, strike, r, q, tau, option_type);
    }

    return extract_price_2d(U, g, spot, params.v0);
}

} // namespace qk::fdm
