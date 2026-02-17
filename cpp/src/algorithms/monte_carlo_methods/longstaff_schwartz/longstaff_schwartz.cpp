#include "algorithms/monte_carlo_methods/longstaff_schwartz/longstaff_schwartz.h"

#include "algorithms/monte_carlo_methods/common/internal_util.h"

#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace qk::mcm {

namespace {

std::array<double, 3> solve_3x3(const std::array<std::array<double, 3>, 3>& a,
                                const std::array<double, 3>& b) {
    std::array<std::array<double, 4>, 3> m{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) m[i][j] = a[i][j];
        m[i][3] = b[i];
    }

    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 3; ++row) {
            if (std::fabs(m[row][col]) > std::fabs(m[pivot][col])) pivot = row;
        }
        if (std::fabs(m[pivot][col]) < 1e-14) return {0.0, 0.0, 0.0};
        if (pivot != col) std::swap(m[pivot], m[col]);

        double diag = m[col][col];
        for (int j = col; j < 4; ++j) m[col][j] /= diag;

        for (int row = 0; row < 3; ++row) {
            if (row == col) continue;
            double factor = m[row][col];
            for (int j = col; j < 4; ++j) m[row][j] -= factor * m[col][j];
        }
    }

    return {m[0][3], m[1][3], m[2][3]};
}

} // namespace

double longstaff_schwartz_price(double spot, double strike, double t, double vol,
                                double r, double q, int32_t option_type,
                                int32_t paths, int32_t steps, uint64_t seed) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        !detail::valid_mc_counts(paths, steps)) {
        return detail::nan_value();
    }

    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    const double dt = t / static_cast<double>(steps);
    const double disc = std::exp(-r * dt);
    const double drift = (r - q - 0.5 * vol * vol) * dt;
    const double vol_step = vol * std::sqrt(dt);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<std::vector<double>> path_values(
        static_cast<size_t>(paths), std::vector<double>(static_cast<size_t>(steps + 1), spot)
    );

    for (int32_t i = 0; i < paths; ++i) {
        for (int32_t j = 1; j <= steps; ++j) {
            double z = normal(rng);
            path_values[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                path_values[static_cast<size_t>(i)][static_cast<size_t>(j - 1)] *
                std::exp(drift + vol_step * z);
        }
    }

    std::vector<double> cashflow(static_cast<size_t>(paths), 0.0);
    for (int32_t i = 0; i < paths; ++i) {
        cashflow[static_cast<size_t>(i)] = detail::payoff(
            path_values[static_cast<size_t>(i)][static_cast<size_t>(steps)], strike, option_type
        );
    }

    for (int32_t j = steps - 1; j >= 1; --j) {
        std::vector<int32_t> itm_idx;
        itm_idx.reserve(static_cast<size_t>(paths));

        for (int32_t i = 0; i < paths; ++i) {
            cashflow[static_cast<size_t>(i)] *= disc;
            double exercise = detail::payoff(
                path_values[static_cast<size_t>(i)][static_cast<size_t>(j)], strike, option_type
            );
            if (exercise > 0.0) itm_idx.push_back(i);
        }

        if (itm_idx.size() < 3U) continue;

        std::array<std::array<double, 3>, 3> a{{{0.0, 0.0, 0.0},
                                                {0.0, 0.0, 0.0},
                                                {0.0, 0.0, 0.0}}};
        std::array<double, 3> b{{0.0, 0.0, 0.0}};

        for (int32_t idx : itm_idx) {
            double s = path_values[static_cast<size_t>(idx)][static_cast<size_t>(j)];
            double y = cashflow[static_cast<size_t>(idx)];
            std::array<double, 3> x{{1.0, s, s * s}};
            for (int r_i = 0; r_i < 3; ++r_i) {
                b[r_i] += x[r_i] * y;
                for (int c_i = 0; c_i < 3; ++c_i) {
                    a[r_i][c_i] += x[r_i] * x[c_i];
                }
            }
        }

        auto beta = solve_3x3(a, b);

        for (int32_t idx : itm_idx) {
            double s = path_values[static_cast<size_t>(idx)][static_cast<size_t>(j)];
            double exercise = detail::payoff(s, strike, option_type);
            double continuation = beta[0] + beta[1] * s + beta[2] * s * s;
            if (exercise > continuation) {
                cashflow[static_cast<size_t>(idx)] = exercise;
            }
        }
    }

    double mean_cf = std::accumulate(cashflow.begin(), cashflow.end(), 0.0) /
                     static_cast<double>(paths);
    return mean_cf * disc;
}

} // namespace qk::mcm
