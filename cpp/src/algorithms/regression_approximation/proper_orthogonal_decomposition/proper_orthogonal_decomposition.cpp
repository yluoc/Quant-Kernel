#include "algorithms/regression_approximation/proper_orthogonal_decomposition/proper_orthogonal_decomposition.h"

#include "algorithms/regression_approximation/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace qk::ram {

double proper_orthogonal_decomposition_price(
    double spot, double strike, double t, double vol, double r, double q, int32_t option_type,
    const ProperOrthogonalDecompositionParams& params
) {
    if (!detail::valid_common_inputs(spot, strike, t, vol, r, q, option_type) ||
        params.modes < 1 || params.snapshots < params.modes) {
        return detail::nan_value();
    }
    const double bsm_call = detail::call_from_bsm(spot, strike, t, vol, r, q);
    if (!is_finite_safe(bsm_call)) return detail::nan_value();

    const int n_modes = std::min(params.modes, 32);
    const int n_snaps = std::min(params.snapshots, 128);
    const int n_quad = std::min(256, std::max(64, n_snaps));

    std::vector<double> gh_nodes;
    std::vector<double> gh_weights;
    detail::gauss_hermite_nodes(n_quad, gh_nodes, gh_weights);
    if (static_cast<int>(gh_nodes.size()) != n_quad) return detail::nan_value();

    const double mu = (r - q - 0.5 * vol * vol) * t;
    const double sigma_t = vol * std::sqrt(t);

    std::vector<double> X(static_cast<std::size_t>(n_quad) * static_cast<std::size_t>(n_snaps), 0.0);
    for (int s = 0; s < n_snaps; ++s) {
        const double xi = (n_snaps == 1) ? 0.0 : (-1.0 + 2.0 * static_cast<double>(s) / static_cast<double>(n_snaps - 1));
        const double vol_s = std::max(1e-6, vol * (1.0 + 0.35 * xi));
        const double mu_s = (r - q - 0.5 * vol_s * vol_s) * t;
        const double sigma_s_t = vol_s * std::sqrt(t);

        for (int i = 0; i < n_quad; ++i) {
            const double z = std::sqrt(2.0) * gh_nodes[static_cast<std::size_t>(i)];
            const double st = spot * std::exp(mu_s + sigma_s_t * z);
            X[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_snaps) + static_cast<std::size_t>(s)] =
                std::max(st - strike, 0.0);
        }
    }

    std::vector<double> G(static_cast<std::size_t>(n_snaps) * static_cast<std::size_t>(n_snaps), 0.0);
    for (int a = 0; a < n_snaps; ++a) {
        for (int b = a; b < n_snaps; ++b) {
            double v = 0.0;
            for (int i = 0; i < n_quad; ++i) {
                v += X[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_snaps) + static_cast<std::size_t>(a)] *
                     X[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_snaps) + static_cast<std::size_t>(b)];
            }
            G[static_cast<std::size_t>(a) * static_cast<std::size_t>(n_snaps) + static_cast<std::size_t>(b)] = v;
            G[static_cast<std::size_t>(b) * static_cast<std::size_t>(n_snaps) + static_cast<std::size_t>(a)] = v;
        }
    }

    std::vector<std::vector<double>> eigvecs;
    eigvecs.reserve(static_cast<std::size_t>(n_modes));
    std::vector<double> eigvals;
    eigvals.reserve(static_cast<std::size_t>(n_modes));

    for (int m = 0; m < n_modes; ++m) {
        std::vector<double> v(static_cast<std::size_t>(n_snaps), 1.0 / std::sqrt(static_cast<double>(n_snaps)));

        for (int it = 0; it < 64; ++it) {
            std::vector<double> w(static_cast<std::size_t>(n_snaps), 0.0);
            for (int i = 0; i < n_snaps; ++i) {
                for (int j = 0; j < n_snaps; ++j) {
                    w[static_cast<std::size_t>(i)] +=
                        G[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_snaps) + static_cast<std::size_t>(j)]
                        * v[static_cast<std::size_t>(j)];
                }
            }

            for (int k = 0; k < m; ++k) {
                double proj = 0.0;
                for (int i = 0; i < n_snaps; ++i) {
                    proj += w[static_cast<std::size_t>(i)] * eigvecs[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)];
                }
                for (int i = 0; i < n_snaps; ++i) {
                    w[static_cast<std::size_t>(i)] -= proj * eigvecs[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)];
                }
            }

            double norm_w = 0.0;
            for (double wi : w) norm_w += wi * wi;
            norm_w = std::sqrt(norm_w);
            if (norm_w <= 1e-14) break;
            for (int i = 0; i < n_snaps; ++i) {
                v[static_cast<std::size_t>(i)] = w[static_cast<std::size_t>(i)] / norm_w;
            }
        }

        double lambda = 0.0;
        for (int i = 0; i < n_snaps; ++i) {
            double gvi = 0.0;
            for (int j = 0; j < n_snaps; ++j) {
                gvi += G[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_snaps) + static_cast<std::size_t>(j)]
                    * v[static_cast<std::size_t>(j)];
            }
            lambda += v[static_cast<std::size_t>(i)] * gvi;
        }
        if (lambda <= 1e-12) break;
        eigvecs.push_back(v);
        eigvals.push_back(lambda);
    }
    if (eigvecs.empty()) return detail::nan_value();

    const int used_modes = static_cast<int>(eigvecs.size());
    std::vector<double> modes(static_cast<std::size_t>(used_modes) * static_cast<std::size_t>(n_quad), 0.0);
    for (int m = 0; m < used_modes; ++m) {
        const double inv_sigma = 1.0 / std::sqrt(eigvals[static_cast<std::size_t>(m)]);
        for (int i = 0; i < n_quad; ++i) {
            double u = 0.0;
            for (int s = 0; s < n_snaps; ++s) {
                u += X[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_snaps) + static_cast<std::size_t>(s)] *
                     eigvecs[static_cast<std::size_t>(m)][static_cast<std::size_t>(s)];
            }
            modes[static_cast<std::size_t>(m) * static_cast<std::size_t>(n_quad) + static_cast<std::size_t>(i)] = u * inv_sigma;
        }
    }

    // Weighted Gram-Schmidt under Gaussian measure to stabilize projection.
    std::vector<int> active_mode_indices;
    active_mode_indices.reserve(static_cast<std::size_t>(used_modes));
    for (int m = 0; m < used_modes; ++m) {
        for (int k_idx = 0; k_idx < static_cast<int>(active_mode_indices.size()); ++k_idx) {
            const int k = active_mode_indices[static_cast<std::size_t>(k_idx)];
            double proj = 0.0;
            for (int i = 0; i < n_quad; ++i) {
                const double w = gh_weights[static_cast<std::size_t>(i)] / std::sqrt(M_PI);
                proj += w *
                    modes[static_cast<std::size_t>(m) * static_cast<std::size_t>(n_quad) + static_cast<std::size_t>(i)] *
                    modes[static_cast<std::size_t>(k) * static_cast<std::size_t>(n_quad) + static_cast<std::size_t>(i)];
            }
            for (int i = 0; i < n_quad; ++i) {
                modes[static_cast<std::size_t>(m) * static_cast<std::size_t>(n_quad) + static_cast<std::size_t>(i)] -=
                    proj * modes[static_cast<std::size_t>(k) * static_cast<std::size_t>(n_quad) + static_cast<std::size_t>(i)];
            }
        }

        double norm_m = 0.0;
        for (int i = 0; i < n_quad; ++i) {
            const double w = gh_weights[static_cast<std::size_t>(i)] / std::sqrt(M_PI);
            const double v = modes[static_cast<std::size_t>(m) * static_cast<std::size_t>(n_quad) + static_cast<std::size_t>(i)];
            norm_m += w * v * v;
        }
        norm_m = std::sqrt(norm_m);
        if (norm_m <= 1e-12) continue;
        for (int i = 0; i < n_quad; ++i) {
            modes[static_cast<std::size_t>(m) * static_cast<std::size_t>(n_quad) + static_cast<std::size_t>(i)] /= norm_m;
        }
        active_mode_indices.push_back(m);
    }
    if (active_mode_indices.empty()) return detail::nan_value();

    std::vector<double> target(static_cast<std::size_t>(n_quad), 0.0);
    for (int i = 0; i < n_quad; ++i) {
        const double z = std::sqrt(2.0) * gh_nodes[static_cast<std::size_t>(i)];
        const double st = spot * std::exp(mu + sigma_t * z);
        target[static_cast<std::size_t>(i)] = std::max(st - strike, 0.0);
    }

    std::vector<double> recon(static_cast<std::size_t>(n_quad), 0.0);
    for (int k_idx = 0; k_idx < static_cast<int>(active_mode_indices.size()); ++k_idx) {
        const int m = active_mode_indices[static_cast<std::size_t>(k_idx)];
        double coef = 0.0;
        for (int i = 0; i < n_quad; ++i) {
            const double w = gh_weights[static_cast<std::size_t>(i)] / std::sqrt(M_PI);
            coef += w *
                modes[static_cast<std::size_t>(m) * static_cast<std::size_t>(n_quad) + static_cast<std::size_t>(i)] *
                target[static_cast<std::size_t>(i)];
        }
        for (int i = 0; i < n_quad; ++i) {
            recon[static_cast<std::size_t>(i)] += coef *
                modes[static_cast<std::size_t>(m) * static_cast<std::size_t>(n_quad) + static_cast<std::size_t>(i)];
        }
    }

    double expected_payoff_recon = 0.0;
    double expected_payoff_target = 0.0;
    double recon_err2 = 0.0;
    double target_norm2 = 0.0;
    for (int i = 0; i < n_quad; ++i) {
        const double w = gh_weights[static_cast<std::size_t>(i)] / std::sqrt(M_PI);
        const double r_i = std::max(0.0, recon[static_cast<std::size_t>(i)]);
        const double t_i = target[static_cast<std::size_t>(i)];
        expected_payoff_recon += w * r_i;
        expected_payoff_target += w * t_i;
        const double diff = r_i - t_i;
        recon_err2 += w * diff * diff;
        target_norm2 += w * t_i * t_i;
    }
    const double trust = std::exp(-recon_err2 / (target_norm2 + 1e-12));
    const double expected_payoff =
        trust * expected_payoff_recon + (1.0 - trust) * expected_payoff_target;
    const double approx_call = std::exp(-r * t) * std::max(0.0, expected_payoff);
    const double stable_call = detail::stabilized_call_price(approx_call, bsm_call, spot, strike, t, r, q);
    return detail::call_put_from_call_parity(stable_call, spot, strike, t, r, q, option_type);
}

} // namespace qk::ram
