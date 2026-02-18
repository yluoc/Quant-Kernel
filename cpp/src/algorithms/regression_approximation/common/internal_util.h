#ifndef QK_RAM_INTERNAL_UTIL_H
#define QK_RAM_INTERNAL_UTIL_H

#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"
#include "common/option_util.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace qk::ram::detail {

constexpr double kEps = 1e-12;

inline double nan_value() { return qk::nan_value(); }

inline bool valid_option_type(int32_t option_type) { return qk::valid_option_type(option_type); }

inline bool valid_common_inputs(double spot, double strike, double t, double vol,
                                double r, double q, int32_t option_type) {
    return qk::valid_common_inputs(spot, strike, t, vol, r, q, option_type);
}

inline double call_from_bsm(double spot, double strike, double t, double vol, double r, double q) {
    return qk::cfa::black_scholes_merton_price(spot, strike, t, vol, r, q, QK_CALL);
}

inline double call_put_from_call_parity(double call_price, double spot, double strike,
                                        double t, double r, double q, int32_t option_type) {
    if (option_type == QK_CALL) return call_price;
    if (option_type == QK_PUT) {
        return call_price - spot * std::exp(-q * t) + strike * std::exp(-r * t);
    }
    return nan_value();
}

inline double clamp_unit_interval(double x) {
    return std::min(1.0, std::max(0.0, x));
}

// Gaussian elimination with partial pivoting.
// Solves A * x = b in place. A is n x n stored row-major. b is length n.
// Returns false if singular.
inline bool solve_linear_system(std::vector<double>& A, std::vector<double>& b, int n) {
    for (int col = 0; col < n; ++col) {
        // Partial pivoting
        int max_row = col;
        double max_val = std::fabs(A[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            double v = std::fabs(A[row * n + col]);
            if (v > max_val) { max_val = v; max_row = row; }
        }
        if (max_val < 1e-15) return false;
        if (max_row != col) {
            for (int j = col; j < n; ++j) std::swap(A[col * n + j], A[max_row * n + j]);
            std::swap(b[col], b[max_row]);
        }
        // Elimination
        const double pivot = A[col * n + col];
        for (int row = col + 1; row < n; ++row) {
            const double factor = A[row * n + col] / pivot;
            for (int j = col; j < n; ++j) {
                A[row * n + j] -= factor * A[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }
    // Back substitution
    for (int row = n - 1; row >= 0; --row) {
        for (int j = row + 1; j < n; ++j) {
            b[row] -= A[row * n + j] * b[j];
        }
        b[row] /= A[row * n + row];
    }
    return true;
}

// Gauss-Hermite quadrature nodes and weights via Golub-Welsch algorithm.
// Returns (nodes, weights) for n-point rule integrating f(x)*exp(-x^2).
inline void gauss_hermite_nodes(int n, std::vector<double>& nodes, std::vector<double>& weights) {
    nodes.resize(n);
    weights.resize(n);
    if (n <= 0) return;
    if (n == 1) {
        nodes[0] = 0.0;
        weights[0] = std::sqrt(M_PI);
        return;
    }

    // Tridiagonal matrix: diagonal = 0, sub-diagonal[i] = sqrt(i/2)
    std::vector<double> diag(n, 0.0);
    std::vector<double> sub(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        sub[i] = std::sqrt(static_cast<double>(i + 1) / 2.0);
    }

    // Eigenvectors stored as rows of V (init to identity)
    std::vector<double> V(n * n, 0.0);
    for (int i = 0; i < n; ++i) V[i * n + i] = 1.0;

    // QR iteration (implicit symmetric tridiagonal)
    for (int iter = 0; iter < 100 * n; ++iter) {
        bool converged = true;
        for (int i = 0; i < n - 1; ++i) {
            if (std::fabs(sub[i]) > 1e-14 * (std::fabs(diag[i]) + std::fabs(diag[i + 1]) + 1.0)) {
                converged = false;
                break;
            }
        }
        if (converged) break;

        // Wilkinson shift
        double d = (diag[n - 2] - diag[n - 1]) / 2.0;
        double mu = diag[n - 1] - sub[n - 2] * sub[n - 2] /
                    (d + (d >= 0 ? 1.0 : -1.0) * std::sqrt(d * d + sub[n - 2] * sub[n - 2]));

        double x = diag[0] - mu;
        double z = sub[0];
        for (int k = 0; k < n - 1; ++k) {
            double r = std::sqrt(x * x + z * z);
            double c = x / r;
            double s = z / r;

            double w = c * diag[k] + s * sub[k];
            double dk1 = -s * diag[k] + c * sub[k];
            diag[k] = c * w + s * sub[k];
            sub[k] = dk1;

            double w2 = c * sub[k] + s * diag[k + 1];
            diag[k + 1] = -s * sub[k] + c * diag[k + 1];
            sub[k] = w2;

            if (k < n - 2) {
                double tmp = c * 0.0 + s * sub[k + 1];
                sub[k + 1] = c * sub[k + 1];
                z = tmp;
                x = sub[k];
            }

            // Update eigenvectors
            for (int j = 0; j < n; ++j) {
                double v0 = V[j * n + k];
                double v1 = V[j * n + k + 1];
                V[j * n + k] = c * v0 + s * v1;
                V[j * n + k + 1] = -s * v0 + c * v1;
            }
        }
    }

    // Eigenvalues are nodes, weights from first component of eigenvectors
    for (int i = 0; i < n; ++i) {
        nodes[i] = diag[i];
        weights[i] = std::sqrt(M_PI) * V[0 * n + i] * V[0 * n + i];
    }

    // Sort by node value
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (nodes[j] < nodes[i]) {
                std::swap(nodes[i], nodes[j]);
                std::swap(weights[i], weights[j]);
            }
        }
    }
}

} // namespace qk::ram::detail

#endif /* QK_RAM_INTERNAL_UTIL_H */
