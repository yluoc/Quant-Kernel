#ifndef QK_MLM_INTERNAL_UTIL_H
#define QK_MLM_INTERNAL_UTIL_H

#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"
#include "common/option_util.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace qk::mlm::detail {

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

inline double clamp01(double x) {
    return std::min(1.0, std::max(0.0, x));
}

inline double splitmix64_hash(uint64_t x) { return qk::splitmix64_hash(x); }

// Simple 2-layer MLP for inline ML algorithms
struct SimpleMLP {
    int input_dim;
    int hidden_dim;
    int output_dim;
    std::vector<double> W1; // hidden_dim x input_dim
    std::vector<double> b1; // hidden_dim
    std::vector<double> W2; // output_dim x hidden_dim
    std::vector<double> b2; // output_dim

    // Intermediate storage for backward pass
    std::vector<double> h_pre;  // pre-activation hidden
    std::vector<double> h_post; // post-activation hidden (tanh)
    std::vector<double> out;    // output

    void init(int in_dim, int hid_dim, int out_dim, std::mt19937_64& rng) {
        input_dim = in_dim;
        hidden_dim = hid_dim;
        output_dim = out_dim;

        W1.resize(hidden_dim * input_dim);
        b1.resize(hidden_dim, 0.0);
        W2.resize(output_dim * hidden_dim);
        b2.resize(output_dim, 0.0);
        h_pre.resize(hidden_dim);
        h_post.resize(hidden_dim);
        out.resize(output_dim);

        // Xavier initialization
        std::normal_distribution<double> dist1(0.0, std::sqrt(2.0 / (input_dim + hidden_dim)));
        std::normal_distribution<double> dist2(0.0, std::sqrt(2.0 / (hidden_dim + output_dim)));

        for (auto& w : W1) w = dist1(rng);
        for (auto& w : W2) w = dist2(rng);
    }

    const std::vector<double>& forward(const std::vector<double>& x) {
        // Hidden layer: tanh(W1 * x + b1)
        for (int i = 0; i < hidden_dim; ++i) {
            double sum = b1[i];
            for (int j = 0; j < input_dim; ++j) {
                sum += W1[i * input_dim + j] * x[j];
            }
            h_pre[i] = sum;
            h_post[i] = std::tanh(sum);
        }
        // Output layer: W2 * h + b2
        for (int i = 0; i < output_dim; ++i) {
            double sum = b2[i];
            for (int j = 0; j < hidden_dim; ++j) {
                sum += W2[i * hidden_dim + j] * h_post[j];
            }
            out[i] = sum;
        }
        return out;
    }

    // SGD update given dL/d(out). Returns dL/d(input) for chaining.
    std::vector<double> backward(const std::vector<double>& x,
                                 const std::vector<double>& d_out,
                                 double lr) {
        // Gradient for W2, b2
        std::vector<double> d_h(hidden_dim, 0.0);
        for (int i = 0; i < output_dim; ++i) {
            for (int j = 0; j < hidden_dim; ++j) {
                d_h[j] += W2[i * hidden_dim + j] * d_out[i];
                W2[i * hidden_dim + j] -= lr * d_out[i] * h_post[j];
            }
            b2[i] -= lr * d_out[i];
        }
        // Through tanh
        std::vector<double> d_pre(hidden_dim);
        for (int i = 0; i < hidden_dim; ++i) {
            d_pre[i] = d_h[i] * (1.0 - h_post[i] * h_post[i]);
        }
        // Gradient for W1, b1 and dL/d(input)
        std::vector<double> d_input(input_dim, 0.0);
        for (int i = 0; i < hidden_dim; ++i) {
            for (int j = 0; j < input_dim; ++j) {
                d_input[j] += W1[i * input_dim + j] * d_pre[i];
                W1[i * input_dim + j] -= lr * d_pre[i] * x[j];
            }
            b1[i] -= lr * d_pre[i];
        }
        return d_input;
    }
};

} // namespace qk::mlm::detail

#endif /* QK_MLM_INTERNAL_UTIL_H */
