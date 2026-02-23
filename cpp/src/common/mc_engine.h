#ifndef QK_MC_ENGINE_H
#define QK_MC_ENGINE_H

#include <cmath>
#include <cstdint>
#include <random>

namespace qk::mc {

// ---------------------------------------------------------------------------
// Convenience: build a standard N(0,1) generator from seed.
// Returns a stateful lambda that draws from mt19937_64 + normal(0,1).
// The RNG + distribution are captured by value (moved into the lambda).
// ---------------------------------------------------------------------------
inline auto make_mt19937_normal(uint64_t seed) {
    struct Gen {
        std::mt19937_64 rng;
        std::normal_distribution<double> dist{0.0, 1.0};
        explicit Gen(uint64_t s) : rng(s) {}
        double operator()() { return dist(rng); }
    };
    return Gen(seed);
}

// ---------------------------------------------------------------------------
// Terminal-value MC estimator with antithetic variates.
//
// GenNormal:     () -> double  (returns a standard-normal draw)
// TerminalModel: (double spot, double t, double z) -> double S_T
// Accumulator:   (double S_T, double z, int path_index) -> double
// ---------------------------------------------------------------------------
template<typename GenNormal, typename TerminalModel, typename Accumulator>
double estimate_terminal_antithetic(
    double spot, double t, int32_t n_paths,
    GenNormal&& gen, TerminalModel&& model, Accumulator&& accumulate)
{
    double sum = 0.0;
    const int32_t n_pairs = n_paths / 2;
    for (int32_t i = 0; i < n_pairs; ++i) {
        const double z = gen();
        const double st_up = model(spot, t, z);
        const double st_dn = model(spot, t, -z);
        sum += accumulate(st_up, z, i * 2);
        sum += accumulate(st_dn, -z, i * 2 + 1);
    }
    if ((n_paths & 1) != 0) {
        const double z = gen();
        const double st = model(spot, t, z);
        sum += accumulate(st, z, n_paths - 1);
    }

    return sum / static_cast<double>(n_paths);
}

// ---------------------------------------------------------------------------
// Terminal-value MC estimator without antithetic (simple loop).
//
// GenNormal:     () -> double
// TerminalModel: (double spot, double t, double z) -> double S_T
// Accumulator:   (double S_T, double z, int path_index) -> double
// ---------------------------------------------------------------------------
template<typename GenNormal, typename TerminalModel, typename Accumulator>
double estimate_terminal(
    double spot, double t, int32_t n_paths,
    GenNormal&& gen, TerminalModel&& model, Accumulator&& accumulate)
{
    double sum = 0.0;
    for (int32_t i = 0; i < n_paths; ++i) {
        const double z = gen();
        const double st = model(spot, t, z);
        sum += accumulate(st, z, i);
    }

    return sum / static_cast<double>(n_paths);
}

// ---------------------------------------------------------------------------
// Step-wise MC estimator (Euler/Milstein).
//
// GenNormal:    () -> double
// StepModel:    (double s, double dt, double dw) -> double s_next
// Accumulator:  (double S_T, int path_index) -> double
// ---------------------------------------------------------------------------
template<typename GenNormal, typename StepModel, typename Accumulator>
double estimate_stepwise(
    double spot, double t, int32_t n_paths, int32_t steps,
    GenNormal&& gen, StepModel&& step, Accumulator&& accumulate)
{
    const double dt = t / static_cast<double>(steps);
    const double sqrt_dt = std::sqrt(dt);

    double sum = 0.0;
    for (int32_t i = 0; i < n_paths; ++i) {
        double s = spot;
        for (int32_t j = 0; j < steps; ++j) {
            double dw = sqrt_dt * gen();
            s = step(s, dt, dw);
        }
        sum += accumulate(s, i);
    }

    return sum / static_cast<double>(n_paths);
}

// ---------------------------------------------------------------------------
// Two-dimensional step-wise MC estimator (Heston and other SV models).
//
// GenNormal:     () -> double
// StepModel2D:   (double s, double v, double dt, double sqrt_dt,
//                 double z1, double z2) -> StepResult2D{s, v}
// Accumulator:   (double S_T, int path_index) -> double
// ---------------------------------------------------------------------------
template<typename GenNormal, typename StepModel2D, typename Accumulator>
double estimate_stepwise_2d(
    double spot, double v0, double t, int32_t n_paths, int32_t steps,
    GenNormal&& gen, StepModel2D&& step, Accumulator&& accumulate)
{
    const double dt = t / static_cast<double>(steps);
    const double sqrt_dt = std::sqrt(dt);

    double sum = 0.0;
    for (int32_t i = 0; i < n_paths; ++i) {
        double s = spot;
        double v = v0;
        for (int32_t j = 0; j < steps; ++j) {
            const double z1 = gen();
            const double z2 = gen();
            auto result = step(s, v, dt, sqrt_dt, z1, z2);
            s = result.s;
            v = result.v;
        }
        sum += accumulate(s, i);
    }

    return sum / static_cast<double>(n_paths);
}

} // namespace qk::mc

#endif /* QK_MC_ENGINE_H */
