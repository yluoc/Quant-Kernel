#include <quantkernel/qk_api.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>

static auto now() { return std::chrono::steady_clock::now(); }
static double ms(std::chrono::steady_clock::duration dur) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(dur).count();
}

struct BenchResult {
    const char* name;
    double scalar_ms;
    double batch_ms;
    double speedup;
};

int main(int argc, char** argv) {
    int32_t n = 50000;
    int32_t repeats = 3;
    if (argc > 1) n = std::max<int32_t>(1, std::atoi(argv[1]));
    if (argc > 2) repeats = std::max<int32_t>(1, std::atoi(argv[2]));

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> d_spot(80.0, 120.0);
    std::uniform_real_distribution<double> d_strike(80.0, 120.0);
    std::uniform_real_distribution<double> d_t(0.25, 2.0);
    std::uniform_real_distribution<double> d_vol(0.1, 0.6);
    std::uniform_real_distribution<double> d_r(0.0, 0.08);
    std::uniform_real_distribution<double> d_q(0.0, 0.04);

    std::vector<double> spot(n), strike(n), t(n), vol(n), r(n), q(n), out(n);
    std::vector<int32_t> option_type(n), steps(n);
    for (int32_t i = 0; i < n; ++i) {
        spot[i] = d_spot(rng);
        strike[i] = d_strike(rng);
        t[i] = d_t(rng);
        vol[i] = d_vol(rng);
        r[i] = d_r(rng);
        q[i] = d_q(rng);
        option_type[i] = (i & 1) == 0 ? QK_CALL : QK_PUT;
        steps[i] = 100;
    }

    volatile double sink = 0.0;
    std::vector<BenchResult> results;

    // --- BSM ---
    {
        for (int32_t i = 0; i < std::min(n, (int32_t)128); ++i)
            sink += qk_cf_black_scholes_merton_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i]);

        auto t0 = now();
        for (int32_t rep = 0; rep < repeats; ++rep)
            for (int32_t i = 0; i < n; ++i)
                sink += qk_cf_black_scholes_merton_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i]);
        auto t1 = now();

        auto t2 = now();
        for (int32_t rep = 0; rep < repeats; ++rep)
            qk_cf_black_scholes_merton_price_batch(spot.data(), strike.data(), t.data(), vol.data(), r.data(), q.data(), option_type.data(), n, out.data());
        auto t3 = now();

        double s_ms = ms(t1 - t0) / repeats, b_ms = ms(t3 - t2) / repeats;
        results.push_back({"BSM", s_ms, b_ms, s_ms / std::max(1e-12, b_ms)});
    }

    // --- Black76 ---
    {
        auto t0 = now();
        for (int32_t rep = 0; rep < repeats; ++rep)
            for (int32_t i = 0; i < n; ++i)
                sink += qk_cf_black76_price(spot[i], strike[i], t[i], vol[i], r[i], option_type[i]);
        auto t1 = now();

        auto t2 = now();
        for (int32_t rep = 0; rep < repeats; ++rep)
            qk_cf_black76_price_batch(spot.data(), strike.data(), t.data(), vol.data(), r.data(), option_type.data(), n, out.data());
        auto t3 = now();

        double s_ms = ms(t1 - t0) / repeats, b_ms = ms(t3 - t2) / repeats;
        results.push_back({"Black76", s_ms, b_ms, s_ms / std::max(1e-12, b_ms)});
    }

    // --- Bachelier ---
    {
        auto t0 = now();
        for (int32_t rep = 0; rep < repeats; ++rep)
            for (int32_t i = 0; i < n; ++i)
                sink += qk_cf_bachelier_price(spot[i], strike[i], t[i], vol[i], r[i], option_type[i]);
        auto t1 = now();

        auto t2 = now();
        for (int32_t rep = 0; rep < repeats; ++rep)
            qk_cf_bachelier_price_batch(spot.data(), strike.data(), t.data(), vol.data(), r.data(), option_type.data(), n, out.data());
        auto t3 = now();

        double s_ms = ms(t1 - t0) / repeats, b_ms = ms(t3 - t2) / repeats;
        results.push_back({"Bachelier", s_ms, b_ms, s_ms / std::max(1e-12, b_ms)});
    }

    // --- CRR Tree (steps=100) ---
    {
        int32_t tree_n = std::min(n, (int32_t)2000);
        std::vector<int32_t> am(tree_n, 0);

        auto t0 = now();
        for (int32_t rep = 0; rep < repeats; ++rep)
            for (int32_t i = 0; i < tree_n; ++i)
                sink += qk_tlm_crr_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], 100, 0);
        auto t1 = now();

        auto t2 = now();
        for (int32_t rep = 0; rep < repeats; ++rep)
            qk_tlm_crr_price_batch(spot.data(), strike.data(), t.data(), vol.data(), r.data(), q.data(), option_type.data(), steps.data(), am.data(), tree_n, out.data());
        auto t3 = now();

        double s_ms = ms(t1 - t0) / repeats, b_ms = ms(t3 - t2) / repeats;
        results.push_back({"CRR(100)", s_ms, b_ms, s_ms / std::max(1e-12, b_ms)});
    }

    // --- Standard MC (paths=1000) ---
    {
        int32_t mc_n = std::min(n, (int32_t)500);
        std::vector<int32_t> paths(mc_n, 1000);
        std::vector<uint64_t> seeds(mc_n);
        for (int32_t i = 0; i < mc_n; ++i) seeds[i] = 42 + i;

        auto t0 = now();
        for (int32_t rep = 0; rep < repeats; ++rep)
            for (int32_t i = 0; i < mc_n; ++i)
                sink += qk_mcm_standard_monte_carlo_price(spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i], 1000, seeds[i]);
        auto t1 = now();

        auto t2 = now();
        for (int32_t rep = 0; rep < repeats; ++rep)
            qk_mcm_standard_monte_carlo_price_batch(spot.data(), strike.data(), t.data(), vol.data(), r.data(), q.data(), option_type.data(), paths.data(), seeds.data(), mc_n, out.data());
        auto t3 = now();

        double s_ms = ms(t1 - t0) / repeats, b_ms = ms(t3 - t2) / repeats;
        results.push_back({"MC(1000)", s_ms, b_ms, s_ms / std::max(1e-12, b_ms)});
    }

    // --- Output ---
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "| Algorithm | Scalar (ms) | Batch (ms) | Speedup |\n";
    std::cout << "|-----------|-------------|------------|--------|\n";
    for (auto& res : results) {
        std::cout << "| " << std::setw(9) << res.name
                  << " | " << std::setw(11) << res.scalar_ms
                  << " | " << std::setw(10) << res.batch_ms
                  << " | " << std::setw(6) << res.speedup << "x |\n";
    }
    std::cout << "\nn=" << n << " repeats=" << repeats << " checksum=" << sink << "\n";
    return 0;
}
