#include <quantkernel/qk_api.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>

int main(int argc, char** argv) {
    int32_t n = 200000;
    int32_t repeats = 5;
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
    std::vector<int32_t> option_type(n);
    for (int32_t i = 0; i < n; ++i) {
        spot[i] = d_spot(rng);
        strike[i] = d_strike(rng);
        t[i] = d_t(rng);
        vol[i] = d_vol(rng);
        r[i] = d_r(rng);
        q[i] = d_q(rng);
        option_type[i] = (i & 1) == 0 ? QK_CALL : QK_PUT;
    }

    volatile double sink = 0.0;
    for (int32_t i = 0; i < std::min<int32_t>(n, 1024); ++i) {
        sink += qk_cf_black_scholes_merton_price(
            spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i]
        );
    }
    qk_cf_black_scholes_merton_price_batch(
        spot.data(), strike.data(), t.data(), vol.data(), r.data(), q.data(),
        option_type.data(), n, out.data()
    );
    sink += out[0];

    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto dur) {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(dur).count();
    };

    const auto t0 = now();
    for (int32_t rep = 0; rep < repeats; ++rep) {
        for (int32_t i = 0; i < n; ++i) {
            sink += qk_cf_black_scholes_merton_price(
                spot[i], strike[i], t[i], vol[i], r[i], q[i], option_type[i]
            );
        }
    }
    const auto t1 = now();

    const auto t2 = now();
    for (int32_t rep = 0; rep < repeats; ++rep) {
        const int32_t rc = qk_cf_black_scholes_merton_price_batch(
            spot.data(), strike.data(), t.data(), vol.data(), r.data(), q.data(),
            option_type.data(), n, out.data()
        );
        if (rc != QK_OK) {
            std::cerr << "batch API failed with code " << rc << "\n";
            return 1;
        }
        sink += out[rep % n];
    }
    const auto t3 = now();

    const double scalar_ms = ms(t1 - t0) / static_cast<double>(repeats);
    const double batch_ms = ms(t3 - t2) / static_cast<double>(repeats);
    const double speedup = scalar_ms / std::max(1e-12, batch_ms);

    std::cout << std::fixed << std::setprecision(6)
              << "{\"n\":" << n
              << ",\"repeats\":" << repeats
              << ",\"cpp_scalar_ms\":" << scalar_ms
              << ",\"cpp_batch_ms\":" << batch_ms
              << ",\"cpp_batch_speedup\":" << speedup
              << ",\"checksum\":" << sink
              << "}\n";
    return 0;
}
