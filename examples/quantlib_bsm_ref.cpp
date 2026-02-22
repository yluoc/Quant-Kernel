#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>

#include <ql/pricingengines/blackformula.hpp>

namespace {

void write_error(char* err_buf, int32_t err_buf_size, const char* msg) {
    if (err_buf == nullptr || err_buf_size <= 0) {
        return;
    }
    if (msg == nullptr) {
        err_buf[0] = '\0';
        return;
    }
    std::strncpy(err_buf, msg, static_cast<std::size_t>(err_buf_size - 1));
    err_buf[err_buf_size - 1] = '\0';
}

}  // namespace

extern "C" int32_t ql_ref_black_scholes_merton_price_batch(
    const double* spot,
    const double* strike,
    const double* tau,
    const double* vol,
    const double* r,
    const double* q,
    const int32_t* option_type,
    int32_t n,
    double* out,
    char* err_buf,
    int32_t err_buf_size) {
    if (spot == nullptr || strike == nullptr || tau == nullptr || vol == nullptr ||
        r == nullptr || q == nullptr || option_type == nullptr || out == nullptr) {
        write_error(err_buf, err_buf_size, "Null input pointer.");
        return 1;
    }
    if (n < 0) {
        write_error(err_buf, err_buf_size, "Batch size must be non-negative.");
        return 2;
    }

    constexpr double nan = std::numeric_limits<double>::quiet_NaN();

    try {
        for (int32_t i = 0; i < n; ++i) {
            const double s = spot[i];
            const double k = strike[i];
            const double t = tau[i];
            const double sigma = vol[i];
            const double rf = r[i];
            const double div = q[i];
            const int32_t ot = option_type[i];

            if (!(std::isfinite(s) && std::isfinite(k) && std::isfinite(t) &&
                  std::isfinite(sigma) && std::isfinite(rf) && std::isfinite(div)) ||
                s <= 0.0 || k <= 0.0 || t <= 0.0 || sigma < 0.0 || (ot != 0 && ot != 1)) {
                out[i] = nan;
                continue;
            }

            const QuantLib::Option::Type ql_type =
                (ot == 0) ? QuantLib::Option::Call : QuantLib::Option::Put;
            const double std_dev = sigma * std::sqrt(t);
            const double discount = std::exp(-rf * t);
            const double forward = s * std::exp((rf - div) * t);

            out[i] = QuantLib::blackFormula(ql_type, k, forward, std_dev, discount);
        }
    } catch (const std::exception& e) {
        write_error(err_buf, err_buf_size, e.what());
        return 3;
    } catch (...) {
        write_error(err_buf, err_buf_size, "Unknown QuantLib exception.");
        return 4;
    }

    write_error(err_buf, err_buf_size, "");
    return 0;
}
