#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>

#include <ql/exercise.hpp>
#include <ql/experimental/variancegamma/analyticvariancegammaengine.hpp>
#include <ql/experimental/variancegamma/variancegammaprocess.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/models/equity/hestonmodel.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/pricingengines/vanilla/analytichestonengine.hpp>
#include <ql/pricingengines/vanilla/jumpdiffusionengine.hpp>
#include <ql/processes/hestonprocess.hpp>
#include <ql/processes/merton76process.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/sabr.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

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

bool valid_option_type(int32_t option_type) {
    return option_type == 0 || option_type == 1;
}

QuantLib::Option::Type to_ql_type(int32_t option_type) {
    return option_type == 0 ? QuantLib::Option::Call : QuantLib::Option::Put;
}

QuantLib::Date eval_date() {
    return {15, QuantLib::January, 2024};
}

QuantLib::Date maturity_from_tau(double tau) {
    const int days = std::max(1, static_cast<int>(std::llround(365.0 * tau)));
    return eval_date() + days;
}

double ql_heston_price(
    double spot, double strike, double tau, double r, double q,
    double v0, double kappa, double theta, double sigma, double rho,
    int32_t option_type, int32_t integration_steps
) {
    const QuantLib::Date today = eval_date();
    const QuantLib::Date maturity = maturity_from_tau(tau);
    QuantLib::Settings::instance().evaluationDate() = today;

    const QuantLib::DayCounter dc = QuantLib::Actual365Fixed();
    auto s0 = QuantLib::ext::make_shared<QuantLib::SimpleQuote>(spot);
    auto risk_ts = QuantLib::ext::make_shared<QuantLib::FlatForward>(today, r, dc);
    auto div_ts = QuantLib::ext::make_shared<QuantLib::FlatForward>(today, q, dc);

    auto process = QuantLib::ext::make_shared<QuantLib::HestonProcess>(
        QuantLib::Handle<QuantLib::YieldTermStructure>(risk_ts),
        QuantLib::Handle<QuantLib::YieldTermStructure>(div_ts),
        QuantLib::Handle<QuantLib::Quote>(s0),
        v0, kappa, theta, sigma, rho
    );
    auto model = QuantLib::ext::make_shared<QuantLib::HestonModel>(process);
    const auto order = static_cast<QuantLib::Size>(std::min(192, std::max(32, integration_steps)));
    auto engine = QuantLib::ext::make_shared<QuantLib::AnalyticHestonEngine>(model, order);

    auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(to_ql_type(option_type), strike);
    auto exercise = QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(maturity);
    QuantLib::VanillaOption option(payoff, exercise);
    option.setPricingEngine(engine);
    return option.NPV();
}

double ql_merton_price(
    double spot, double strike, double tau, double vol, double r, double q,
    double jump_intensity, double jump_mean, double jump_vol, int32_t max_terms, int32_t option_type
) {
    const QuantLib::Date today = eval_date();
    const QuantLib::Date maturity = maturity_from_tau(tau);
    QuantLib::Settings::instance().evaluationDate() = today;

    const QuantLib::DayCounter dc = QuantLib::Actual365Fixed();
    const QuantLib::Calendar cal = QuantLib::NullCalendar();
    auto s0 = QuantLib::ext::make_shared<QuantLib::SimpleQuote>(spot);
    auto risk_ts = QuantLib::ext::make_shared<QuantLib::FlatForward>(today, r, dc);
    auto div_ts = QuantLib::ext::make_shared<QuantLib::FlatForward>(today, q, dc);
    auto vol_ts = QuantLib::ext::make_shared<QuantLib::BlackConstantVol>(today, cal, vol, dc);
    auto ji = QuantLib::ext::make_shared<QuantLib::SimpleQuote>(jump_intensity);
    auto jm = QuantLib::ext::make_shared<QuantLib::SimpleQuote>(jump_mean);
    auto jv = QuantLib::ext::make_shared<QuantLib::SimpleQuote>(jump_vol);

    auto process = QuantLib::ext::make_shared<QuantLib::Merton76Process>(
        QuantLib::Handle<QuantLib::Quote>(s0),
        QuantLib::Handle<QuantLib::YieldTermStructure>(div_ts),
        QuantLib::Handle<QuantLib::YieldTermStructure>(risk_ts),
        QuantLib::Handle<QuantLib::BlackVolTermStructure>(vol_ts),
        QuantLib::Handle<QuantLib::Quote>(ji),
        QuantLib::Handle<QuantLib::Quote>(jm),
        QuantLib::Handle<QuantLib::Quote>(jv)
    );

    auto engine = QuantLib::ext::make_shared<QuantLib::JumpDiffusionEngine>(
        process, 1e-10, static_cast<QuantLib::Size>(std::max(8, max_terms))
    );

    auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(to_ql_type(option_type), strike);
    auto exercise = QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(maturity);
    QuantLib::VanillaOption option(payoff, exercise);
    option.setPricingEngine(engine);
    return option.NPV();
}

double ql_vg_price(
    double spot, double strike, double tau, double r, double q,
    double sigma, double theta, double nu, int32_t option_type
) {
    const QuantLib::Date today = eval_date();
    const QuantLib::Date maturity = maturity_from_tau(tau);
    QuantLib::Settings::instance().evaluationDate() = today;

    const QuantLib::DayCounter dc = QuantLib::Actual365Fixed();
    auto s0 = QuantLib::ext::make_shared<QuantLib::SimpleQuote>(spot);
    auto risk_ts = QuantLib::ext::make_shared<QuantLib::FlatForward>(today, r, dc);
    auto div_ts = QuantLib::ext::make_shared<QuantLib::FlatForward>(today, q, dc);

    // Note: QuantLib constructor order is (sigma, nu, theta), while QuantKernel uses (sigma, theta, nu).
    auto process = QuantLib::ext::make_shared<QuantLib::VarianceGammaProcess>(
        QuantLib::Handle<QuantLib::Quote>(s0),
        QuantLib::Handle<QuantLib::YieldTermStructure>(div_ts),
        QuantLib::Handle<QuantLib::YieldTermStructure>(risk_ts),
        sigma, nu, theta
    );
    auto engine = QuantLib::ext::make_shared<QuantLib::VarianceGammaEngine>(process, 1e-8);

    auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(to_ql_type(option_type), strike);
    auto exercise = QuantLib::ext::make_shared<QuantLib::EuropeanExercise>(maturity);
    QuantLib::VanillaOption option(payoff, exercise);
    option.setPricingEngine(engine);
    return option.NPV();
}

double ql_sabr_iv(
    double forward, double strike, double tau,
    double alpha, double beta, double rho, double nu
) {
    // QuantLib order: alpha, beta, nu, rho
    return QuantLib::sabrVolatility(strike, forward, tau, alpha, beta, nu, rho);
}

double ql_sabr_black76_price(
    double forward, double strike, double tau, double r,
    double alpha, double beta, double rho, double nu, int32_t option_type
) {
    const double iv = ql_sabr_iv(forward, strike, tau, alpha, beta, rho, nu);
    const double std_dev = iv * std::sqrt(tau);
    const double discount = std::exp(-r * tau);
    return QuantLib::blackFormula(to_ql_type(option_type), strike, forward, std_dev, discount);
}

}  // namespace

extern "C" int32_t ql_ref_heston_price_batch(
    const double* spot,
    const double* strike,
    const double* tau,
    const double* r,
    const double* q,
    const double* v0,
    const double* kappa,
    const double* theta,
    const double* sigma,
    const double* rho,
    const int32_t* option_type,
    const int32_t* integration_steps,
    int32_t n,
    double* out,
    char* err_buf,
    int32_t err_buf_size) {
    if (spot == nullptr || strike == nullptr || tau == nullptr || r == nullptr || q == nullptr ||
        v0 == nullptr || kappa == nullptr || theta == nullptr || sigma == nullptr || rho == nullptr ||
        option_type == nullptr || integration_steps == nullptr || out == nullptr) {
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
            if (!(std::isfinite(spot[i]) && std::isfinite(strike[i]) && std::isfinite(tau[i]) &&
                  std::isfinite(r[i]) && std::isfinite(q[i]) &&
                  std::isfinite(v0[i]) && std::isfinite(kappa[i]) && std::isfinite(theta[i]) &&
                  std::isfinite(sigma[i]) && std::isfinite(rho[i])) ||
                spot[i] <= 0.0 || strike[i] <= 0.0 || tau[i] <= 0.0 ||
                v0[i] < 0.0 || kappa[i] <= 0.0 || theta[i] < 0.0 || sigma[i] <= 0.0 ||
                rho[i] <= -1.0 || rho[i] >= 1.0 || !valid_option_type(option_type[i])) {
                out[i] = nan;
                continue;
            }
            out[i] = ql_heston_price(
                spot[i], strike[i], tau[i], r[i], q[i], v0[i], kappa[i], theta[i], sigma[i], rho[i],
                option_type[i], integration_steps[i]
            );
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

extern "C" int32_t ql_ref_merton_jump_diffusion_price_batch(
    const double* spot,
    const double* strike,
    const double* tau,
    const double* vol,
    const double* r,
    const double* q,
    const double* jump_intensity,
    const double* jump_mean,
    const double* jump_vol,
    const int32_t* max_terms,
    const int32_t* option_type,
    int32_t n,
    double* out,
    char* err_buf,
    int32_t err_buf_size) {
    if (spot == nullptr || strike == nullptr || tau == nullptr || vol == nullptr ||
        r == nullptr || q == nullptr || jump_intensity == nullptr || jump_mean == nullptr ||
        jump_vol == nullptr || max_terms == nullptr || option_type == nullptr || out == nullptr) {
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
            if (!(std::isfinite(spot[i]) && std::isfinite(strike[i]) && std::isfinite(tau[i]) &&
                  std::isfinite(vol[i]) && std::isfinite(r[i]) && std::isfinite(q[i]) &&
                  std::isfinite(jump_intensity[i]) && std::isfinite(jump_mean[i]) &&
                  std::isfinite(jump_vol[i])) ||
                spot[i] <= 0.0 || strike[i] <= 0.0 || tau[i] <= 0.0 || vol[i] < 0.0 ||
                jump_intensity[i] < 0.0 || jump_vol[i] < 0.0 || max_terms[i] < 1 ||
                !valid_option_type(option_type[i])) {
                out[i] = nan;
                continue;
            }
            out[i] = ql_merton_price(
                spot[i], strike[i], tau[i], vol[i], r[i], q[i],
                jump_intensity[i], jump_mean[i], jump_vol[i], max_terms[i], option_type[i]
            );
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

extern "C" int32_t ql_ref_variance_gamma_price_batch(
    const double* spot,
    const double* strike,
    const double* tau,
    const double* r,
    const double* q,
    const double* sigma,
    const double* theta,
    const double* nu,
    const int32_t* option_type,
    int32_t n,
    double* out,
    char* err_buf,
    int32_t err_buf_size) {
    if (spot == nullptr || strike == nullptr || tau == nullptr || r == nullptr || q == nullptr ||
        sigma == nullptr || theta == nullptr || nu == nullptr || option_type == nullptr || out == nullptr) {
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
            if (!(std::isfinite(spot[i]) && std::isfinite(strike[i]) && std::isfinite(tau[i]) &&
                  std::isfinite(r[i]) && std::isfinite(q[i]) &&
                  std::isfinite(sigma[i]) && std::isfinite(theta[i]) && std::isfinite(nu[i])) ||
                spot[i] <= 0.0 || strike[i] <= 0.0 || tau[i] <= 0.0 || sigma[i] <= 0.0 ||
                nu[i] <= 0.0 || !valid_option_type(option_type[i])) {
                out[i] = nan;
                continue;
            }
            out[i] = ql_vg_price(
                spot[i], strike[i], tau[i], r[i], q[i], sigma[i], theta[i], nu[i], option_type[i]
            );
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

extern "C" int32_t ql_ref_sabr_lognormal_iv_batch(
    const double* forward,
    const double* strike,
    const double* tau,
    const double* alpha,
    const double* beta,
    const double* rho,
    const double* nu,
    int32_t n,
    double* out,
    char* err_buf,
    int32_t err_buf_size) {
    if (forward == nullptr || strike == nullptr || tau == nullptr || alpha == nullptr ||
        beta == nullptr || rho == nullptr || nu == nullptr || out == nullptr) {
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
            if (!(std::isfinite(forward[i]) && std::isfinite(strike[i]) && std::isfinite(tau[i]) &&
                  std::isfinite(alpha[i]) && std::isfinite(beta[i]) &&
                  std::isfinite(rho[i]) && std::isfinite(nu[i])) ||
                forward[i] <= 0.0 || strike[i] <= 0.0 || tau[i] <= 0.0 || alpha[i] <= 0.0 ||
                beta[i] < 0.0 || beta[i] > 1.0 || rho[i] <= -1.0 || rho[i] >= 1.0 || nu[i] < 0.0) {
                out[i] = nan;
                continue;
            }
            out[i] = ql_sabr_iv(forward[i], strike[i], tau[i], alpha[i], beta[i], rho[i], nu[i]);
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

extern "C" int32_t ql_ref_sabr_black76_price_batch(
    const double* forward,
    const double* strike,
    const double* tau,
    const double* r,
    const double* alpha,
    const double* beta,
    const double* rho,
    const double* nu,
    const int32_t* option_type,
    int32_t n,
    double* out,
    char* err_buf,
    int32_t err_buf_size) {
    if (forward == nullptr || strike == nullptr || tau == nullptr || r == nullptr ||
        alpha == nullptr || beta == nullptr || rho == nullptr || nu == nullptr ||
        option_type == nullptr || out == nullptr) {
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
            if (!(std::isfinite(forward[i]) && std::isfinite(strike[i]) && std::isfinite(tau[i]) &&
                  std::isfinite(r[i]) && std::isfinite(alpha[i]) && std::isfinite(beta[i]) &&
                  std::isfinite(rho[i]) && std::isfinite(nu[i])) ||
                forward[i] <= 0.0 || strike[i] <= 0.0 || tau[i] <= 0.0 || alpha[i] <= 0.0 ||
                beta[i] < 0.0 || beta[i] > 1.0 || rho[i] <= -1.0 || rho[i] >= 1.0 || nu[i] < 0.0 ||
                !valid_option_type(option_type[i])) {
                out[i] = nan;
                continue;
            }
            out[i] = ql_sabr_black76_price(
                forward[i], strike[i], tau[i], r[i], alpha[i], beta[i], rho[i], nu[i], option_type[i]
            );
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
