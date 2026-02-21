#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>

#include <ql/exercise.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/pricingengines/vanilla/fdblackscholesvanillaengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
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

double american_fd_price(
    double spot,
    double strike,
    double tau,
    double vol,
    double r,
    double q,
    int32_t option_type,
    int32_t t_grid,
    int32_t x_grid
) {
    const QuantLib::Option::Type ql_type =
        (option_type == 0) ? QuantLib::Option::Call : QuantLib::Option::Put;

    const QuantLib::Date eval_date(15, QuantLib::January, 2024);
    const int maturity_days = std::max(1, static_cast<int>(std::llround(365.0 * tau)));
    const QuantLib::Date maturity_date = eval_date + maturity_days;

    QuantLib::Settings::instance().evaluationDate() = eval_date;
    const QuantLib::DayCounter dc = QuantLib::Actual365Fixed();
    const QuantLib::Calendar cal = QuantLib::NullCalendar();

    auto spot_q = QuantLib::ext::make_shared<QuantLib::SimpleQuote>(spot);
    auto risk_ts = QuantLib::ext::make_shared<QuantLib::FlatForward>(eval_date, r, dc);
    auto div_ts = QuantLib::ext::make_shared<QuantLib::FlatForward>(eval_date, q, dc);
    auto vol_ts = QuantLib::ext::make_shared<QuantLib::BlackConstantVol>(eval_date, cal, vol, dc);

    auto process = QuantLib::ext::make_shared<QuantLib::BlackScholesMertonProcess>(
        QuantLib::Handle<QuantLib::Quote>(spot_q),
        QuantLib::Handle<QuantLib::YieldTermStructure>(div_ts),
        QuantLib::Handle<QuantLib::YieldTermStructure>(risk_ts),
        QuantLib::Handle<QuantLib::BlackVolTermStructure>(vol_ts)
    );

    auto payoff = QuantLib::ext::make_shared<QuantLib::PlainVanillaPayoff>(ql_type, strike);
    auto exercise = QuantLib::ext::make_shared<QuantLib::AmericanExercise>(eval_date, maturity_date);
    QuantLib::VanillaOption option(payoff, exercise);

    auto engine = QuantLib::ext::make_shared<QuantLib::FdBlackScholesVanillaEngine>(
        process, t_grid, x_grid, 0
    );
    option.setPricingEngine(engine);
    return option.NPV();
}

}  // namespace

extern "C" int32_t ql_ref_american_price_batch(
    const double* spot,
    const double* strike,
    const double* tau,
    const double* vol,
    const double* r,
    const double* q,
    const int32_t* option_type,
    int32_t n,
    int32_t t_grid,
    int32_t x_grid,
    double* out,
    char* err_buf,
    int32_t err_buf_size) {
    if (spot == nullptr || strike == nullptr || tau == nullptr || vol == nullptr ||
        r == nullptr || q == nullptr || option_type == nullptr || out == nullptr) {
        write_error(err_buf, err_buf_size, "Null input pointer.");
        return 1;
    }
    if (n < 0 || t_grid < 20 || x_grid < 20) {
        write_error(err_buf, err_buf_size, "Invalid batch size or FD grid settings.");
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

            if (t <= 1e-12 || sigma <= 1e-12) {
                const double fwd_intrinsic = (ot == 0)
                    ? std::max(s - k, 0.0)
                    : std::max(k - s, 0.0);
                out[i] = fwd_intrinsic;
                continue;
            }

            out[i] = american_fd_price(s, k, t, sigma, rf, div, ot, t_grid, x_grid);
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

