#include <quantkernel/qk_api.h>
#include "algorithms/black_scholes/bs_pricer.h"
#include "algorithms/implied_vol/iv_solver.h"

namespace {

const QKPluginAPI k_plugin_api = {
    QK_ABI_MAJOR,
    QK_ABI_MINOR,
    "quantkernel.cpp.bs_iv.v1",
    qk_bs_price,
    qk_iv_solve
};

} /* namespace */

extern "C" {

void qk_abi_version(int32_t* major, int32_t* minor) {
    if (major) *major = QK_ABI_MAJOR;
    if (minor) *minor = QK_ABI_MINOR;
}

int32_t qk_bs_price(const QKBSInput* input, QKBSOutput* output) {
    if (!input || !output) return QK_ERR_NULL_PTR;
    if (input->n <= 0)     return QK_ERR_BAD_SIZE;

    /* Check all required pointers */
    if (!input->spot || !input->strike || !input->time_to_expiry ||
        !input->volatility || !input->risk_free_rate ||
        !input->dividend_yield || !input->option_type)
        return QK_ERR_NULL_PTR;

    if (!output->price || !output->delta || !output->gamma ||
        !output->vega || !output->theta || !output->rho ||
        !output->error_codes)
        return QK_ERR_NULL_PTR;

    qk::bs_price_batch(*input, *output);
    return QK_OK;
}

int32_t qk_iv_solve(const QKIVInput* input, QKIVOutput* output) {
    if (!input || !output) return QK_ERR_NULL_PTR;
    if (input->n <= 0)     return QK_ERR_BAD_SIZE;

    if (!input->spot || !input->strike || !input->time_to_expiry ||
        !input->risk_free_rate || !input->dividend_yield ||
        !input->option_type || !input->market_price)
        return QK_ERR_NULL_PTR;

    if (!output->implied_vol || !output->iterations || !output->error_codes)
        return QK_ERR_NULL_PTR;

    qk::iv_solve_batch(*input, *output);
    return QK_OK;
}

int32_t qk_plugin_get_api(int32_t host_abi_major,
                          int32_t host_abi_minor,
                          const QKPluginAPI** out_api) {
    if (!out_api) return QK_ERR_NULL_PTR;
    if (host_abi_major != QK_ABI_MAJOR || host_abi_minor < QK_ABI_MINOR) {
        *out_api = nullptr;
        return QK_ERR_ABI_MISMATCH;
    }
    *out_api = &k_plugin_api;
    return QK_OK;
}

} /* extern "C" */
