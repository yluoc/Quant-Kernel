#ifndef QK_ABI_H
#define QK_ABI_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

/* ABI version — bump major on breaking changes, minor on additions */
#define QK_ABI_MAJOR 1
#define QK_ABI_MINOR 1

/* Batch-level return codes (returned by qk_bs_price / qk_iv_solve) */
#define QK_OK             0
#define QK_ERR_NULL_PTR  -1
#define QK_ERR_BAD_SIZE  -2
#define QK_ERR_ABI_MISMATCH -3
#define QK_ERR_RUNTIME_INIT -4

/* Per-row error codes (written into error_codes[]) */
#define QK_ROW_OK              0
#define QK_ROW_ERR_NEGATIVE_S  1   /* spot <= 0 */
#define QK_ROW_ERR_NEGATIVE_K  2   /* strike <= 0 */
#define QK_ROW_ERR_NEGATIVE_T  3   /* time_to_expiry <= 0 */
#define QK_ROW_ERR_NEGATIVE_V  4   /* volatility <= 0 */
#define QK_ROW_ERR_BAD_TYPE    5   /* option_type not 0 (call) or 1 (put) */
#define QK_ROW_ERR_IV_NO_CONV  6   /* IV solver did not converge */
#define QK_ROW_ERR_BAD_PRICE   7   /* market_price <= 0 or exceeds bound */
#define QK_ROW_ERR_NON_FINITE  8   /* input contains NaN/Inf */
#define QK_ROW_ERR_BAD_PATHS   9   /* num_paths <= 0 */

/* Option type constants */
#define QK_CALL 0
#define QK_PUT  1

/* --- Columnar (SoA) structs --- */

typedef struct QKBSInput {
    int64_t          n;
    const double*    spot;
    const double*    strike;
    const double*    time_to_expiry;
    const double*    volatility;
    const double*    risk_free_rate;
    const double*    dividend_yield;
    const int32_t*   option_type;       /* QK_CALL=0, QK_PUT=1 */
} QKBSInput;

typedef struct QKBSOutput {
    double*    price;
    double*    delta;
    double*    gamma;
    double*    vega;
    double*    theta;
    double*    rho;
    int32_t*   error_codes;
} QKBSOutput;

typedef struct QKIVInput {
    int64_t          n;
    const double*    spot;
    const double*    strike;
    const double*    time_to_expiry;
    const double*    risk_free_rate;
    const double*    dividend_yield;
    const int32_t*   option_type;
    const double*    market_price;
    double           tol;
    int32_t          max_iter;
} QKIVInput;

typedef struct QKIVOutput {
    double*    implied_vol;
    int32_t*   iterations;
    int32_t*   error_codes;
} QKIVOutput;

typedef struct QKMCInput {
    int64_t            n;
    const double*      spot;
    const double*      strike;
    const double*      time_to_expiry;
    const double*      volatility;
    const double*      risk_free_rate;
    const double*      dividend_yield;
    const int32_t*     option_type;
    const int32_t*     num_paths;
    const uint64_t*    rng_seed;
} QKMCInput;

typedef struct QKMCOutput {
    double*    price;
    double*    std_error;
    int32_t*   paths_used;
    int32_t*   error_codes;
} QKMCOutput;

/* --- Plugin ABI surface --- */

typedef int32_t (*QKBSPriceFn)(const QKBSInput* input, QKBSOutput* output);
typedef int32_t (*QKIVSolveFn)(const QKIVInput* input, QKIVOutput* output);
typedef int32_t (*QKMCPriceFn)(const QKMCInput* input, QKMCOutput* output);

typedef struct QKPluginAPI {
    int32_t     abi_major;
    int32_t     abi_minor;
    const char* plugin_name;
    QKBSPriceFn bs_price;
    QKIVSolveFn iv_solve;
    QKMCPriceFn mc_price;
} QKPluginAPI;

#endif /* QK_ABI_H */
