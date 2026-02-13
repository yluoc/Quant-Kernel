"""ctypes struct mirrors of qk_abi.h — must stay in sync with the C header."""

import ctypes as ct

# ABI version expected by this Python wrapper
ABI_MAJOR = 1
ABI_MINOR = 1

# Batch-level return codes
QK_OK = 0
QK_ERR_NULL_PTR = -1
QK_ERR_BAD_SIZE = -2
QK_ERR_ABI_MISMATCH = -3
QK_ERR_RUNTIME_INIT = -4

# Per-row error codes
QK_ROW_OK = 0
QK_ROW_ERR_NEGATIVE_S = 1
QK_ROW_ERR_NEGATIVE_K = 2
QK_ROW_ERR_NEGATIVE_T = 3
QK_ROW_ERR_NEGATIVE_V = 4
QK_ROW_ERR_BAD_TYPE = 5
QK_ROW_ERR_IV_NO_CONV = 6
QK_ROW_ERR_BAD_PRICE = 7
QK_ROW_ERR_NON_FINITE = 8
QK_ROW_ERR_BAD_PATHS = 9

# Option types
QK_CALL = 0
QK_PUT = 1


class QKBSInput(ct.Structure):
    _fields_ = [
        ("n", ct.c_int64),
        ("spot", ct.POINTER(ct.c_double)),
        ("strike", ct.POINTER(ct.c_double)),
        ("time_to_expiry", ct.POINTER(ct.c_double)),
        ("volatility", ct.POINTER(ct.c_double)),
        ("risk_free_rate", ct.POINTER(ct.c_double)),
        ("dividend_yield", ct.POINTER(ct.c_double)),
        ("option_type", ct.POINTER(ct.c_int32)),
    ]


class QKBSOutput(ct.Structure):
    _fields_ = [
        ("price", ct.POINTER(ct.c_double)),
        ("delta", ct.POINTER(ct.c_double)),
        ("gamma", ct.POINTER(ct.c_double)),
        ("vega", ct.POINTER(ct.c_double)),
        ("theta", ct.POINTER(ct.c_double)),
        ("rho", ct.POINTER(ct.c_double)),
        ("error_codes", ct.POINTER(ct.c_int32)),
    ]


class QKIVInput(ct.Structure):
    _fields_ = [
        ("n", ct.c_int64),
        ("spot", ct.POINTER(ct.c_double)),
        ("strike", ct.POINTER(ct.c_double)),
        ("time_to_expiry", ct.POINTER(ct.c_double)),
        ("risk_free_rate", ct.POINTER(ct.c_double)),
        ("dividend_yield", ct.POINTER(ct.c_double)),
        ("option_type", ct.POINTER(ct.c_int32)),
        ("market_price", ct.POINTER(ct.c_double)),
        ("tol", ct.c_double),
        ("max_iter", ct.c_int32),
    ]


class QKIVOutput(ct.Structure):
    _fields_ = [
        ("implied_vol", ct.POINTER(ct.c_double)),
        ("iterations", ct.POINTER(ct.c_int32)),
        ("error_codes", ct.POINTER(ct.c_int32)),
    ]


class QKMCInput(ct.Structure):
    _fields_ = [
        ("n", ct.c_int64),
        ("spot", ct.POINTER(ct.c_double)),
        ("strike", ct.POINTER(ct.c_double)),
        ("time_to_expiry", ct.POINTER(ct.c_double)),
        ("volatility", ct.POINTER(ct.c_double)),
        ("risk_free_rate", ct.POINTER(ct.c_double)),
        ("dividend_yield", ct.POINTER(ct.c_double)),
        ("option_type", ct.POINTER(ct.c_int32)),
        ("num_paths", ct.POINTER(ct.c_int32)),
        ("rng_seed", ct.POINTER(ct.c_uint64)),
    ]


class QKMCOutput(ct.Structure):
    _fields_ = [
        ("price", ct.POINTER(ct.c_double)),
        ("std_error", ct.POINTER(ct.c_double)),
        ("paths_used", ct.POINTER(ct.c_int32)),
        ("error_codes", ct.POINTER(ct.c_int32)),
    ]
