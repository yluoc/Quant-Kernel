use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::{OnceLock, RwLock};

const QK_ABI_MAJOR: i32 = 1;
const QK_ABI_MINOR: i32 = 0;

const QK_OK: i32 = 0;
const QK_ERR_NULL_PTR: i32 = -1;
const QK_ERR_BAD_SIZE: i32 = -2;
const QK_ERR_ABI_MISMATCH: i32 = -3;
const QK_ERR_RUNTIME_INIT: i32 = -4;

#[repr(C)]
pub struct QKBSInput {
    n: i64,
    spot: *const f64,
    strike: *const f64,
    time_to_expiry: *const f64,
    volatility: *const f64,
    risk_free_rate: *const f64,
    dividend_yield: *const f64,
    option_type: *const i32,
}

#[repr(C)]
pub struct QKBSOutput {
    price: *mut f64,
    delta: *mut f64,
    gamma: *mut f64,
    vega: *mut f64,
    theta: *mut f64,
    rho: *mut f64,
    error_codes: *mut i32,
}

#[repr(C)]
pub struct QKIVInput {
    n: i64,
    spot: *const f64,
    strike: *const f64,
    time_to_expiry: *const f64,
    risk_free_rate: *const f64,
    dividend_yield: *const f64,
    option_type: *const i32,
    market_price: *const f64,
    tol: f64,
    max_iter: i32,
}

#[repr(C)]
pub struct QKIVOutput {
    implied_vol: *mut f64,
    iterations: *mut i32,
    error_codes: *mut i32,
}

type QKBSPriceFn = unsafe extern "C" fn(*const QKBSInput, *mut QKBSOutput) -> i32;
type QKIVSolveFn = unsafe extern "C" fn(*const QKIVInput, *mut QKIVOutput) -> i32;
type QKPluginGetApiFn = unsafe extern "C" fn(i32, i32, *mut *const QKPluginAPI) -> i32;

#[repr(C)]
pub struct QKPluginAPI {
    abi_major: i32,
    abi_minor: i32,
    plugin_name: *const c_char,
    bs_price: Option<QKBSPriceFn>,
    iv_solve: Option<QKIVSolveFn>,
}

struct PluginHandle {
    dl_handle: *mut c_void,
    api: *const QKPluginAPI,
}

unsafe impl Send for PluginHandle {}
unsafe impl Sync for PluginHandle {}

static RUNTIME_STATE: OnceLock<RwLock<Option<PluginHandle>>> = OnceLock::new();

fn runtime_state() -> &'static RwLock<Option<PluginHandle>> {
    RUNTIME_STATE.get_or_init(|| RwLock::new(None))
}

#[cfg(unix)]
const RTLD_NOW: c_int = 2;

#[cfg(unix)]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

#[cfg(unix)]
unsafe fn close_plugin(handle: *mut c_void) {
    if !handle.is_null() {
        let _ = dlclose(handle);
    }
}

#[cfg(unix)]
unsafe fn load_plugin_internal(path: *const c_char) -> Result<PluginHandle, i32> {
    let dl_handle = dlopen(path, RTLD_NOW);
    if dl_handle.is_null() {
        return Err(QK_ERR_RUNTIME_INIT);
    }

    let sym = dlsym(dl_handle, b"qk_plugin_get_api\0".as_ptr().cast());
    if sym.is_null() {
        close_plugin(dl_handle);
        return Err(QK_ERR_RUNTIME_INIT);
    }

    let get_api: QKPluginGetApiFn = std::mem::transmute(sym);
    let mut api_ptr: *const QKPluginAPI = ptr::null();
    let rc = get_api(QK_ABI_MAJOR, QK_ABI_MINOR, &mut api_ptr as *mut _);
    if rc != QK_OK {
        close_plugin(dl_handle);
        return Err(rc);
    }
    if api_ptr.is_null() {
        close_plugin(dl_handle);
        return Err(QK_ERR_RUNTIME_INIT);
    }

    let api = &*api_ptr;
    if api.abi_major != QK_ABI_MAJOR || api.abi_minor < QK_ABI_MINOR {
        close_plugin(dl_handle);
        return Err(QK_ERR_ABI_MISMATCH);
    }
    if api.bs_price.is_none() || api.iv_solve.is_none() {
        close_plugin(dl_handle);
        return Err(QK_ERR_RUNTIME_INIT);
    }

    Ok(PluginHandle { dl_handle, api: api_ptr })
}

#[cfg(not(unix))]
unsafe fn load_plugin_internal(_path: *const c_char) -> Result<PluginHandle, i32> {
    Err(QK_ERR_RUNTIME_INIT)
}

fn with_plugin<F>(f: F) -> i32
where
    F: FnOnce(&QKPluginAPI) -> i32,
{
    let state = runtime_state();
    let guard = match state.read() {
        Ok(g) => g,
        Err(_) => return QK_ERR_RUNTIME_INIT,
    };

    let Some(plugin) = guard.as_ref() else {
        return QK_ERR_RUNTIME_INIT;
    };

    let api = unsafe { plugin.api.as_ref() };
    let Some(api) = api else {
        return QK_ERR_RUNTIME_INIT;
    };

    f(api)
}

fn ffi_guard<F>(f: F) -> i32
where
    F: FnOnce() -> i32,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(rc) => rc,
        Err(_) => QK_ERR_RUNTIME_INIT,
    }
}

fn validate_bs(input: *const QKBSInput, output: *mut QKBSOutput) -> i32 {
    if input.is_null() || output.is_null() {
        return QK_ERR_NULL_PTR;
    }

    let in_ref = unsafe { &*input };
    let out_ref = unsafe { &*output };

    if in_ref.n <= 0 {
        return QK_ERR_BAD_SIZE;
    }

    if in_ref.spot.is_null()
        || in_ref.strike.is_null()
        || in_ref.time_to_expiry.is_null()
        || in_ref.volatility.is_null()
        || in_ref.risk_free_rate.is_null()
        || in_ref.dividend_yield.is_null()
        || in_ref.option_type.is_null()
    {
        return QK_ERR_NULL_PTR;
    }

    if out_ref.price.is_null()
        || out_ref.delta.is_null()
        || out_ref.gamma.is_null()
        || out_ref.vega.is_null()
        || out_ref.theta.is_null()
        || out_ref.rho.is_null()
        || out_ref.error_codes.is_null()
    {
        return QK_ERR_NULL_PTR;
    }

    QK_OK
}

fn validate_iv(input: *const QKIVInput, output: *mut QKIVOutput) -> i32 {
    if input.is_null() || output.is_null() {
        return QK_ERR_NULL_PTR;
    }

    let in_ref = unsafe { &*input };
    let out_ref = unsafe { &*output };

    if in_ref.n <= 0 {
        return QK_ERR_BAD_SIZE;
    }

    if in_ref.spot.is_null()
        || in_ref.strike.is_null()
        || in_ref.time_to_expiry.is_null()
        || in_ref.risk_free_rate.is_null()
        || in_ref.dividend_yield.is_null()
        || in_ref.option_type.is_null()
        || in_ref.market_price.is_null()
    {
        return QK_ERR_NULL_PTR;
    }

    if out_ref.implied_vol.is_null() || out_ref.iterations.is_null() || out_ref.error_codes.is_null() {
        return QK_ERR_NULL_PTR;
    }

    QK_OK
}

#[no_mangle]
pub extern "C" fn qk_abi_version(major: *mut i32, minor: *mut i32) {
    if !major.is_null() {
        unsafe { *major = QK_ABI_MAJOR };
    }
    if !minor.is_null() {
        unsafe { *minor = QK_ABI_MINOR };
    }
}

#[no_mangle]
pub extern "C" fn qk_runtime_load_plugin(path: *const c_char) -> i32 {
    ffi_guard(|| {
        if path.is_null() {
            return QK_ERR_NULL_PTR;
        }
        if unsafe { *path } == 0 {
            return QK_ERR_BAD_SIZE;
        }

        let loaded = unsafe { load_plugin_internal(path) };
        let loaded = match loaded {
            Ok(plugin) => plugin,
            Err(rc) => return rc,
        };

        let state = runtime_state();
        let mut guard = match state.write() {
            Ok(g) => g,
            Err(_) => return QK_ERR_RUNTIME_INIT,
        };

        #[cfg(unix)]
        if let Some(previous) = guard.take() {
            unsafe { close_plugin(previous.dl_handle) };
        }

        #[cfg(not(unix))]
        {
            guard.take();
        }

        *guard = Some(loaded);
        QK_OK
    })
}

#[no_mangle]
pub extern "C" fn qk_runtime_unload_plugin() -> i32 {
    ffi_guard(|| {
        let state = runtime_state();
        let mut guard = match state.write() {
            Ok(g) => g,
            Err(_) => return QK_ERR_RUNTIME_INIT,
        };

        #[cfg(unix)]
        if let Some(previous) = guard.take() {
            unsafe { close_plugin(previous.dl_handle) };
        }

        #[cfg(not(unix))]
        {
            guard.take();
        }

        QK_OK
    })
}

#[no_mangle]
pub extern "C" fn qk_bs_price(input: *const QKBSInput, output: *mut QKBSOutput) -> i32 {
    ffi_guard(|| {
        let rc = validate_bs(input, output);
        if rc != QK_OK {
            return rc;
        }

        with_plugin(|api| {
            let Some(bs_price) = api.bs_price else {
                return QK_ERR_RUNTIME_INIT;
            };
            unsafe { bs_price(input, output) }
        })
    })
}

#[no_mangle]
pub extern "C" fn qk_iv_solve(input: *const QKIVInput, output: *mut QKIVOutput) -> i32 {
    ffi_guard(|| {
        let rc = validate_iv(input, output);
        if rc != QK_OK {
            return rc;
        }

        with_plugin(|api| {
            let Some(iv_solve) = api.iv_solve else {
                return QK_ERR_RUNTIME_INIT;
            };
            unsafe { iv_solve(input, output) }
        })
    })
}
