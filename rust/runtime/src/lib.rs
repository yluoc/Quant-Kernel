use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::{OnceLock, RwLock};

const QK_ABI_MAJOR: i32 = 2;
const QK_ABI_MINOR: i32 = 0;

const QK_OK: i32 = 0;
const QK_ERR_NULL_PTR: i32 = -1;
const QK_ERR_BAD_SIZE: i32 = -2;
const QK_ERR_ABI_MISMATCH: i32 = -3;
const QK_ERR_RUNTIME_INIT: i32 = -4;

type QKPluginGetApiFn = unsafe extern "C" fn(i32, i32, *mut *const QKPluginAPI) -> i32;

#[repr(C)]
pub struct QKPluginAPI {
    abi_major: i32,
    abi_minor: i32,
    plugin_name: *const c_char,
}

struct PluginHandle {
    dl_handle: *mut c_void,
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
    if api.plugin_name.is_null() {
        close_plugin(dl_handle);
        return Err(QK_ERR_RUNTIME_INIT);
    }
    Ok(PluginHandle { dl_handle })
}

#[cfg(not(unix))]
unsafe fn load_plugin_internal(_path: *const c_char) -> Result<PluginHandle, i32> {
    Err(QK_ERR_RUNTIME_INIT)
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
