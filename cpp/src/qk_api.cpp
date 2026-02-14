#include <quantkernel/qk_api.h>

namespace {

const QKPluginAPI k_plugin_api = {
    QK_ABI_MAJOR,
    QK_ABI_MINOR,
    "quantkernel.cpp.closed_form.v2"
};

} /* namespace */

extern "C" {

void qk_abi_version(int32_t* major, int32_t* minor) {
    if (major) *major = QK_ABI_MAJOR;
    if (minor) *minor = QK_ABI_MINOR;
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
