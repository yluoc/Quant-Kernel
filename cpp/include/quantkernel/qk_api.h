#ifndef QK_API_H
#define QK_API_H

#include "qk_abi.h"

#ifdef _WIN32
  #define QK_EXPORT __declspec(dllexport)
#else
  #define QK_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

QK_EXPORT void    qk_abi_version(int32_t* major, int32_t* minor);
QK_EXPORT int32_t qk_bs_price(const QKBSInput* input, QKBSOutput* output);
QK_EXPORT int32_t qk_iv_solve(const QKIVInput* input, QKIVOutput* output);
QK_EXPORT int32_t qk_mc_price(const QKMCInput* input, QKMCOutput* output);
QK_EXPORT int32_t qk_plugin_get_api(int32_t host_abi_major,
                                    int32_t host_abi_minor,
                                    const QKPluginAPI** out_api);
/* Runtime-only symbols (exported by Rust runtime shell, not by plugins) */
QK_EXPORT int32_t qk_runtime_load_plugin(const char* plugin_path_utf8);
QK_EXPORT int32_t qk_runtime_unload_plugin(void);

#ifdef __cplusplus
}
#endif

#endif /* QK_API_H */
