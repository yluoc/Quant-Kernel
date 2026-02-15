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
QK_EXPORT int32_t qk_plugin_get_api(int32_t host_abi_major,
                                    int32_t host_abi_minor,
                                    const QKPluginAPI** out_api);

#ifdef __cplusplus
}
#endif

#endif /* QK_API_H */
