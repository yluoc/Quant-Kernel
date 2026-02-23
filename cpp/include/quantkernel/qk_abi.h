#ifndef QK_ABI_H
#define QK_ABI_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

/* ABI version — bump major on breaking changes, minor on additions */
#define QK_ABI_MAJOR 2
#define QK_ABI_MINOR 11

/* Return codes */
#define QK_OK             0
#define QK_ERR_NULL_PTR  -1
#define QK_ERR_BAD_SIZE  -2
#define QK_ERR_ABI_MISMATCH -3
#define QK_ERR_RUNTIME_INIT -4
#define QK_ERR_INVALID_INPUT -5

/* Option type constants */
#define QK_CALL 0
#define QK_PUT  1


typedef struct QKPluginAPI {
    int32_t     abi_major;
    int32_t     abi_minor;
    const char* plugin_name;
} QKPluginAPI;

#endif /* QK_ABI_H */
