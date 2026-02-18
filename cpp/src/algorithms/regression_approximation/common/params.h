#ifndef QK_RAM_PARAMS_H
#define QK_RAM_PARAMS_H

#include <cstdint>

namespace qk::ram {

struct PolynomialChaosExpansionParams {
    int32_t polynomial_order;
    int32_t quadrature_points;
};

struct RadialBasisFunctionParams {
    int32_t centers;
    double rbf_shape;
    double ridge;
};

struct SparseGridCollocationParams {
    int32_t level;
    int32_t nodes_per_dim;
};

struct ProperOrthogonalDecompositionParams {
    int32_t modes;
    int32_t snapshots;
};

} // namespace qk::ram

#endif /* QK_RAM_PARAMS_H */
