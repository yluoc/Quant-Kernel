#ifndef QK_TEST_HARNESS_H
#define QK_TEST_HARNESS_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

struct QKTestCase {
    const char* name;
    void (*func)();
};

static std::vector<QKTestCase>& qk_test_registry() {
    static std::vector<QKTestCase> tests;
    return tests;
}

struct QKTestRegistrar {
    QKTestRegistrar(const char* name, void (*func)()) {
        qk_test_registry().push_back({name, func});
    }
};

static int qk_tests_failed = 0;
static int qk_tests_passed = 0;

#define QK_TEST(name) \
    static void qk_test_##name(); \
    static QKTestRegistrar qk_reg_##name(#name, qk_test_##name); \
    static void qk_test_##name()

#define QK_ASSERT_NEAR(a, b, tol) \
    do { \
        double _a = (a), _b = (b), _t = (tol); \
        if (std::isnan(_a) || std::isnan(_b) || std::abs(_a - _b) > _t) { \
            std::fprintf(stderr, "  FAIL %s:%d: |%.12g - %.12g| = %.12g > %.12g\n", \
                         __FILE__, __LINE__, _a, _b, std::abs(_a - _b), _t); \
            ++qk_tests_failed; return; \
        } \
    } while(0)

#define QK_ASSERT_EQ(a, b) \
    do { \
        auto _a = (a); auto _b = (b); \
        if (_a != _b) { \
            std::fprintf(stderr, "  FAIL %s:%d: %d != %d\n", \
                         __FILE__, __LINE__, (int)_a, (int)_b); \
            ++qk_tests_failed; return; \
        } \
    } while(0)

#define QK_ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "  FAIL %s:%d: condition false\n", \
                         __FILE__, __LINE__); \
            ++qk_tests_failed; return; \
        } \
    } while(0)

#define QK_ASSERT_NAN(val) \
    do { \
        double _v = (val); \
        if (!std::isnan(_v)) { \
            std::fprintf(stderr, "  FAIL %s:%d: expected NaN, got %.12g\n", \
                         __FILE__, __LINE__, _v); \
            ++qk_tests_failed; return; \
        } \
    } while(0)

static int qk_run_all_tests() {
    auto& tests = qk_test_registry();
    std::printf("Running %zu tests...\n", tests.size());
    for (auto& tc : tests) {
        int before = qk_tests_failed;
        tc.func();
        if (qk_tests_failed == before) {
            ++qk_tests_passed;
            std::printf("  PASS %s\n", tc.name);
        } else {
            std::printf("  FAIL %s\n", tc.name);
        }
    }
    std::printf("\n%d passed, %d failed\n", qk_tests_passed, qk_tests_failed);
    return qk_tests_failed > 0 ? 1 : 0;
}

#define QK_TEST_MAIN() int main() { return qk_run_all_tests(); }

#endif
