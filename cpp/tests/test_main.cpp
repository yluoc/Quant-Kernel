#include <cmath>
#include <cstdio>
#include <cstdlib>

/* Minimal test harness — no external framework needed */

static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name)                                                       \
    static void test_##name();                                           \
    static struct Register_##name {                                      \
        Register_##name() { test_registry_add(#name, test_##name); }     \
    } reg_##name;                                                        \
    static void test_##name()

#define ASSERT_NEAR(a, b, tol)                                           \
    do {                                                                 \
        double _a = (a), _b = (b), _t = (tol);                          \
        if (std::fabs(_a - _b) > _t) {                                  \
            std::fprintf(stderr,                                         \
                "  FAIL %s:%d: |%.10g - %.10g| = %.10g > %.10g\n",      \
                __FILE__, __LINE__, _a, _b, std::fabs(_a - _b), _t);    \
            throw 1;                                                     \
        }                                                                \
    } while (0)

#define ASSERT_EQ(a, b)                                                  \
    do {                                                                 \
        auto _a = (a); auto _b = (b);                                    \
        if (_a != _b) {                                                  \
            std::fprintf(stderr,                                         \
                "  FAIL %s:%d: %lld != %lld\n",                          \
                __FILE__, __LINE__,                                      \
                (long long)_a, (long long)_b);                           \
            throw 1;                                                     \
        }                                                                \
    } while (0)

#define ASSERT_TRUE(cond)                                                \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::fprintf(stderr,                                         \
                "  FAIL %s:%d: condition false: %s\n",                   \
                __FILE__, __LINE__, #cond);                              \
            throw 1;                                                     \
        }                                                                \
    } while (0)

struct TestEntry {
    const char* name;
    void (*func)();
};

constexpr int MAX_TESTS = 128;
TestEntry g_tests[MAX_TESTS];
int g_test_count = 0;

void test_registry_add(const char* name, void (*func)()) {
    if (g_test_count < MAX_TESTS) {
        g_tests[g_test_count++] = {name, func};
    }
}

/* Declare tests from other TUs */
extern void register_cf_tests();
extern void register_tlm_tests();

int main() {
    register_cf_tests();
    register_tlm_tests();

    std::printf("Running %d tests...\n", g_test_count);
    for (int i = 0; i < g_test_count; ++i) {
        ++g_tests_run;
        std::printf("[%d/%d] %s ... ", i + 1, g_test_count, g_tests[i].name);
        try {
            g_tests[i].func();
            std::printf("OK\n");
            ++g_tests_passed;
        } catch (...) {
            std::printf("FAILED\n");
            ++g_tests_failed;
        }
    }

    std::printf("\n%d passed, %d failed, %d total\n",
                g_tests_passed, g_tests_failed, g_tests_run);
    return g_tests_failed > 0 ? 1 : 0;
}
