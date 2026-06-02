#include "exprtk.hpp"
#include <string>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <limits>

// Forward declaration so we can use the type in both C and C++ APIs
struct evaluator_handle;

extern "C" {

// C-style API exposed to other translation units (like ctypes.cpp)
evaluator_handle* exprtk_create_from_string(const char* expr_cstr);
double            exprtk_eval(evaluator_handle* handle, double e, double e2, double z);
void              exprtk_destroy(evaluator_handle* handle);
const char*       exprtk_last_error();

// For testing helper – returns last result as const char*
const char* expression_test(double e, double e2, double z, const char* h_expr);

} // extern "C"

namespace {
    // Thread-local so simultaneous calls from different threads do not overwrite each other's error text.
    thread_local std::string g_last_error;

    constexpr double EXPRTK_ERROR_VALUE = std::numeric_limits<double>::quiet_NaN();

    bool valid_number(double x)
    {
        return std::isfinite(x);
    }
}

// Pure C++ implementation details
struct evaluator_handle {
    double e  = 0.0;
    double e2 = 0.0;
    double z  = 0.0;
    double x0 = 0.0;
    double x1 = 0.0;
    double x2 = 0.0;
    double x3 = 0.0;

    bool compiled = false;
    std::string error_message;

    exprtk::symbol_table<double> symbol_table;
    exprtk::expression<double>   expression;
    exprtk::parser<double>       parser;

    explicit evaluator_handle(const std::string& expr_str)
    {
        if (expr_str.empty()) {
            error_message = "empty expression";
            return;
        }

        // Bind variables into the symbol table.
        symbol_table.add_variable("e",  e);
        symbol_table.add_variable("e2", e2);
        symbol_table.add_variable("z",  z);
        symbol_table.add_variable("x0", x0);
        symbol_table.add_variable("x1", x1);
        symbol_table.add_variable("x2", x2);
        symbol_table.add_variable("x3", x3);
        symbol_table.add_constants();

        expression.register_symbol_table(symbol_table);

        compiled = parser.compile(expr_str, expression);
        if (!compiled) {
            error_message = parser.error();
            if (error_message.empty()) {
                error_message = "expression compilation failed";
            }
        }
    }

    double eval(double e_val, double e2_val, double z_val)
    {
        if (!compiled) {
            g_last_error = error_message.empty()
                ? "attempted to evaluate an uncompiled expression"
                : error_message;
            return EXPRTK_ERROR_VALUE;
        }

        if (!valid_number(e_val) || !valid_number(e2_val) || !valid_number(z_val)) {
            g_last_error = "non-finite input to expression";
            return EXPRTK_ERROR_VALUE;
        }

        e  = e_val;
        e2 = e2_val;
        z  = z_val;

        const double result = expression.value();

        if (!valid_number(result)) {
            g_last_error = "expression evaluated to NaN or Inf";
            return EXPRTK_ERROR_VALUE;
        }

        g_last_error.clear();
        return result;
    }
};


// ---------- C API implementation ----------

extern "C" evaluator_handle* exprtk_create_from_string(const char* expr_cstr)
{
    g_last_error.clear();

    try {
        if (!expr_cstr || *expr_cstr == '\0') {
            g_last_error = "empty expression";
            return nullptr;
        }

        evaluator_handle* handle = new evaluator_handle(std::string(expr_cstr));

        if (!handle->compiled) {
            g_last_error = handle->error_message.empty()
                ? "expression compilation failed"
                : handle->error_message;
            delete handle;
            return nullptr;
        }

        return handle;
    }
    catch (const std::exception& ex) {
        g_last_error = std::string("exception while compiling expression: ") + ex.what();
        return nullptr;
    }
    catch (...) {
        g_last_error = "unknown exception while compiling expression";
        return nullptr;
    }
}

extern "C" double exprtk_eval(evaluator_handle* handle, double e, double e2, double z)
{
    if (!handle) {
        g_last_error = "null expression handle";
        return EXPRTK_ERROR_VALUE;
    }

    try {
        return handle->eval(e, e2, z);
    }
    catch (const std::exception& ex) {
        g_last_error = std::string("exception while evaluating expression: ") + ex.what();
        return EXPRTK_ERROR_VALUE;
    }
    catch (...) {
        g_last_error = "unknown exception while evaluating expression";
        return EXPRTK_ERROR_VALUE;
    }
}

extern "C" void exprtk_destroy(evaluator_handle* handle)
{
    delete handle;
}

extern "C" const char* exprtk_last_error()
{
    return g_last_error.c_str();
}


// ---------- Testing helper ----------

extern "C" const char* expression_test(double e, double e2, double z, const char* h_expr)
{
    static thread_local std::string last_result;

    evaluator_handle* handle = exprtk_create_from_string(h_expr);
    if (!handle) {
        last_result = std::string("Error: ") + exprtk_last_error();
        return last_result.c_str();
    }

    const double res = exprtk_eval(handle, e, e2, z);

    if (!std::isfinite(res)) {
        last_result = std::string("Error: ") + exprtk_last_error();
        exprtk_destroy(handle);
        return last_result.c_str();
    }

    exprtk_destroy(handle);

    std::ostringstream oss;
    oss << std::setprecision(17) << res;
    last_result = oss.str();
    return last_result.c_str();
}
