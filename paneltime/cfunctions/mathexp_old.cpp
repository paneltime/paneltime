#include "exprtk.hpp"
#include <string>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <sstream>
#include <iomanip>

// Forward declaration so we can use the type in both C and C++ APIs
struct evaluator_handle;

extern "C" {

// C-style API exposed to other translation units (like ctypes.cpp)
evaluator_handle* exprtk_create_from_string(const char* expr_cstr);
double            exprtk_eval(evaluator_handle* handle, double e, double e2, double z);
void              exprtk_destroy(evaluator_handle* handle);

// For testing helper – returns last result as const char*
const char* expression_test(double e, double e2, double z, const char* h_expr);

} // extern "C"

// Pure C++ implementation details
struct evaluator_handle {
    double e  = 0.0;
    double e2 = 0.0;
    double z  = 0.0;
    double x0 = 0.0;
    double x1 = 0.0;
    double x2 = 0.0;
    double x3 = 0.0;

    std::string error_message;

    exprtk::symbol_table<double> symbol_table;
    exprtk::expression<double>   expression;
    exprtk::parser<double>       parser;

    explicit evaluator_handle(const std::string& expr_str)
    {
        // Bind variables into the symbol table
        symbol_table.add_variable("e",  e);
        symbol_table.add_variable("e2", e2);
        symbol_table.add_variable("z",  z);
        symbol_table.add_variable("x0", x0);
        symbol_table.add_variable("x1", x1);
        symbol_table.add_variable("x2", x2);
        symbol_table.add_variable("x3", x3);

        expression.register_symbol_table(symbol_table);

        if (!parser.compile(expr_str, expression)) {
            error_message = parser.error();
        }
    }

    double eval(double e_val, double e2_val, double z_val)
    {
        e  = e_val;
        e2 = e2_val;
        z  = z_val;
        return expression.value();
    }
};


// ---------- C API implementation ----------

extern "C" evaluator_handle* exprtk_create_from_string(const char* expr_cstr)
{
    try {
        if (!expr_cstr || *expr_cstr == '\0') {
            return nullptr;
        }
        std::string expr(expr_cstr);
        evaluator_handle* handle = new evaluator_handle(expr);
        if (!handle->error_message.empty()) {
            // You *could* log handle->error_message here if desired
            // but still return the handle; caller can decide what to do.
        }
        return handle;
    }
    catch (...) {
        // Allocation or parser could throw; return nullptr on failure
        return nullptr;
    }
}

extern "C" double exprtk_eval(evaluator_handle* handle, double e, double e2, double z)
{
    if (!handle) {
        return -9999.0; // same sentinel as before
    }
    return handle->eval(e, e2, z);
}

extern "C" void exprtk_destroy(evaluator_handle* handle)
{
    delete handle;
}


// ---------- Testing helper ----------

extern "C" const char* expression_test(double e, double e2, double z, const char* h_expr)
{
    static std::string last_result;  // static storage, returned as c_str()

    if (!h_expr || *h_expr == '\0') {
        last_result = "Error: empty expression";
        return last_result.c_str();
    }

    evaluator_handle handle{std::string(h_expr)};

    if (!handle.error_message.empty()) {
        last_result = std::string("Error: ") + handle.error_message;
        return last_result.c_str();
    }

    double res = handle.eval(e, e2, z);

    std::ostringstream oss;
    oss << std::setprecision(17) << res;
    last_result = oss.str();
    return last_result.c_str();
}
