#include "exprtk.hpp"
#include <string>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <sstream>
#include <iomanip>

extern "C" {

struct evaluator_handle;
std::string last_result;

evaluator_handle* exprtk_create_from_string(const std::string& expr);
double exprtk_eval(evaluator_handle* handle, double e, double e2, double z);
void exprtk_destroy(evaluator_handle* handle);

}

struct evaluator_handle {
    double e = 0.0;
    double e2 = 0.0;
    double z = 0.0;
    double x0 = 0.0;
    double x1 = 0.0;
    double x2 = 0.0;
    double x3 = 0.0;

    std::string error_message; 

    exprtk::symbol_table<double> symbol_table;
    exprtk::expression<double> expression;
    exprtk::parser<double> parser;

    evaluator_handle(const std::string& expr_str) {
        symbol_table.add_variable("e", e);
        symbol_table.add_variable("e2", e2); 
        symbol_table.add_variable("z", z);
        symbol_table.add_variable("x0", x0);
        symbol_table.add_variable("x1", x1);
        symbol_table.add_variable("x2", x2);
        symbol_table.add_variable("x3", x3);
        expression.register_symbol_table(symbol_table);
        
        if (!parser.compile(expr_str, expression)) {
            error_message = parser.error() + "\n";
        }
    }

    double eval(double e_val,double e2_val, double z_val) {
        e = e_val;
        e2 = e2_val;
        z = z_val;
        return expression.value();
    }
};

extern "C" evaluator_handle* exprtk_create_from_string(const std::string& expr) {
    try {
        return new evaluator_handle(expr);
    } catch (...) {
        return nullptr;
    }
}

extern "C" double exprtk_eval(evaluator_handle* handle, double e, double e2, double z) {
    if (!handle) return -9999.0;
    return handle->eval(e, e2, z);
}

extern "C" void exprtk_destroy(evaluator_handle* handle) {
    delete handle;
}


EXPORT const char* expression_test(double e, double e2, double z, const char* h_expr) {
    static std::string last_result;
    evaluator_handle handle(h_expr);

    if (!handle.error_message.empty()) {
        last_result = std::string("Error:") + handle.error_message;
        return last_result.c_str();  // Return error message immediately
    }

    double res = handle.eval(e, e2, z);
    //printf("Result1: %f\n\n", handle.x0);
    //printf("Result2: %f\n\n", handle.x1);
    //printf("Result3: %f\n\n", handle.x2);
    std::ostringstream oss;
    oss << std::setprecision(17) << res;
    last_result = oss.str();
    return last_result.c_str();  // Return result as string
}
