#include "exprtk.hpp"
#include <string>
#include <iostream>
#include <cstdio>
#include <algorithm>

extern "C" {

struct evaluator_handle;
std::string last_result;

evaluator_handle* exprtk_create_from_string(const std::string& expr);
double exprtk_eval(evaluator_handle* handle, double e, double z);
void exprtk_destroy(evaluator_handle* handle);

}

struct evaluator_handle {
    double e = 0.0;
    double z = 0.0;
    double x0 = 0.0;
    double x1 = 0.0;
    double x2 = 0.0;
    double x3 = 0.0;

    exprtk::symbol_table<double> symbol_table;
    exprtk::expression<double> expression;
    exprtk::parser<double> parser;

    evaluator_handle(const std::string& expr_str) {
        symbol_table.add_variable("e", e);
        symbol_table.add_variable("z", z);
        symbol_table.add_variable("x0", x0);
        symbol_table.add_variable("x1", x1);
        symbol_table.add_variable("x2", x2);
        symbol_table.add_variable("x3", x3);
        expression.register_symbol_table(symbol_table);
        if (!parser.compile(expr_str, expression)) {
            std::cerr << "ExprTk parse error:\n" << parser.error() << "\n";
            std::cerr << "Expression was:\n" << expr_str << "\n";
            throw std::runtime_error("ExprTk parse error");
        }
    }

    double eval(double e_val, double z_val) {
        e = e_val;
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

extern "C" double exprtk_eval(evaluator_handle* handle, double e, double z) {
    if (!handle) return -9999.0;
    return handle->eval(e, z);
}

extern "C" void exprtk_destroy(evaluator_handle* handle) {
    delete handle;
}


EXPORT const char* expression_test(double e, double z, const char* h_expr) {
    try {
        std::string expr_str(h_expr);
        auto* h_func = exprtk_create_from_string(expr_str);
        if (!h_func) {
            last_result = "error: failed to create expression";
            return last_result.c_str();
        }

        double x = exprtk_eval(h_func, e, z);
        exprtk_destroy(h_func);

        last_result = std::to_string(x);
        return last_result.c_str();
    } catch (const std::exception& ex) {
        last_result = std::string("error: ") + ex.what();
        return last_result.c_str();
    } catch (...) {
        last_result = "error: unknown exception";
        return last_result.c_str();
    }
}
