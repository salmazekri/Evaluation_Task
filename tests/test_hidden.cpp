// This file demonstrates the computation of the Jacobian matrix for a simple model
// using Clad's automatic differentiation and compares it with finite difference results.
#include "clad/Differentiator/Differentiator.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cassert>
#include <algorithm>

void model(double w0, double w1,
           double x0, double x1, double b0,
           double w2, double w3, double b1,
           double o0, double o1,
           double* out) {
    double h0 = std::max(0.0, w0 * x0 + w1 * x1 + b0);
    double h1 = std::max(0.0, w2 * x0 + w3 * x1 + b1);
    out[0] = o0 * h0;
    out[1] = o1 * h1;
}

void model_wrapper(const std::vector<double>& p, double* out) {
    model(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], out);
}

int main() {
    constexpr int N_IN = 10;
    constexpr int N_OUT = 2;
    constexpr double EPS = 1e-6;

    std::vector<double> params = {
        1.0, 2.0,   // x0, x1
        0.5, -0.25, 0.1,   // w0, w1, b0
        -0.3, 0.6, -0.2,   // w2, w3, b1
        1.0, -1.0          // o0, o1
    };

    double out[2];
    clad::matrix<double> J(N_OUT, N_IN + 2); 
    clad::jacobian(model).execute(
        params[0], params[1],
        params[2], params[3], params[4],
        params[5], params[6], params[7],
        params[8], params[9],
        out, &J
    );

    std::vector<std::vector<double>> J_numeric(N_OUT, std::vector<double>(N_IN));
    for (int i = 0; i < N_IN; ++i) {
        std::vector<double> p_plus = params, p_minus = params;
        p_plus[i] += EPS;
        p_minus[i] -= EPS;

        double o_plus[2], o_minus[2];
        model_wrapper(p_plus, o_plus);
        model_wrapper(p_minus, o_minus);

        for (int j = 0; j < N_OUT; ++j) {
            J_numeric[j][i] = (o_plus[j] - o_minus[j]) / (2 * EPS);
        }
    }

    // Compare
    std::cout << "=== Jacobian Comparison (Inputs + Weights) ===\n";
    const char* names[] = {"x0", "x1", "w0", "w1", "b0", "w2", "w3", "b1", "o0", "o1"};
    std::cout << std::fixed << std::setprecision(6);
    for (int j = 0; j < N_OUT; ++j) {
        for (int i = 0; i < N_IN; ++i) {
            double clad_val = J(j, i);
            double num_val = J_numeric[j][i];
            double err = std::abs(clad_val - num_val);
            std::cout << "∂out[" << j << "]/∂" << names[i] << ": "
                      << "Clad = " << clad_val
                      << ", Numeric = " << num_val
                      << ", Error = " << err << "\n";
        }
    }

    return 0;
}
