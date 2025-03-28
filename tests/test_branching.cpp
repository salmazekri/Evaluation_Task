/**
 * @file test_branching.cpp
 * @brief Demonstrates the use of Clad's automatic differentiation capabilities
 *        on a simple ReLU-based branching model.
 *
 * This example implements a ReLU (Rectified Linear Unit) model with branching
 * logic and computes its gradients using both Clad's automatic differentiation
 * and central difference finite difference methods. The results are compared
 * to evaluate the accuracy of Clad's differentiation.
 *
 */


#include "clad/Differentiator/Differentiator.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

void relu_branch_model(double x, double w, double b, double* out) {
    double z = w * x + b;
    if (z > 0)
        out[0] = z;
    else
        out[0] = 0;
}

// Central difference finite difference
double finite_diff(double x, double w, double b, int param_idx, double eps = 1e-6) {
    double out_plus, out_minus;

    if (param_idx == 0) { // x
        relu_branch_model(x + eps, w, b, &out_plus);
        relu_branch_model(x - eps, w, b, &out_minus);
    } else if (param_idx == 1) { // w
        relu_branch_model(x, w + eps, b, &out_plus);
        relu_branch_model(x, w - eps, b, &out_minus);
    } else if (param_idx == 2) { // b
        relu_branch_model(x, w, b + eps, &out_plus);
        relu_branch_model(x, w, b - eps, &out_minus);
    } else {
        assert(false && "Invalid param index");
    }

    return (out_plus - out_minus) / (2 * eps);
}

int main() {
    double x = 1.0, w = 2.0, b = -2.0;  
    double out[1];

    // Call forward
    relu_branch_model(x, w, b, out);
    std::cout << "z = " << w * x + b << ", ReLU Output = " << out[0] << "\n";

    // Compute Clad Jacobian
    clad::matrix<double> J(1, 5); 
    clad::jacobian(relu_branch_model).execute(x, w, b, out, &J);

    std::cout << "\n=== Gradient Comparison ===\n";
    const char* names[] = {"x", "w", "b"};
    for (int i = 0; i < 3; ++i) {
        double grad_clad = J(0, i);
        double grad_numeric = finite_diff(x, w, b, i);
        double error = std::abs(grad_clad - grad_numeric);

        std::cout << "∂out/∂" << names[i]
                  << ": Clad = " << std::setw(10) << grad_clad
                  << ", Numeric = " << std::setw(10) << grad_numeric
                  << ", Error = " << error << "\n";
    }

    return 0;
}
