/**
 * 
 * Functions:
 * - `f1(double x, double y)`: A scalar function of two variables.
 * - `f2(double x, double y, double* result)`: A vector-valued function of two variables.
 * - `relu(double x)`: A ReLU activation function.
 * - `forward_pass(double x1, double x2, double* result)`: Simulates a forward pass of a simple neural network.
 */

#include "clad/Differentiator/Differentiator.h"
#include <iostream>
#include <cmath>
#include <cstring>

double f1(double x, double y) {
    return x * x + x * y;
}

void f2(double x, double y, double* result) {
    result[0] = x * x + y;
    result[1] = x * y;
}

=double relu(double x) {
    return (x > 0) * x; 
}

void forward_pass(double x1, double x2, double* result) {
    const double w11 = 0.5, w12 = 0.3;
    const double w21 = 0.4, w22 = 0.6;
    const double b1 = 0.1, b2 = 0.2;
    
    double z1 = w11 * x1 + w12 * x2 + b1;
    double z2 = w21 * x1 + w22 * x2 + b2;
    double h1 = relu(z1);
    double h2 = relu(z2);
    
    // Output layer
    result[0] = h1;
    result[1] = h2;
}

int main() {
    double x = 2.0, y = 3.0;
    
    std::cout << "=== Test 1: Gradient of f1(x,y) at (2,3) ===" << std::endl;
    auto d_f1 = clad::gradient(f1);
    double dx1 = 0, dy1 = 0;
    d_f1.execute(x, y, &dx1, &dy1);

    std::cout << "df1/dx = " << dx1 << std::endl;
    std::cout << "df1/dy = " << dy1 << std::endl;

    std::cout << "\n=== Test 2: Jacobian of f2(x,y) at (2,3) ===" << std::endl;

    double result[2] = {0, 0};  

    clad::matrix<double> d_output(2, 4);  

    auto j_f2 = clad::jacobian(f2);
    j_f2.execute(x, y, result, &d_output);

    std::cout << "f2(2,3) = [" << result[0] << ", " << result[1] << "]" << std::endl;
    std::cout << "Jacobian Matrix:" << std::endl;
    std::cout << "[[" << d_output[0][0] << ", " << d_output[0][1] << "]," << std::endl;
    std::cout << " [" << d_output[1][0] << ", " << d_output[1][1] << "]]" << std::endl;

    return 0;
}
