/**
 * @file test_discontinuity.cpp
 * @brief Demonstrates the use of clad's automatic differentiation to compute
 *        gradients for a simple neural network model with a ReLU activation function.
 *
 * This example defines a simple model with one ReLU hidden unit and a linear output layer.
 * The model computes the loss function as the squared difference between the predicted
 * output and a target value. The program uses clad's `jacobian` function to compute
 * the gradients of the loss with respect to the model parameters, even at points where
 * the ReLU introduces a discontinuity in the gradient.
 */

#include "clad/Differentiator/Differentiator.h"
#include <iostream>
#include <algorithm>  

void model_and_loss(double x0, double x1,
                    double w0, double w1, double b1,
                    double w2, double b2,
                    double target,
                    double* output) {
    double h = std::max(0.0, w0 * x0 + w1 * x1 + b1);  // ReLU
    double y = w2 * h + b2;
    double diff = y - target;
    output[0] = 0.5 * diff * diff;
}

int main() {
    // Inputs that lead to z == 0
    double x0 = 1.0;
    double x1 = 2.0;
    double target = 1.0;

    // Set weights to force w0*x0 + w1*x1 + b1 == 0
    double w0 = 0.5;
    double w1 = -0.25;
    double b1 = 0.0;

    double w2 = 1.0;
    double b2 = 0.0;

    double loss_output[1];
    clad::matrix<double> dJ(1, 9);  

    model_and_loss(x0, x1, w0, w1, b1, w2, b2, target, loss_output);
    clad::jacobian(model_and_loss).execute(x0, x1, w0, w1, b1, w2, b2, target, loss_output, &dJ);

    std::cout << "=== Probing ReLU Discontinuity ===\n";
    std::cout << "Loss: " << loss_output[0] << "\n";
    std::cout << "Gradients:\n";
    std::cout << " ∂L/∂w0 = " << dJ(0, 2) << "\n";
    std::cout << " ∂L/∂w1 = " << dJ(0, 3) << "\n";
    std::cout << " ∂L/∂b1 = " << dJ(0, 4) << "\n";
    std::cout << " ∂L/∂w2 = " << dJ(0, 5) << "\n";
    std::cout << " ∂L/∂b2 = " << dJ(0, 6) << "\n";

    return 0;
}
