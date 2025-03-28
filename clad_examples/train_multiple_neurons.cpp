#include "clad/Differentiator/Differentiator.h"
#include <iostream>
#include <vector>
/*clang++ -std=c++17 -O0 -g     -I/root/llvm-project/llvm/tools/clad/include     -I/usr/include/eigen3     -fplugin=/root/llvm-project/llvm/tools/clad/build/lib/clad.so     train_multiple_neurons.cpp     -o diffwithclad
./diffwithclad*/
const double LEARNING_RATE = 0.01;
const int EPOCHS = 100;
const int BATCH_SIZE = 2;
const int OUTPUT_SIZE = 2;
const int PARAM_COUNT = 6;  

// Multi-output Linear Model: Computes y1 and y2
void forward(
    double W0, double W1, double b1, 
    double W2, double W3, double b2,
    double x0, double x1, double* y)  
{
    y[0] = W0 * x0 + W1 * x1 + b1;  // First neuron
    y[1] = W2 * x0 + W3 * x1 + b2;  // Second neuron
}

// Loss Function: Mean Squared Error (MSE) for multiple outputs
double loss(const double* y, const double* target) {
    double loss_val = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        loss_val += 0.5 * (y[i] - target[i]) * (y[i] - target[i]);
    }
    return loss_val;
}

int main() {
    // Initialize weights and biases
    double W[4] = {0.5, -0.3, 0.2, 0.4};  
    double b[2] = {0.1, -0.2};            

    // Training data (2D inputs and corresponding multi-dimensional targets)
    std::vector<std::vector<double>> x_data = {
        {1.0, 2.0}, 
        {2.0, -1.0}
    };
    std::vector<std::vector<double>> target_data = {
        {1.5, -0.5},
        {0.2, 0.8}
    };

    // Training loop over multiple epochs
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        std::cout << "\n=== Epoch " << (epoch + 1) << " ===" << std::endl;

        // Initialize batch gradients
        double total_dW[4] = {0, 0, 0, 0};
        double total_db[2] = {0, 0};

        // Process all examples in the batch
        for (int i = 0; i < BATCH_SIZE; i++) {
            double x0 = x_data[i][0];
            double x1 = x_data[i][1];
            double target[OUTPUT_SIZE] = {target_data[i][0], target_data[i][1]};

            // Output storage
            double y[OUTPUT_SIZE] = {0.0, 0.0};

            // Compute forward pass
            forward(W[0], W[1], b[0], W[2], W[3], b[1], x0, x1, y);
            std::cout << "Sample " << i + 1 << " Prediction: [" << y[0] << ", " << y[1] << "]" << std::endl;

            // Compute loss
            double loss_val = loss(y, target);
            std::cout << "Loss: " << loss_val << std::endl;

            // Compute Jacobian using Clad
            clad::matrix<double> d_output(2, 10);  

            // Use temporary variables for output
            double temp_y[OUTPUT_SIZE] = {0.0, 0.0};
            
            // Execute jacobian with correct parameter order
            clad::jacobian(forward).execute(
                W[0], W[1], b[0], W[2], W[3], b[1], 
                x0, x1, temp_y, &d_output
            );

            // Debug print Jacobian dimensions
            std::cout << "Jacobian Matrix: " << d_output.rows() << "x" << d_output.cols() << std::endl;


            // Compute gradients using the Jacobian
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                double dL_dW0 = d_output(j, 0) * (y[j] - target[j]);
                double dL_dW1 = d_output(j, 1) * (y[j] - target[j]);
                double dL_db  = d_output(j, 2) * (y[j] - target[j]);

                double dL_dW2 = d_output(j, 3) * (y[j] - target[j]);
                double dL_dW3 = d_output(j, 4) * (y[j] - target[j]);
                double dL_db2 = d_output(j, 5) * (y[j] - target[j]);

                // Accumulate gradients for batch update
                total_dW[0] += dL_dW0;
                total_dW[1] += dL_dW1;
                total_db[0] += dL_db;

                total_dW[2] += dL_dW2;
                total_dW[3] += dL_dW3;
                total_db[1] += dL_db2;
            }
        }

        // Compute batch average gradient
        for (int k = 0; k < 4; k++) total_dW[k] /= BATCH_SIZE;
        for (int k = 0; k < 2; k++) total_db[k] /= BATCH_SIZE;

        // Apply gradient descent update
        for (int k = 0; k < 4; k++) W[k] -= LEARNING_RATE * total_dW[k];
        for (int k = 0; k < 2; k++) b[k] -= LEARNING_RATE * total_db[k];

        // Print updated parameters
        std::cout << "Updated Weights: [" << W[0] << ", " << W[1] << ", " << W[2] << ", " << W[3] << "]" << std::endl;
        std::cout << "Updated Biases: [" << b[0] << ", " << b[1] << "]" << std::endl;
    }

    return 0;
}
/*=== Epoch 100 ===
Sample 1 Prediction: [1.38549, -0.401802]
Loss: 0.0113776
Jacobian Matrix: 2x10
Sample 2 Prediction: [0.305692, 0.710619]
Loss: 0.00957992
Jacobian Matrix: 2x10
Updated Weights: [0.313932, 0.466412, 0.349249, -0.255954]
Updated Biases: [0.141642, -0.241642]*/