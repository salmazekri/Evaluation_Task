// Train a simple Transformer model using manual backpropagation
// build by: clang++ -O3 -std=c++20 -fopenmp train_transformer.cpp -o train_transformer
// run by: ./train_transformer

#include <iostream>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

const int EMBEDDING_SIZE = 16;  // Token embedding size
const int HIDDEN_SIZE = 32;     // Hidden layer size
const int HEADS = 2;            // Number of attention heads
const double LEARNING_RATE = 0.01;
const int EPOCHS = 10;          // Number of training iterations

// Scaled Dot-Product Attention
MatrixXd scaled_dot_product_attention(const MatrixXd& Q, const MatrixXd& K, const MatrixXd& V) {
    MatrixXd scores = (Q * K.transpose()) / sqrt(K.cols());  
    MatrixXd attention = scores.array().exp();  
    return attention * V;
}

// Transformer Block
class TransformerBlock {
public:
    MatrixXd Wq, Wk, Wv, Wo;
    MatrixXd W1, W2;
    VectorXd b1, b2;

    TransformerBlock(int dim, int heads) {
        Wq = MatrixXd::Random(dim, dim);
        Wk = MatrixXd::Random(dim, dim);
        Wv = MatrixXd::Random(dim, dim);
        Wo = MatrixXd::Random(dim, dim);
        W1 = MatrixXd::Random(HIDDEN_SIZE, dim);
        W2 = MatrixXd::Random(dim, HIDDEN_SIZE);
        b1 = VectorXd::Random(HIDDEN_SIZE);
        b2 = VectorXd::Random(dim);
    }

    // Forward pass
    MatrixXd forward(const MatrixXd& input) {
        MatrixXd Q = Wq * input;
        MatrixXd K = Wk * input;
        MatrixXd V = Wv * input;

        MatrixXd attention_output = scaled_dot_product_attention(Q, K, V);
        MatrixXd output = Wo * attention_output;

        VectorXd hidden = (W1 * output.col(0) + b1).array().tanh();
        return W2 * hidden + b2;
    }

    // Compute loss (MSE)
    double loss(const VectorXd& pred, const VectorXd& target) {
        return 0.5 * (pred - target).squaredNorm();
    }

    // Backpropagation (Manual)
    void train_step(const MatrixXd& input, const VectorXd& target) {
        // Forward pass
        MatrixXd Q = Wq * input;
        MatrixXd K = Wk * input;
        MatrixXd V = Wv * input;
        MatrixXd attention_output = scaled_dot_product_attention(Q, K, V);
        MatrixXd output = Wo * attention_output;

        VectorXd hidden = (W1 * output.col(0) + b1).array().tanh();
        VectorXd pred = W2 * hidden + b2;
        double loss_val = 0.5 * (pred - target).squaredNorm();

        // Print loss
        std::cout << "Loss: " << loss_val << std::endl;

        // Compute gradients
        VectorXd d_pred = pred - target;
        MatrixXd dW2 = d_pred * hidden.transpose();
        VectorXd db2 = d_pred;

        VectorXd d_hidden = (W2.transpose() * d_pred).array() * (1 - hidden.array().square());
        MatrixXd dW1 = d_hidden * output.col(0).transpose();
        VectorXd db1 = d_hidden;

        MatrixXd d_output = W1.transpose() * d_hidden;
        MatrixXd dWo = d_output * attention_output.transpose();

        // Gradient Descent Update
        W1 -= LEARNING_RATE * dW1;
        W2 -= LEARNING_RATE * dW2;
        b1 -= LEARNING_RATE * db1;
        b2 -= LEARNING_RATE * db2;
        Wo -= LEARNING_RATE * dWo;

        // Print weight updates
        std::cout << "W1 norm: " << W1.norm() << ", Wo norm: " << Wo.norm() << std::endl;
    }
};

// Training loop
void train(TransformerBlock& model, const std::vector<MatrixXd>& data, const std::vector<VectorXd>& labels) {
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double total_loss = 0.0;
        std::cout << "\n=== Epoch " << (epoch + 1) << " ===" << std::endl;
        for (size_t i = 0; i < data.size(); i++) {
            model.train_step(data[i], labels[i]);  
        }
    }
}

int main() {
    TransformerBlock model(EMBEDDING_SIZE, HEADS);

    // Generate toy training data
    std::vector<MatrixXd> inputs = { MatrixXd::Random(EMBEDDING_SIZE, 1), MatrixXd::Random(EMBEDDING_SIZE, 1) };
    std::vector<VectorXd> targets = { VectorXd::Random(EMBEDDING_SIZE), VectorXd::Random(EMBEDDING_SIZE) };

    // Train the model
    train(model, inputs, targets);

    return 0;
}
/*=== Epoch 10 ===
Loss: 0.0947092
W1 norm: 13.0047, Wo norm: 9.1445
Loss: 0.147898
W1 norm: 13.0047, Wo norm: 9.1445*/