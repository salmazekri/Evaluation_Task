#include "clad/Differentiator/Differentiator.h"
#include <iostream>

double power_func(double x) {
    return exp(x);  
}

int main() {
    double x = 1;  
    auto grad = clad::gradient(power_func);
    
    double dx = 0;
    grad.execute(x, &dx);

    std::cout << "Gradient at x = " << x << ": " << dx << std::endl;

    return 0;
}
