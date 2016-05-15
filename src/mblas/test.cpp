//#include <iostream>
#include "mblas.h"

#include <iostream>

/*
template <Function, Array>
Array SGD(const Function& J, const Array& X, const Array Y,
          batch_size=50, steps=100) {
    auto dJ = J.d('theta');
    Array theta = zeros(X.shape[1]);
    for(size_t i = 0; i < steps * batch_size && i < X.shape[0]; i += batch_size) {
        ArrayPiece XBatch(X, i, batch_size);
        ArrayPiece YBatch(Y, i, batch_size);
        theta += mean(dJ(XBatch, YBatch) * theta, 1);
    }
    return theta;
}*/

int main(int argc, char** argv) {
    
    const auto& z = s(5) + pow(s(5), s(4)) * s(9);
    const auto& b = z();
    
    std::cerr << b << std::endl;
    
    //gpu::Array X, XTest;
    //gpu::Array Y, YTest;
    //
    //load("mnist.train.tsv", X, Y);
    //load("mnist.test.tsv", XTest, YTest);
    //
    //gpu::Tensor theta('theta');
    //gpu::Tensor x('x');
    //gpu::Tensor y('y');
    //
    //T<'x'> x
    //S<0> s0;
    //S<1> s1;
    //
    //auto h = softmax(dot(trans(theta), x));
    //auto J = zeros(1);
    //
    //const auto& grads = network.forward(_1).backprop(_2).grads();
    //network.grads(x, y);
    //const auto& all_gradients = grads(x,y);
    //network.update(new_gradients);
    //
    //for(size_t i = 0; i < 10; ++i)
    //    J += (1 - i) * log(1 - h(x)) + i * log(h(x));
    //
    //auto h_opt = h(theta, SGD(J, X, Y, 50, 1000));
    //auto decision = h_opt >= 0.5;
    //
    //auto accuracy = mean(decision(XTest) == YTest);
    //
    //std::cerr << "Accuracy: " << accuracy() << std::endl;    

    return 0;
}
