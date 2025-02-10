#pragma once
#include "MSE.hpp"
#include <iostream>
#include <vector>

class Perceptron 
{
public:
    Perceptron(std::vector<double> weights, double bias, double learningRate);
    int predict(const std::vector<float>& x) const;
    void update(const std::vector<std::vector<float>>& inputs, const std::vector<int>& targets, int epochs);
    double loss(const std::vector<std::vector<float>>& inputs, const std::vector<int>& targets) const;
    void __str__(int verbose) const;
    
private:
    std::vector<double> weights;
    double bias;
    double learningRate;
};
