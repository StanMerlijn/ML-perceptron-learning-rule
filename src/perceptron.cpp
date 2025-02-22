/**
 * @file perceptron.cpp
 * @author Stan Merlijn
 * @brief In this file the Perceptron class is implemented. 
 * @version 0.1
 * @date 2025-02-14
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include "header/perceptron.hpp"

Perceptron::Perceptron(std::vector<double> weights, double bias, double learningRate)
    : weights(weights), bias(bias), learningRate(learningRate) {}

int Perceptron::predict(const std::vector<float>& inputs) const 
{
    // Dot prodcut for an array of size 2  
    double dot_product = bias;
    for (int i = 0; i < weights.size(); i++) {
        dot_product += weights[i] * inputs[i];
    }
    // Threshold function
    return dot_product >= 0 ? 1 : 0;
}

void Perceptron::update(const std::vector<std::vector<float>>& inputs, const std::vector<int>& targets, int epochs) 
{
    // ensure both arrays are the same size
    if (inputs.size() != targets.size()) return;

    // Train the perceptron
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Loop through each input
        for (int i = 0; i < inputs.size(); i++) {
            // Get the prediction and error
            double pred = predict(inputs[i]);
            double error = targets[i] - pred;

            // Update each weight based on the input value
            for (int j = 0; j < weights.size(); j++) {
                weights[j] += learningRate * error * inputs[i][j];
            }
            // Update bias 
            bias += learningRate * error;
        }
    }
}

double Perceptron::loss(const std::vector<std::vector<float>>& inputs, const std::vector<int>& targets) const
{
    // Get the predictions for the inputs
    std::vector<int> predictions;
    predictions.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); i++) {
        predictions.push_back(predict(inputs[i]));
    }
    // Calculate the mean squared error between the targets and predictions
    return MSE(targets, predictions);
}

void Perceptron::__str__(int verbose) const
{
    // Printing the weights 
    std::cout << "weights for the perceptron:\n";
    for (int i = 0; i < weights.size(); i++) {
        std::cout << weights[i] << " ";
    }
    // Other info 
    if (verbose >= 1) {
        std::cout << "\nbias = " << bias << "\n";
        std::cout << "Learning rate = " << learningRate << std::endl;
    }
}