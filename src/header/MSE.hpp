#pragma once
#include <iostream>
#include <vector>
#include <cmath>

/**
 * @brief Calculates the mean squared error between two vectors.
 * MSE = Σ | d – y |2 / n
 *  
 * This function computes the mean squared error (MSE) between the target values
 * and the predicted values. The MSE is a measure of the average squared difference
 * between the estimated values and the actual value.
 * 
 * @param targets A vector of target values.
 * @param predictions A vector of predicted values.
 * @return The mean squared error between the targets and predictions. 
 *         Returns -1 if the sizes of the input vectors do not match.
 */

inline double MSE(const std::vector<int>& targets, const std::vector<int>& predictions) 
{   
    // Ensure both arrays are the same size
    if (targets.size() != predictions.size()) return -1;

    double sum = 0;
    for (int i = 0; i < targets.size(); i++) {
        sum += pow(std::abs(targets[i] - predictions[i]), 2);
    }
    return sum / targets.size();
}