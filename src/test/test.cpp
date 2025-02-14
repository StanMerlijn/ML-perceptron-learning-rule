/**
 * @file test.cpp
 * @author Stan Merlijn
 * @brief In this file the tests for the Perceptron class are defined.
 * @version 0.1
 * @date 2025-02-14
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include "../header/perceptron.hpp"
#include "../header/csv_reader.hpp"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <vector>
#include <cstdlib>   // For rand()


/**
 * @file test.cpp
 * @brief Unit tests for the Perceptron, PerceptronLayer and PerceptronNetwork classes.
 *
 * This file contains a series of test cases to verify the functionality of the Perceptron and PerceptronLayer classes.
 * The tests include training and prediction for various logic gates. 
 *
 * Test Cases:
 * - Perceptron for AND Gate: Tests the perceptron's ability to learn the AND gate.
 * - Perceptron for XOR Gate: Tests the perceptron's ability to learn the XOR gate.
 * - Perceptron for Iris Data Set: Tests the perceptron's ability to learn the Setosa and Versicolor classes.
 * - Perceptron for Iris Data Set: Tests the perceptron's ability to learn the Versicolor and Virginica classes.
 * 
 * @note The tests use the Catch2 framework for unit testing.
 */

// Define the default inputs
#define WEIGHTS std::vector<double>{0.5, 0.5} /**< Default weights for the perceptron. */
#define BIAS 0.5 /**< Default bias for the perceptron. */
#define LEARNING_RATE 0.1 /**< Default learning rate for the perceptron. */

// Read the iris data set
std::vector<std::vector<std::string>> data = read_csv("../../data/iris.csv"); 

// Extract the features and targets
std::vector<int>                targets  = get_targets(data); 
std::vector<std::vector<float>> features = get_features(data);

/**
 * @brief Test case for the AND gate using the Perceptron model.
 *
 * @details This test case trains a perceptron on the AND gate truth table.
 * The perceptron is expected to correctly learn the AND behavior:
 * - For inputs {0, 0}, {0, 1}, and {1, 0} the output should be 0.
 * - For input {1, 1} the output should be 1.
 * The test prints the final weights and computed loss after training.
 */
TEST_CASE("Perceptron AND gate", "[perceptron]")
{
    Perceptron andGate(WEIGHTS, BIAS, LEARNING_RATE);
    std::vector<std::vector<float>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<int> targets = {0, 0, 0, 1};
    
    // Train the perceptron
    andGate.update(inputs, targets, 100);

    REQUIRE(andGate.predict({0, 0}) == 0);
    REQUIRE(andGate.predict({0, 1}) == 0);
    REQUIRE(andGate.predict({1, 0}) == 0);
    REQUIRE(andGate.predict({1, 1}) == 1);

    // Print the weights
    std::cout << "Training for the AND gate:\n";
    andGate.__str__(1);

    // Calculate the loss
    double loss = andGate.loss(inputs, targets);
    std::cout << "Loss: " << loss << "\n" << std::endl;
}

/**
 * @brief Test case for the XOR gate using the Perceptron model.
 *
 * @details This test case trains a perceptron on the XOR gate truth table.
 * Since the XOR function is non-linearly separable, a single perceptron cannot correctly
 * learn the XOR behavior. Therefore, the expected behavior is:
 * - The perceptron incorrectly predicts the output for {0, 0} (i.e. output is not 0).
 * - The perceptron correctly predicts the output for {0, 1} (i.e. output is 1).
 * - The perceptron incorrectly predicts the output for {1, 0} (i.e. output is not 1).
 * - The perceptron correctly predicts the output for {1, 1} (i.e. output is 0).
 *
 * The test prints the final weights and computed loss after training.
 */
TEST_CASE("Perceptron XOR gate", "[Perceptron XOR]")
{
    Perceptron xorGate(WEIGHTS, BIAS, LEARNING_RATE);
    std::vector<std::vector<float>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<int> targets = {0, 1, 1, 0};

    // Train the perceptron
    xorGate.update(inputs, targets, 100);

    REQUIRE_FALSE(xorGate.predict({0, 0}) == 0); // This should fail for XOR
    REQUIRE(xorGate.predict({0, 1}) == 1); // This should pass
    REQUIRE_FALSE(xorGate.predict({1, 0}) == 1); // This should fail for XOR
    REQUIRE(xorGate.predict({1, 1}) == 0); // This should pass

    // Print the weights
    std::cout << "Training for the XOR gate:\n";
    xorGate.__str__(1);

    // Calculate the loss
    double loss = xorGate.loss(inputs, targets);
    std::cout << "Loss: " << loss << "\n" << std::endl;
}


// Perceptron XOR gate
// -------------------------------------------------------------------------------

// FAILED:
//   REQUIRE( xorGate.predict({0, 0}) == 0 )
// with expansion:
//   1 == 0

// FAILED:
//   REQUIRE( xorGate.predict({1, 0}) == 1 )
// with expansion:
//   0 == 1

// weights:
// -0.1 2.77556e-17 
// bias = 2.77556e-17
// Learning rate = 0.1
// Loss: 0.5

// A XOR gate is a linearly inseparable function, which means that a single perceptron
// cannot learn the weights to create a XOR gate.

/**
 * @brief Test case for the Iris data set (Setosa vs Versicolor) using the Perceptron model.
 *
 * @details This test trains a perceptron on the subset of the Iris data set
 * that contains only the Setosa and Versicolor classes. The perceptron is expected to
 * correctly separate the two classes with a loss of 0. The final weights and loss are printed.
 */
TEST_CASE("Iris data set - Perceptron Setosa en Versicolor", "[Perceptron Iris]")
{
    irisData iris01 = filter_data(features, targets, 2);

    // Student ID for seeding
    std::srand(1863967);

    // Generate random weights, bias, and learning rate
    std::vector<double> weights;
    double bias = double(std::rand()) / RAND_MAX * 0.5;
    double learningRate = double(std::rand()) / RAND_MAX * 0.5;
    
    // Generate random weights 
    for (int i = 0; i < iris01.features[0].size(); i++)
    {
        weights.push_back((double)std::rand() / RAND_MAX - 0.5);
    }

    // Create a perceptron object
    Perceptron irisPerceptron(weights, BIAS, LEARNING_RATE);

    // Train the perceptron
    irisPerceptron.update(iris01.features, iris01.targets, 1000);

    // Print the weights and loss after training
    double loss = irisPerceptron.loss(iris01.features, iris01.targets);
    std::cout << "Training the perceptron on the Setosa and Versicolor" << std::endl;
    irisPerceptron.__str__(1);
    std::cout << "Loss: " << loss << "\n" << std::endl;

    std::vector<int> predictions;
    std::vector<int> targets;

    for (int i = 0; i < iris01.features.size(); i++)
    {
        int prediction = irisPerceptron.predict(iris01.features[i]);
        int target = iris01.targets[i];
        predictions.push_back(prediction);
        targets.push_back(target);
    }

    CHECK(predictions == targets);

    // Training the perceptron on the iris data set
    // ------------------------------------------------
    // Training the perceptron on the Setosa and Versicolor
    // weights:
    // -0.244404 -0.199549 0.569217 0.417662 
    // bias = 0.4
    // Learning rate = 0.1
    // Loss: 0
    // The loss is 0 because the perceptron is able to separate the two classes
}

/**
 * @brief Test case for the Iris data set (Versicolor vs Virginica) using the Perceptron model.
 *
 * @details This test trains a perceptron on the subset of the Iris data set
 * that contains only the Versicolor and Virginica classes. In this scenario, the perceptron
 * cannot correctly separate the two classes, hence a non-zero loss is expected.
 * The test prints the final weights and computed loss after training.
 */
TEST_CASE("Iris data set - Perceptron Versicolor en Virginica", "[Perceptron Iris]")
{
    irisData iris12 = filter_data(features, targets, 0);

    // Student ID for seeding
    std::srand(1863967);

    // Generate random weights, bias, and learning rate
    std::vector<double> weights;
    double bias = double(std::rand()) / RAND_MAX * 0.5;
    double learningRate = double(std::rand()) / RAND_MAX * 0.5;
    
    // Generate random weights 
    for (int i = 0; i < iris12.features[0].size(); i++)
    {
        weights.push_back((double)std::rand() / RAND_MAX - 0.5);
    }

    // Create a perceptron object
    Perceptron irisPerceptron(weights, BIAS, LEARNING_RATE);

    // Train the perceptron
    irisPerceptron.update(iris12.features, iris12.targets, 1000);

    // Print the weights and loss after training
    double loss = irisPerceptron.loss(iris12.features, iris12.targets);
    std::cout << "Training the perceptron on the Versicolor and Virginica" << std::endl;
    irisPerceptron.__str__(1);
    std::cout << "Loss: " << loss << "\n" << std::endl;

    std::vector<int> predictions;
    std::vector<int> targets;

    for (int i = 0; i < iris12.features.size(); i++)
    {
        int prediction = irisPerceptron.predict(iris12.features[i]);
        int target = iris12.targets[i];
        predictions.push_back(prediction);
        targets.push_back(target);
    }

    CHECK(predictions == targets);

    // Training the perceptron on the Versicolor and Virginica
    // weights:
    // 32939.9 14870.2 27760 10130.2 
    // bias = 5000.5
    // Learning rate = 0.1
    // Loss: 0.5
    // The loss is 0.5 because the perceptron is not able to separate the two classes
}