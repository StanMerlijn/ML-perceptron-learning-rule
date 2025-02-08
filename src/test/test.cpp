#include "../header/perceptron.hpp"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <vector>

// Define the default inputs
#define WEIGHTS std::vector<double>{0.5, 0.5}
#define BIAS 0.5
#define LEARNING_RATE 0.1

// Define the binary inputs 
std::vector<int> in00 = {0, 0};
std::vector<int> in01 = {0, 1};
std::vector<int> in10 = {1, 0};
std::vector<int> in11 = {1, 1};

TEST_CASE("Perceptron AND gate", "[perceptron]")
{
    // Create a perceptron object
    Perceptron andGate(WEIGHTS, BIAS, LEARNING_RATE);
    std::vector<std::vector<int>> inputs = {in00, in01, in10, in11};
    std::vector<int> targets = {0, 0, 0, 1};
    
    // Train the perceptron
    andGate.update(inputs, targets, 100);

    CHECK(andGate.predict(in00) == 0);
    CHECK(andGate.predict(in01) == 0);
    CHECK(andGate.predict(in10) == 0);
    CHECK(andGate.predict(in11) == 1);

    // Print the weights
    andGate.__str__(1);

    // Calculate the loss
    double loss = andGate.loss(inputs, targets);
    std::cout << "Loss: " << loss << std::endl;
}

TEST_CASE("Perceptron XOR gate", "[Perceptron XOR]")
{
    Perceptron xorGate(WEIGHTS, BIAS, LEARNING_RATE);
    std::vector<std::vector<int>> inputs = {in00, in01, in10, in11};
    std::vector<int> targets = {0, 1, 1, 0};

    // Train the perceptron
    xorGate.update(inputs, targets, 100);

    CHECK(xorGate.predict(in00) == 0);
    CHECK(xorGate.predict(in01) == 1);
    CHECK(xorGate.predict(in10) == 1);
    CHECK(xorGate.predict(in11) == 0);

    // Print the weights
    xorGate.__str__(1);

    // Calculate the loss
    double loss = xorGate.loss(inputs, targets);
    std::cout << "Loss: " << loss << std::endl;
}


// Perceptron XOR gate
// -------------------------------------------------------------------------------

// FAILED:
//   CHECK( xorGate.predict(in00) == 0 )
// with expansion:
//   1 == 0

// FAILED:
//   CHECK( xorGate.predict(in10) == 1 )
// with expansion:
//   0 == 1

// weights:
// -0.1 2.77556e-17 
// bias = 2.77556e-17
// Learning rate = 0.1
// Loss: 0.5

// A XOR gate is a linearly inseparable function, which means that a single perceptron
// cannot learn the weights to create a XOR gate.
