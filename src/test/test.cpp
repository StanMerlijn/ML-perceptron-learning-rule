#include "../header/perceptron.hpp"
#include "../header/MSE.hpp"

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
}




