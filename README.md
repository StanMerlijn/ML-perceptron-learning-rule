# ML-Perceptron-learning-rule

## Student

Name: Stan Merlijn

Student nummer: 1863967

## Introduction
In this repository, we will implement and test a perceptron using the learning rule. This will be demonstrated by creating AND and XOR gates and by evaluating the perceptron’s ability to classify the [Iris dataset](https://scikit-learn.org/1.5/auto_examples/datasets/plot_iris_dataset.html). The performance will be measured using the [MSE](src/header/MSE.hpp) metric. You can find the assignment [here](https://canvas.hu.nl/courses/44675/assignments/343529).

## Documentation
For this assignment, the documentation was generated with Doxygen. The LaTeX documentation is available [here](docs/latex/refman.pdf) and, to view the HTML documentation locally, open [index.html](docs/html/index.html) in a browser.

## Installing
Enter the test directory and then Generate build files:

```
cmake -S . -B build
```

Build the project:

```
cmake --build build
```

Run the executable:

```
./build/MLPerceptronTest
```