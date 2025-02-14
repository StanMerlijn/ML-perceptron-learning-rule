## @file importing_dataset.py
## @brief Script to import and save the Iris dataset.
## 
## This script uses scikit-learn to load the Iris dataset and writes the data and 
## corresponding target values into CSV files. The features and targets are saved
## in 'data/iris.csv' and 'data/iris_target.csv' respectively.

from sklearn.datasets import load_iris

## @brief Main entry point for importing and saving the Iris dataset.
## 
## Loads the Iris dataset, then saves the features and targets to separate CSV files.
if __name__ == '__main__':
    iris = load_iris()
    # Save the dataset (features and target) into a file.
    with open('data/iris.csv', 'w') as f:
        for i in range(len(iris.data)):
            f.write(','.join([str(x) for x in iris.data[i]]) + ',' + str(iris.target[i]) + '\n')
    
    # Save the target values into a separate file.
    with open('data/iris_target.csv', 'w') as f:
        for i in range(len(iris.target)):
            f.write(str(iris.target[i]) + '\n')