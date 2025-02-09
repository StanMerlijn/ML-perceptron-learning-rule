from sklearn.datasets import load_iris

# this file is purelly for importing the dataset
# and saving it to a file

if __name__ == '__main__':
    iris = load_iris()
    # save the dataset to a file
    with open('data/iris.csv', 'w') as f:
        for i in range(len(iris.data)):
            f.write(','.join([str(x) for x in iris.data[i]]) + ',' + str(iris.target[i]) + '\n')
    
    # save the targets to a file
    with open('data/iris_target.csv', 'w') as f:
        for i in range(len(iris.target)):
            f.write(str(iris.target[i]) + '\n')
