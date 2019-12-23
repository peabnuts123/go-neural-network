# Golang Neural Network

Learning about the details of Machine Learning and Neural Networks by implementing one in Golang (also learning about Golang). This project heavily relies on [gonum](https://github.com/gonum/gonum), a data science library for Go, similar to numpy.

## Prerequisites

You will need the following to run this project:

  - [Go](https://golang.org/dl/) programming language
  - MNIST data sets as CSV, available from [this source](https://pjreddie.com/projects/mnist-in-csv/), placed into the `data/` directory
    ```shell
    curl https://pjreddie.com/media/files/mnist_train.csv -o data/mnist_train.csv
    curl https://pjreddie.com/media/files/mnist_test.csv -o data/mnist_test.csv
    ```

## Running the project

Restore the project's dependencies by running:
```shell
go get -u ./...
```

<small>_(I think? I'm not 100% clear on Go package management, and whether you have to explicitly install dependencies)_</small>

Run the main project with:

```shell
go run src/main/main.go
```

## References

  * A lot of this project is inspired by following [this book](http://neuralnetworksanddeeplearning.com/), by [Michael Nielsen](http://michaelnielsen.org/)