package main

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"

	"github.com/peabnuts123/go-neural-network/src/network"
	"gonum.org/v1/gonum/mat"
)

func main() {
	numEpochs := 30
	batchSize := 10
	learningRate := 3.0

	trainingData, testData := getDataFromFiles()

	net := network.NewNetwork([]int{784, 30, 10})

	net.Train(trainingData, numEpochs, batchSize, learningRate, testData)

	// @NOTE uncomment to inspect training data
	// for i := 0; i < 5; i++ {
	// 	log.Printf("Sample (%d):", i)
	// 	sample := trainingData[i]

	// 	// log.Printf("Input:\n %v\n", mat.Formatted(sample.InputActivation))
	// 	log.Printf("Output:\n %v\n\n", mat.Formatted(sample.ExpectedOutputActivation))
	// }
}

// Data from: https://pjreddie.com/projects/mnist-in-csv/
func getDataFromFiles() ([]*network.DataSample, []*network.DataSample) {
	log.Println("Reading training data...")
	trainingData := getDataSamplesFromFile("data/mnist_train.csv")
	log.Println("Reading test data...")
	testData := getDataSamplesFromFile("data/mnist_test.csv")

	log.Println("Finished reading data from disk.")
	return trainingData, testData
}

func getDataSamplesFromFile(filePath string) []*network.DataSample {
	fileReader, err := os.Open(filePath)
	if err != nil {
		log.Fatalln(err)
	}

	csvReader := csv.NewReader(fileReader)

	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatalln(err)
	}

	samples := make([]*network.DataSample, len(records))
	for i := 0; i < len(records); i++ {
		line := records[i]

		inputData := make([]float64, len(line)-1)
		outputData := make([]float64, 10)
		for tokenIndex := 0; tokenIndex < len(line); tokenIndex++ {
			value, _ := strconv.ParseInt(line[tokenIndex], 10, 8)

			if tokenIndex == 0 {
				// Convert value into an output class
				outputData[value] = 1
			} else {
				// Normalise pixel value input
				inputData[tokenIndex-1] = float64(value) / 255.0
			}
		}

		inputActivation := mat.NewDense(len(line)-1, 1, inputData)
		outputActivation := mat.NewDense(10, 1, outputData)

		samples[i] = &network.DataSample{
			InputActivation:          inputActivation,
			ExpectedOutputActivation: outputActivation,
		}
	}

	return samples
}
