package network

import (
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/peabnuts123/go-neural-network/src/util"
	"gonum.org/v1/gonum/mat"
)

type Network struct {
	layerSizes []int
	weights    []*mat.Dense
	biases     []*mat.Dense
}

type DataSample struct {
	InputActivation          *mat.Dense
	ExpectedOutputActivation *mat.Dense
}

func NewNetwork(layerSizes []int) *Network {
	var network Network
	network.layerSizes = layerSizes

	// @NOTE: do not iterate layer 1
	for i := 1; i < len(layerSizes); i++ {
		// Weights is a matrix of values.
		//	i.e. weight at (3,4) is the connection between node 4
		//	on the previous layer and node 3 on this layer
		network.weights = append(network.weights, util.RandomMatrix(layerSizes[i], layerSizes[i-1]))
		// Biases is a column vector of values. 1 value for each node in this layer
		network.biases = append(network.biases, util.RandomMatrix(layerSizes[i], 1))
	}

	return &network
}

func (self *Network) FeedForward(inputActivations *mat.Dense) *mat.Dense {
	inputDimR, _ := inputActivations.Dims()

	if inputDimR != self.layerSizes[0] {
		log.Fatalf("Cannot feed data forward. Expected vector of length %d, but data was length %d", self.layerSizes[0], inputDimR)
	}

	currentActivation := inputActivations

	for layerIndex := 0; layerIndex < len(self.layerSizes)-1; layerIndex++ {
		layerBiases := self.biases[layerIndex]
		layerWeights := self.weights[layerIndex]

		result := &mat.Dense{}
		result.Mul(layerWeights, currentActivation)
		result.Add(result, layerBiases)
		currentActivation = sigmoid(result)
	}

	return currentActivation
}

func (self *Network) Train(trainingData []*DataSample, epochs int, batchSize int, learningRate float64, testData []*DataSample) {
	rand.Seed(time.Now().UnixNano())
	log.Printf("(Pre-training): %d / %d", self.evaluateTestData(testData), len(testData))

	for epoch := 0; epoch < epochs; epoch++ {
		// shuffle data
		rand.Shuffle(len(trainingData), func(i, j int) { trainingData[i], trainingData[j] = trainingData[j], trainingData[i] })

		// Iterate through training data in batches
		// 	perform backwards propagation / learning for each batch
		for batchIndex := 0; batchIndex < len(trainingData); batchIndex += batchSize {
			self.processTrainingBatch(trainingData, batchIndex, batchIndex+batchSize, learningRate)
		}

		// 	output performance on test data
		log.Printf("Epoch %d: %d / %d test data correct", epoch, self.evaluateTestData(testData), len(testData))
	}
}

func (self *Network) processTrainingBatch(trainingData []*DataSample, batchStartIndex int, batchEndIndex int, learningRate float64) {
	// Sum up all the changes to every weight and bias in the network
	// Divide the changes by batch size
	// Multiply the amount of change by learning rate
	// Add all these changes to the biases and weights in the network

	var weightSums []*mat.Dense
	var biasSums []*mat.Dense
	for i := 0; i < len(self.layerSizes)-1; i++ {
		weightSums = append(weightSums, util.EmptyDense(self.weights[i]))
		biasSums = append(biasSums, util.EmptyDense(self.biases[i]))
	}

	for batchIndex := batchStartIndex; batchIndex < batchEndIndex; batchIndex++ {
		expectedOutputActivations := trainingData[batchIndex].ExpectedOutputActivation
		inputActivations := trainingData[batchIndex].InputActivation

		weightDeltas, biasDeltas := self.backprop(inputActivations, expectedOutputActivations)

		for i := 0; i < len(weightDeltas); i++ {
			weightSums[i].Add(weightSums[i], weightDeltas[i])
		}

		for i := 0; i < len(biasDeltas); i++ {
			biasSums[i].Add(biasSums[i], biasDeltas[i])
		}
	}

	for i := 0; i < len(self.weights); i++ {
		weightSums[i].Scale(learningRate/float64(batchEndIndex-batchStartIndex), weightSums[i])
		self.weights[i].Sub(self.weights[i], weightSums[i])
	}

	for i := 0; i < len(self.biases); i++ {
		biasSums[i].Scale(learningRate/float64(batchEndIndex-batchStartIndex), biasSums[i])
		self.biases[i].Sub(self.biases[i], biasSums[i])
	}
}

// This is pretty terrible, ported from Python, and designed to take advantage of numpy specifically
// This should be rewritten in a gonum style
// It also needs annotation
func (self *Network) backprop(inputActivations *mat.Dense, expectedOutputActivations *mat.Dense) ([]*mat.Dense, []*mat.Dense) {
	// Return a tuple `(biasDeltas, weightDeltas) representing the
	// gradient for the cost function.  `biasDeltas` and
	// `weightDeltas` are layer-by-layer lists of gonum matrices, similar
	// to `self.biases` and `self.weights`.

	// Initialise deltas to the same size as corresponding layer matrices
	weightDeltas := []*mat.Dense{}
	biasDeltas := []*mat.Dense{}
	for i := 0; i < len(self.layerSizes)-1; i++ {
		weightDeltas = append(weightDeltas, util.EmptyDense(self.weights[i]))
		biasDeltas = append(biasDeltas, util.EmptyDense(self.biases[i]))
	}

	// Feed forward
	var currentActivation *mat.Dense = inputActivations
	// Array of activations for every layer
	var layerActivations []*mat.Dense = []*mat.Dense{inputActivations}
	var zValues []*mat.Dense
	for layerIndex := 0; layerIndex < len(self.biases); layerIndex++ {
		layerBiases := self.biases[layerIndex]
		layerWeights := self.weights[layerIndex]

		z := &mat.Dense{}
		z.Mul(layerWeights, currentActivation)
		z.Add(z, layerBiases)

		zValues = append(zValues, z)
		currentActivation = sigmoid(z)
		layerActivations = append(layerActivations, currentActivation)
	}

	// Backward pass
	delta := &mat.Dense{}
	delta.Sub(layerActivations[len(layerActivations)-1], expectedOutputActivations)
	delta.MulElem(delta, sigmoidPrime(zValues[len(zValues)-1]))

	biasDeltas[len(biasDeltas)-1] = delta

	currentWeightDelta := &mat.Dense{}
	currentWeightDelta.Mul(delta, layerActivations[len(layerActivations)-2].T())
	weightDeltas[len(weightDeltas)-1] = currentWeightDelta

	for l := 2; l < len(self.layerSizes)-1; l++ {
		z := zValues[len(zValues)-l]
		sp := sigmoidPrime(z)
		newDelta := &mat.Dense{}
		newDelta.Mul(self.weights[len(self.weights)-l+1].T(), delta)
		newDelta.MulElem(newDelta, sp)

		biasDeltas[len(biasDeltas)-l] = newDelta

		newWeightDelta := &mat.Dense{}
		newWeightDelta.Mul(newDelta, layerActivations[len(layerActivations)-l-1].T())
		weightDeltas[len(weightDeltas)-l] = newWeightDelta
		delta = newDelta
	}

	return weightDeltas, biasDeltas
}

// Return the number of test inputs for which the neural
// network outputs the correct result. Note that the neural
// network's output is assumed to be the index of whichever
// neuron in the final layer has the highest activation
func (self *Network) evaluateTestData(testData []*DataSample) int {
	numCorrectResults := 0

	// Feed each test input through the network
	for testIndex := 0; testIndex < len(testData); testIndex++ {
		result := self.FeedForward(testData[testIndex].InputActivation).ColView(0)

		// Find largest output, largest expected output
		resultIndex := util.MaxIndex(result)
		expectedIndex := util.MaxIndex(testData[testIndex].ExpectedOutputActivation.ColView(0))

		// Tally results that are correct
		if resultIndex == expectedIndex {
			numCorrectResults++
		}
	}

	return numCorrectResults
}

func sigmoid(z *mat.Dense) *mat.Dense {
	result := &mat.Dense{}
	result.Apply(func(_ int, _ int, value float64) float64 {
		return rawSigmoid(value)
	}, z)

	return result
}

func sigmoidPrime(z *mat.Dense) *mat.Dense {
	result := &mat.Dense{}
	result.Apply(func(_ int, _ int, value float64) float64 {
		return rawSigmoid(value) * (1 - rawSigmoid(value))
	}, z)

	return result
}

func rawSigmoid(value float64) float64 {
	return 1.0 / (1.0 + math.Exp(-value))
}
