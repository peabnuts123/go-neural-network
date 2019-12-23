package util

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Find the index of the largest value in a vector
func MaxIndex(vec mat.Vector) int {
	max := math.Inf(-1)
	maxIndex := -1

	for i := 0; i < vec.Len(); i++ {
		value := vec.AtVec(i)
		if value > max {
			max = value
			maxIndex = i
		}
	}

	return maxIndex
}
