package util

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Create a Dense with dimensions `dimR`, `dimC`, initialised
// with random data (uses `rand.NormFloat64`)
func RandomMatrix(dimR int, dimC int) *mat.Dense {
	data := make([]float64, dimR*dimC)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	return mat.NewDense(dimR, dimC, data)
}
