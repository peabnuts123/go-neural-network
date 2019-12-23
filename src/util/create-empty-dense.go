package util

import "gonum.org/v1/gonum/mat"

// Create an empty (zeroed) Dense with the same shape as source Dense
func EmptyDense(src *mat.Dense) *mat.Dense {
	dimR, dimC := src.Dims()
	return mat.NewDense(dimR, dimC, nil)
}
