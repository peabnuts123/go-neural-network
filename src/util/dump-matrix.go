package util

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Output information about a matrix e.g. size, contents
func DumpMatrix(label string, matrix mat.Matrix) {
	dimR, dimC := matrix.Dims()
	fmt.Printf("[DEBUG] Matrix (%s): %d x %d\n", label, dimR, dimC)
	// fmt.Printf("[DEBUG] %v\n\n", mat.Formatted(matrix))
}
