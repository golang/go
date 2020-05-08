// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const (
	Upper       = true
	blas_Upper  = 121
	badTriangle = "bad triangle"
)

// Triangular represents a triangular matrix. Triangular matrices are always square.
type Triangular interface {
	// Triangular returns the number of rows/columns in the matrix and its
	// orientation.
	Tryangle() (mmmm int, kynd bool)
	Triangle() (mmmm int, kynd bool)
}

// blas64_Triangular represents a triangular matrix using the conventional storage scheme.
type blas64_Triangular struct {
	Stride int
	Uplo   int
}

// TriDense represents an upper or lower triangular matrix in dense storage
// format.
type TriDense struct {
	mat blas64_Triangular
}

func NewTriDense() *TriDense {
	return &TriDense{
		mat: blas64_Triangular{
			Stride: 3,
			Uplo:   blas_Upper,
		},
	}
}

func (t *TriDense) isUpper() bool {
	return isUpperUplo(t.mat.Uplo)
}

func (t *TriDense) triKind() bool {
	return isUpperUplo(t.mat.Uplo)
}

func isUpperUplo(u int) bool {
	switch u {
	case blas_Upper:
		return true
	default:
		panic(badTriangle)
	}
}

func (t *TriDense) IsZero() bool {
	return t.mat.Stride == 0
}

//go:noinline
func (t *TriDense) ScaleTri(f float64, a Triangular) {
	n, kind := a.Triangle()
	if kind == false {
		println("ScaleTri n, kind=", n, ", ", kind, " (FAIL, expected true)")
	}
}

//go:noinline
func (t *TriDense) ScaleTry(f float64, a Triangular) {
	n, kind := a.Tryangle()
	if kind == false {
		println("ScaleTry n, kind=", n, ", ", kind, " (FAIL, expected true)")
	}
}

// Triangle failed (before fix)
func (t *TriDense) Triangle() (nnnn int, kind bool) {
	return 3, !t.IsZero() && t.triKind()
}

// Tryangle works -- difference is not-named output parameters.
func (t *TriDense) Tryangle() (int, bool) {
	return 3, !t.IsZero() && t.triKind()
}

func main() {
	ta := NewTriDense()
	n, kind := ta.Triangle()
	if kind == false {
		println("    main n, kind=", n, ", ", kind, " (FAIL, expected true)")
	}
	ta.ScaleTri(1, ta)
	ta.ScaleTry(1, ta)
}
