// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// #include <complex.h>
import "C"

//export exportComplex64
func exportComplex64(v complex64) complex64 {
	return v
}

//export exportComplex128
func exportComplex128(v complex128) complex128 {
	return v
}

//export exportComplexfloat
func exportComplexfloat(v C.complexfloat) C.complexfloat {
	return v
}

//export exportComplexdouble
func exportComplexdouble(v C.complexdouble) C.complexdouble {
	return v
}

func main() {}
