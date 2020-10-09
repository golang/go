// +build amd64
// asmcheck -gcflags=-spectre=index

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func IndexArray(x *[10]int, i int) int {
	// amd64:`CMOVQCC`
	return x[i]
}

func IndexString(x string, i int) byte {
	// amd64:`CMOVQCC`
	return x[i]
}

func IndexSlice(x []float64, i int) float64 {
	// amd64:`CMOVQCC`
	return x[i]
}

func SliceArray(x *[10]int, i, j int) []int {
	// amd64:`CMOVQHI`
	return x[i:j]
}

func SliceString(x string, i, j int) string {
	// amd64:`CMOVQHI`
	return x[i:j]
}

func SliceSlice(x []float64, i, j int) []float64 {
	// amd64:`CMOVQHI`
	return x[i:j]
}
