// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

func add[T interface{ type int, float64 }](vec []T) T {
	var sum T
	for _, elt := range vec {
		sum = sum + elt
	}
	return sum
}

func abs(f float64) float64 {
	if f < 0.0 {
		return -f
	}
	return f
}

func main() {
	vec1 := []int{3, 4}
	vec2 := []float64{5.8, 9.6}
	want := vec1[0] + vec1[1]
	got := add[int](vec1)
	if want != got {
		panic(fmt.Sprintf("Want %d, got %d", want, got))
	}
	got = add(vec1)
	if want != got {
		panic(fmt.Sprintf("Want %d, got %d", want, got))
	}

	fwant := vec2[0] + vec2[1]
	fgot := add[float64](vec2)
	if abs(fgot - fwant) > 1e-10 {
		panic(fmt.Sprintf("Want %f, got %f", fwant, fgot))
	}
	fgot = add(vec2)
	if abs(fgot - fwant) > 1e-10 {
		panic(fmt.Sprintf("Want %f, got %f", fwant, fgot))
	}
}
