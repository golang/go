// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

func Sum[T interface{ int | float64 }](vec []T) T {
	var sum T
	for _, elt := range vec {
		sum = sum + elt
	}
	return sum
}

func Abs(f float64) float64 {
	if f < 0.0 {
		return -f
	}
	return f
}

func main() {
	vec1 := []int{3, 4}
	vec2 := []float64{5.8, 9.6}
	got := Sum[int](vec1)
	want := vec1[0] + vec1[1]
	if got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
	got = Sum(vec1)
	if want != got {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	fwant := vec2[0] + vec2[1]
	fgot := Sum[float64](vec2)
	if Abs(fgot-fwant) > 1e-10 {
		panic(fmt.Sprintf("got %f, want %f", fgot, fwant))
	}
	fgot = Sum(vec2)
	if Abs(fgot-fwant) > 1e-10 {
		panic(fmt.Sprintf("got %f, want %f", fgot, fwant))
	}
}
