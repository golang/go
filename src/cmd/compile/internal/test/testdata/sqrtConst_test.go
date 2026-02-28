// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"testing"
)

var tests = [...]struct {
	name string
	in   float64 // used for error messages, not an input
	got  float64
	want float64
}{
	{"sqrt0", 0, math.Sqrt(0), 0},
	{"sqrt1", 1, math.Sqrt(1), 1},
	{"sqrt2", 2, math.Sqrt(2), math.Sqrt2},
	{"sqrt4", 4, math.Sqrt(4), 2},
	{"sqrt100", 100, math.Sqrt(100), 10},
	{"sqrt101", 101, math.Sqrt(101), 10.04987562112089},
}

var nanTests = [...]struct {
	name string
	in   float64 // used for error messages, not an input
	got  float64
}{
	{"sqrtNaN", math.NaN(), math.Sqrt(math.NaN())},
	{"sqrtNegative", -1, math.Sqrt(-1)},
	{"sqrtNegInf", math.Inf(-1), math.Sqrt(math.Inf(-1))},
}

func TestSqrtConst(t *testing.T) {
	for _, test := range tests {
		if test.got != test.want {
			t.Errorf("%s: math.Sqrt(%f): got %f, want %f\n", test.name, test.in, test.got, test.want)
		}
	}
	for _, test := range nanTests {
		if math.IsNaN(test.got) != true {
			t.Errorf("%s: math.Sqrt(%f): got %f, want NaN\n", test.name, test.in, test.got)
		}
	}
	if got := math.Sqrt(math.Inf(1)); !math.IsInf(got, 1) {
		t.Errorf("math.Sqrt(+Inf), got %f, want +Inf\n", got)
	}
}
