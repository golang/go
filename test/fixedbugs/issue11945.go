// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

// issue 17446
const (
	_ = real(0) // from bug report
	_ = imag(0) // from bug report

	// same as above, but exported for #43891
	Real0 = real(0)
	Imag0 = imag(0)

	// if the arguments are untyped, the results must be untyped
	// (and compatible with types that can represent the values)
	_ int = real(1)
	_ int = real('a')
	_ int = real(2.0)
	_ int = real(3i)

	_ float32 = real(1)
	_ float32 = real('a')
	_ float32 = real(2.1)
	_ float32 = real(3.2i)

	_ float64 = real(1)
	_ float64 = real('a')
	_ float64 = real(2.1)
	_ float64 = real(3.2i)

	_ int = imag(1)
	_ int = imag('a')
	_ int = imag(2.1 + 3i)
	_ int = imag(3i)

	_ float32 = imag(1)
	_ float32 = imag('a')
	_ float32 = imag(2.1 + 3.1i)
	_ float32 = imag(3i)

	_ float64 = imag(1)
	_ float64 = imag('a')
	_ float64 = imag(2.1 + 3.1i)
	_ float64 = imag(3i)
)

var tests = []struct {
	code      string
	got, want interface{}
}{
	{"real(1)", real(1), 1.0},
	{"real('a')", real('a'), float64('a')},
	{"real(2.0)", real(2.0), 2.0},
	{"real(3.2i)", real(3.2i), 0.0},

	{"imag(1)", imag(1), 0.0},
	{"imag('a')", imag('a'), 0.0},
	{"imag(2.1 + 3.1i)", imag(2.1 + 3.1i), 3.1},
	{"imag(3i)", imag(3i), 3.0},
}

func main() {
	// verify compile-time evaluated constant expressions
	for _, test := range tests {
		if test.got != test.want {
			panic(fmt.Sprintf("%s: %v (%T) != %v (%T)", test.code, test.got, test.got, test.want, test.want))
		}
	}
}
