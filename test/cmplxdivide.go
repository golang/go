// run cmplxdivide1.go

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Driver for complex division table defined in cmplxdivide1.go
// For details, see the comment at the top of cmplxdivide.c.

package main

import (
	"fmt"
	"math"
)

func calike(a, b complex128) bool {
	if imag(a) != imag(b) && !(math.IsNaN(imag(a)) && math.IsNaN(imag(b))) {
		return false
	}

	if real(a) != real(b) && !(math.IsNaN(real(a)) && math.IsNaN(real(b))) {
		return false
	}

	return true
}

func main() {
	bad := false
	for _, t := range tests {
		x := t.f / t.g
		if !calike(x, t.out) {
			if !bad {
				fmt.Printf("BUG\n")
				bad = true
			}
			fmt.Printf("%v/%v: expected %v error; got %v\n", t.f, t.g, t.out, x)
		}
	}
	if bad {
		panic("cmplxdivide failed.")
	}
}
