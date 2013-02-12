// run cmplxdivide1.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Driver for complex division table defined in cmplxdivide1.go

package main

import (
	"fmt"
	"math"
	"math/cmplx"
)

type Test struct {
	f, g complex128
	out  complex128
}

var nan = math.NaN()
var inf = math.Inf(1)
var negzero = math.Copysign(0, -1)

func calike(a, b complex128) bool {
	switch {
	case cmplx.IsInf(a) && cmplx.IsInf(b):
		return true
	case cmplx.IsNaN(a) && cmplx.IsNaN(b):
		return true
	}
	return a == b
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
