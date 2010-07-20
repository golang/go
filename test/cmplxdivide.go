// $G $D/$F.go $D/cmplxdivide1.go && $L $D/$F.$A && ./$A.out

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Driver for complex division table defined in cmplxdivide1.go

package main

import (
	"cmath"
	"fmt"
	"math"
)

type Test struct{
	f, g	complex128
	out	complex128
}

var nan = math.NaN()
var inf = math.Inf(1)
var negzero = math.Copysign(0, -1)

func calike(a, b complex128) bool {
	switch {
	case cmath.IsInf(a) && cmath.IsInf(b):
		return true
	case cmath.IsNaN(a) && cmath.IsNaN(b):
		return true
	}
	return a == b
}

func main() {
	bad := false
	for _, t := range tests {
		x := t.f/t.g
		if !calike(x, t.out) {
			if !bad {
				fmt.Printf("BUG\n")
				bad = true
			}
			fmt.Printf("%v/%v: expected %v error; got %v\n", t.f, t.g, t.out, x)
		}
	}
}
