// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: bug120

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "os"
import "strconv"

type Test struct {
	f   float64
	in  string
	out string
}

var tests = []Test{
	Test{123.5, "123.5", "123.5"},
	Test{456.7, "456.7", "456.7"},
	Test{1e23 + 8.5e6, "1e23+8.5e6", "1.0000000000000001e+23"},
	Test{100000000000000008388608, "100000000000000008388608", "1.0000000000000001e+23"},
	Test{1e23 + 8388609, "1e23+8388609", "1.0000000000000001e+23"},

	// "x" = the floating point value from converting the string x.
	// These are exactly representable in 64-bit floating point:
	//	1e23-8388608
	//	1e23+8388608
	// The former has an even mantissa, so "1e23" rounds to 1e23-8388608.
	// If "1e23+8388608" is implemented as "1e23" + "8388608",
	// that ends up computing 1e23-8388608 + 8388608 = 1e23,
	// which rounds back to 1e23-8388608.
	// The correct answer, of course, would be "1e23+8388608" = 1e23+8388608.
	// This is not going to be correct until 6g has multiprecision floating point.
	// A simpler case is "1e23+1", which should also round to 1e23+8388608.
	Test{1e23 + 8.388608e6, "1e23+8.388608e6", "1.0000000000000001e+23"},
	Test{1e23 + 1, "1e23+1", "1.0000000000000001e+23"},
}

func main() {
	ok := true
	for i := 0; i < len(tests); i++ {
		t := tests[i]
		v := strconv.Ftoa64(t.f, 'g', -1)
		if v != t.out {
			println("Bad float64 const:", t.in, "want", t.out, "got", v)
			x, err := strconv.Atof64(t.out)
			if err != nil {
				println("bug120: strconv.Atof64", t.out)
				panic("fail")
			}
			println("\twant exact:", strconv.Ftoa64(x, 'g', 1000))
			println("\tgot exact: ", strconv.Ftoa64(t.f, 'g', 1000))
			ok = false
		}
	}
	if !ok {
		os.Exit(1)
	}
}
