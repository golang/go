// errchk $G -e $D/$F.go

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var (
	f32 float32
	f64 float64

	c64  complex64
	c128 complex128
)

func main() {
	// ok
	c64 = complex(f32, f32)
	c128 = complex(f64, f64)

	_ = complex128(0)     // ok
	_ = complex(f32, f64) // ERROR "complex"
	_ = complex(f64, f32) // ERROR "complex"
}
