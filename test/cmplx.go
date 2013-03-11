// errorcheck

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that incorrect invocations of the complex predeclared function are detected.
// Does not compile.

package main

type (
	Float32    float32
	Float64    float64
	Complex64  complex64
	Complex128 complex128
)

var (
	f32 float32
	f64 float64
	F32 Float32
	F64 Float64

	c64  complex64
	c128 complex128
	C64  Complex64
	C128 Complex128
)

func main() {
	// ok
	c64 = complex(f32, f32)
	c128 = complex(f64, f64)

	_ = complex128(0)     // ok
	_ = complex(f32, f64) // ERROR "complex"
	_ = complex(f64, f32) // ERROR "complex"
	_ = complex(f32, F32) // ERROR "complex"
	_ = complex(F32, f32) // ERROR "complex"
	_ = complex(f64, F64) // ERROR "complex"
	_ = complex(F64, f64) // ERROR "complex"

	c128 = complex(f32, f32) // ERROR "cannot use"
	c64 = complex(f64, f64)  // ERROR "cannot use"

	c64 = complex(1.0, 2.0) // ok, constant is untyped
	c128 = complex(1.0, 2.0)
	C64 = complex(1.0, 2.0)
	C128 = complex(1.0, 2.0)

	C64 = complex(f32, f32)  // ERROR "cannot use"
	C128 = complex(f64, f64) // ERROR "cannot use"
}
