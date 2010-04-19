// errchk $G -e $D/$F.go

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var (
	f float
	f32 float32
	f64 float64

	c complex
	c64 complex64
	c128 complex128
)
	
func main() {
	// ok
	c = cmplx(f, f)
	c64 = cmplx(f32, f32)
	c128 = cmplx(f64, f64)

	_ = cmplx(f, f32)	// ERROR "cmplx"
	_ = cmplx(f, f64)	// ERROR "cmplx"
	_ = cmplx(f32, f)	// ERROR "cmplx"
	_ = cmplx(f32, f64)	// ERROR "cmplx"
	_ = cmplx(f64, f)	// ERROR "cmplx"
	_ = cmplx(f64, f32)	// ERROR "cmplx"
}
