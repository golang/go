// asmcheck

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func i64(a, b int64) int64 { // arm64:`STP\s`,`LDP\s`
	g()
	return a + b
}

func i32(a, b int32) int32 { // arm64:`STPW`,`LDPW`
	g()
	return a + b
}

func f64(a, b float64) float64 { // arm64:`FSTPD`,`FLDPD`
	g()
	return a + b
}

func f32(a, b float32) float32 { // arm64:`FSTPS`,`FLDPS`
	g()
	return a + b
}

//go:noinline
func g() {
}
