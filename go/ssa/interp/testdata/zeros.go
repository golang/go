// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test interpretation on zero values with type params.
package zeros

func assert(cond bool, msg string) {
	if !cond {
		panic(msg)
	}
}

func tp0[T int | string | float64]() T { return T(0) }

func tpFalse[T ~bool]() T { return T(false) }

func tpEmptyString[T string | []byte]() T { return T("") }

func tpNil[T *int | []byte]() T { return T(nil) }

func main() {
	// zero values
	var zi int
	var zf float64
	var zs string

	assert(zi == int(0), "zero value of int is int(0)")
	assert(zf == float64(0), "zero value of float64 is float64(0)")
	assert(zs != string(0), "zero value of string is not string(0)")

	assert(zi == tp0[int](), "zero value of int is int(0)")
	assert(zf == tp0[float64](), "zero value of float64 is float64(0)")
	assert(zs != tp0[string](), "zero value of string is not string(0)")

	assert(zf == -0.0, "constant -0.0 is converted to 0.0")

	assert(!tpFalse[bool](), "zero value of bool is false")

	assert(tpEmptyString[string]() == zs, `zero value of string is string("")`)
	assert(len(tpEmptyString[[]byte]()) == 0, `[]byte("") is empty`)

	assert(tpNil[*int]() == nil, "nil is nil")
	assert(tpNil[[]byte]() == nil, "nil is nil")
}
