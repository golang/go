// errorcheck -0 -m

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test b.Loop escape analysis behavior.

package bloop

import (
	"testing"
)

// An example where mid-stack inlining allows stack allocation of a slice.
// This is from the example in go.dev/issue/73137.

func NewX(x int) []byte { // ERROR "can inline NewX"
	out := make([]byte, 8) // ERROR "make\(\[\]byte, 8\) escapes to heap"
	return use1(out)
}

//go:noinline
func use1(out []byte) []byte { // ERROR "leaking param: out to result ~r0 level=0"
	return out
}

//go:noinline
func BenchmarkBloop(b *testing.B) { // ERROR "leaking param: b"
	for b.Loop() { // ERROR "inlining call to testing.\(\*B\).Loop"
		NewX(42) // ERROR "make\(\[\]byte, 8\) does not escape" "inlining call to NewX"
	}
}

// A traditional b.N benchmark using a sink variable for comparison,
// also from the example in go.dev/issue/73137.

var sink byte

//go:noinline
func BenchmarkBN(b *testing.B) { // ERROR "b does not escape"
	for i := 0; i < b.N; i++ {
		out := NewX(42) // ERROR "make\(\[\]byte, 8\) does not escape" "inlining call to NewX"
		sink = out[0]
	}
}

// An example showing behavior of a simple function argument in the b.Loop body.

//go:noinline
func use2(x any) {} // ERROR "x does not escape"

//go:noinline
func BenchmarkBLoopFunctionArg(b *testing.B) { // ERROR "leaking param: b"
	for b.Loop() { // ERROR "inlining call to testing.\(\*B\).Loop"
		use2(42) // ERROR "42 does not escape"
	}
}

// A similar call outside of b.Loop for comparison.

func simpleFunctionArg() { // ERROR "can inline simpleFunctionArg"
	use2(42) // ERROR "42 does not escape"
}
