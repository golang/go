// asmcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure we reuse large constant loads, if we can.
// See issue 33580.

package codegen

const (
	A = 7777777777777777
	B = 8888888888888888
)

func f(x, y uint64) uint64 {
	p := x & A
	q := y & A
	r := x & B
	// amd64:-"MOVQ.*8888888888888888"
	s := y & B

	return p * q * r * s
}
