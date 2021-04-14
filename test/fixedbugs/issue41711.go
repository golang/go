// compile -d=ssa/check/on

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f() uint32 {
	s := "food"
	x := uint32(s[0]) + uint32(s[1])<<8 + uint32(s[2])<<16 + uint32(s[3])<<24
	// x is a constant, but that's not known until lowering.
	// shifting it by 8 moves the high byte up into the high 32 bits of
	// a 64-bit word. That word is not properly sign-extended by the faulty
	// rule, which causes the compiler to fail.
	return x << 8
}
