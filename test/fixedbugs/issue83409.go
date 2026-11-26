// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// The loads from s become constants only during lowering.
const s = "\xff\xff\xff\x7f\x01\x00\x00\x00\x00\x00\x00\x80\xff\xff\xff\xff"

func load(i int) int32 {
	return int32(uint32(s[i]) | uint32(s[i+1])<<8 | uint32(s[i+2])<<16 | uint32(s[i+3])<<24)
}

//go:noinline
func lt() bool {
	return load(0)+load(4) < 0 // math.MaxInt32 + 1 wraps to math.MinInt32
}

//go:noinline
func ge() bool {
	return load(8)+load(12) >= 0 // math.MinInt32 - 1 wraps to math.MaxInt32
}

//go:noinline
func le() bool {
	if load(0)+load(4) <= 0 {
		return true
	}
	return false
}

//go:noinline
func gt() bool {
	if load(8)+load(12) > 0 {
		return true
	}
	return false
}

func main() {
	for i, f := range []func() bool{lt, ge, le, gt} {
		if !f() {
			panic(i)
		}
	}
}
