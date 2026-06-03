// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func knownBitsPhiAnd(cond bool) int {
	x := 1
	if cond {
		x = 3
	}
	// amd64:-"AND"
	// arm64:-"AND"
	return x & 1
}

func knownBitsDeferPattern(a, b bool) int {
	bits := 0
	bits |= 1 << 0
	if a {
		bits |= 1 << 1
	}
	bits |= 1 << 2
	if b {
		bits |= 1 << 3
	}
	// amd64:-"AND"
	// arm64:-"AND"
	return bits & (1<<2 | 1<<0)
}

func knownBitsXorToggle(a, b, c bool) int {
	bits := 0
	bits ^= 1 << 0
	if a {
		bits ^= 1 << 1
	}
	bits ^= 1 << 2
	if b {
		bits ^= 1 << 3
	}
	bits ^= 1 << 2
	if c {
		bits ^= 1 << 4
	}
	// amd64:-"AND"
	// arm64:-"AND"
	return bits & (1<<2 | 1<<0)
}
