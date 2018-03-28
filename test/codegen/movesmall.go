// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// These tests check that memmoves calls on small data are replaced
// with MOVs

func movesmall4() {
	x := [...]byte{1, 2, 3, 4}
	// 386:-".*memmove"
	// amd64:-".*memmove"
	// arm:-".*memmove"
	// arm64:-".*memmove"
	copy(x[1:], x[:])
}

func movesmall7() {
	x := [...]byte{1, 2, 3, 4, 5, 6, 7}
	// 386:-".*memmove"
	// amd64:-".*memmove"
	// arm64:-".*memmove"
	copy(x[1:], x[:])
}

func movesmall16() {
	x := [...]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	// amd64:-".*memmove"
	copy(x[1:], x[:])
}
