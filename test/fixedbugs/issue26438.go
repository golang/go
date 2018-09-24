// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 26438: arm64 backend may use 64-bit TST for
// "if uint32(a)&uint32(b) == 0", which should be
// 32-bit TSTW

package main

//go:noinline
func tstw(a, b uint64) uint64 {
	if uint32(a)&uint32(b) == 0 {
		return 100
	} else {
		return 200
	}
}

func main() {
	if tstw(0xff00000000, 0xaa00000000) == 200 {
		panic("impossible")
	}
}
