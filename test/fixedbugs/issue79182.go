// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 79182: SHLQconst/SHLLconst rewrite rule for (x+x)<<c
// missed a bounds check on c, causing c+1 to overflow the valid
// shift range and producing incorrect results on amd64.

package main

//go:noinline
func shl32(x uint32) uint32 {
	return (x + x) << 31
}

//go:noinline
func shl64(x uint64) uint64 {
	return (x + x) << 63
}

func main() {
	if got := shl32(1); got != 0 {
		println("shl32(1) =", got, "want 0")
		panic("FAIL")
	}
	if got := shl64(1); got != 0 {
		println("shl64(1) =", got, "want 0")
		panic("FAIL")
	}
}
