// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// On s390x, comparing a right-shifted uint64 against a constant in
// [1<<31, 1<<32) was folded to a constant result because the rewrite
// rule sign-extended the 32-bit immediate.

package main

//go:noinline
func gt(x uint64) bool { return x>>6 > 0xffffffff }

func main() {
	if !gt(1 << 40) {
		panic("gt(1<<40) = false, want true")
	}
}
