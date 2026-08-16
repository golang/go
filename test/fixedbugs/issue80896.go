// errorcheck -std

//go:build !(386 || arm || mips || mipsle)

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const N = 2e6

type Big = [4 * 3 * N * N]int

func sink(x Big) {} // ERROR "stack frame too large"

func h(x0, x1, x2 Big) { // ERROR "stack frame too large"
	sink(x0)
	sink(x1)
	sink(x2)
}
