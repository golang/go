// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test interpretation on 32 bit widths.

package main

func main() {
	mapSize()
}

func mapSize() {
	// Tests for the size argument of make on a map type.
	const tooBigFor32 = 1<<33 - 1
	wantPanic(
		func() {
			_ = make(map[int]int, int64(tooBigFor32))
		},
		"runtime error: ssa.MakeMap.Reserve value 8589934591 does not fit in int",
	)

	// TODO: Enable the following if sizeof(int) can be different for host and target.
	// _ = make(map[int]int, tooBigFor32)
	//
	// Second arg to make in `make(map[int]int, tooBigFor32)` is an untyped int and
	// is converted into an int explicitly in ssa.
	// This has a different value on 32 and 64 bit systems.
}

func wantPanic(fn func(), s string) {
	defer func() {
		err := recover()
		if err == nil {
			panic("expected panic")
		}
		if got := err.(error).Error(); got != s {
			panic("expected panic " + s + " got " + got)
		}
	}()
	fn()
}
