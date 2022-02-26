// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test conversion operations.

package main

func left(x int)  { _ = 1 << x }
func right(x int) { _ = 1 >> x }

func main() {
	wantPanic(
		func() {
			left(-1)
		},
		"runtime error: negative shift amount",
	)
	wantPanic(
		func() {
			right(-1)
		},
		"runtime error: negative shift amount",
	)
	wantPanic(
		func() {
			const maxInt32 = 1<<31 - 1
			var idx int64 = maxInt32*2 + 8
			x := make([]int, 16)
			_ = x[idx]
		},
		"runtime error: runtime error: index out of range [4294967302] with length 16",
	)
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
