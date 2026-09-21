// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The prove pass must not use a fact that only becomes valid after a
// later value executes to simplify an earlier value. Here make([]byte, n)
// teaches prove that n >= 0, but that is only true after the make runs.
// A buggy prove lets that fact travel back in time and rewrites the
// earlier signed shift n>>1 into an unsigned shift, corrupting the result
// for negative n.

package main

var sink []byte

//go:noinline
func trigger(n int) (res int) {
	defer func() { recover() }()
	if n < 100 {
		res = n >> 1           // signed arithmetic shift right
		sink = make([]byte, n) // only asserts n >= 0 after this point
	}
	return
}

func main() {
	if got := trigger(-2); got != -1 {
		println("n>>1 =", got, "want -1")
		panic("prove miscompiled a signed shift")
	}
}
