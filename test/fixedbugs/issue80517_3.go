// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Same time-traveling prove bug as issue80517_2.go, but the victims are a
// signed division and a signed modulo. make([]byte, n) teaches prove that
// n >= 0 only after it runs; a buggy prove lets that fact travel back and
// rewrites the earlier n/4 and n%3 into unsigned operations, corrupting
// the result for negative n.

package main

var sink []byte

//go:noinline
func trigger(n int) (q, r int) {
	defer func() { recover() }()
	if n < 100 {
		q = n / 4              // signed division
		r = n % 3              // signed modulo
		sink = make([]byte, n) // only asserts n >= 0 after this point
	}
	return
}

func main() {
	if q, r := trigger(-8); q != -2 || r != -2 {
		println("n/4 =", q, "want -2;  n%3 =", r, "want -2")
		panic("prove miscompiled a signed div/mod")
	}
}
