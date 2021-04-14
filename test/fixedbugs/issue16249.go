// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Liveness calculations were wrong for a result parameter pushed onto
// the heap in a function that used defer.  Program would crash with
//     runtime: bad pointer in frame main.A at 0xc4201e6838: 0x1

package main

import "errors"

var sink interface{}

//go:noinline
func f(err *error) {
	if err != nil {
		sink = err
	}
}

//go:noinline
func A(n, m int64) (res int64, err error) {
	defer f(&err) // output parameter's address escapes to a defer.
	if n < 0 {
		err = errors.New("No negative")
		return
	}
	if n <= 1 {
		res = n
		return
	}
	res = B(m) // This call to B drizzles a little junk on the stack.
	res, err = A(n-1, m)
	res++
	return
}

// B does a little bit of recursion dribbling not-zero onto the stack.
//go:noinline
func B(n int64) (res int64) {
	if n <= 1 { // Prefer to leave a 1 on the stack.
		return n
	}
	return 1 + B(n-1)
}

func main() {
	x, e := A(0, 0)
	for j := 0; j < 4; j++ { // j controls amount of B's stack dribble
		for i := 0; i < 1000; i++ { // try more and more recursion until stack growth occurs in newobject in prologue
			x, e = A(int64(i), int64(j))
		}
	}
	_, _ = x, e
}
