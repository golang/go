// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7944:
// Liveness bitmaps said b was live at call to g,
// but no one told the register optimizer.

package main

import "runtime"

func f(b []byte) {
	for len(b) > 0 {
		n := len(b)
		n = f1(n)
		f2(b[n:])
		b = b[n:]
	}
	g()
}

func f1(n int) int {
	runtime.GC()
	return n
}

func f2(b []byte) {
	runtime.GC()
}

func g() {
	runtime.GC()
}

func main() {
	f(make([]byte, 100))
}
