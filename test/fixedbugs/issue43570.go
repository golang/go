// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type T [8]*int

//go:noinline
func f(x int) T {
	return T{}
}

//go:noinline
func g(x int, t T) {
	if t != (T{}) {
		panic(fmt.Sprintf("bad: %v", t))
	}
}

func main() {
	const N = 10000
	var q T
	func() {
		for i := 0; i < N; i++ {
			q = f(0)
			g(0, q)
			sink = make([]byte, 1024)
		}
	}()
	// Note that the closure is a trick to get the write to q to be a
	// write to a pointer that is known to be non-nil and requires
	// a write barrier.
}

var sink []byte
