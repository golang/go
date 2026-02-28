// +build cgo,linux,amd64
// run -race

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that CL 281293 doesn't interfere with race detector
// instrumentation.

package main

import "fmt"

const N = 2e6

type Big = [N]int

var sink interface{}

func main() {
	g(0, f(0))

	x1 := f(1)
	sink = &x1
	g(1, x1)
	g(7, f(7))
	g(1, x1)

	x3 := f(3)
	sink = &x3
	g(1, x1)
	g(3, x3)

	h(f(0), x1, f(2), x3, f(4))
}

//go:noinline
func f(k int) (x Big) {
	for i := range x {
		x[i] = k*N + i
	}
	return
}

//go:noinline
func g(k int, x Big) {
	for i := range x {
		if x[i] != k*N+i {
			panic(fmt.Sprintf("x%d[%d] = %d", k, i, x[i]))
		}
	}
}

//go:noinline
func h(x0, x1, x2, x3, x4 Big) {
	g(0, x0)
	g(1, x1)
	g(2, x2)
	g(3, x3)
	g(4, x4)
}
