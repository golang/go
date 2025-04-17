// run -gcflags=-d=softfloat

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

// When using soft-float, OMUL might be rewritten to function
// call so we should ensure it was evaluated first. Stack frame
// setup for "test" function call should happen after call to runtime.fmul32

var x int32 = 1

func main() {
	var y float32 = 1.0
	test(x, y*y)
}

//go:noinline
func test(id int32, a float32) {

	if id != x {
		fmt.Printf("got: %d, want: %d\n", id, x)
		panic("FAIL")
	}
}
