// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 16948: make sure intrinsified atomic ops won't
// confuse the scheduler.

package main

import "sync/atomic"

func main() {
	f()
}

var x int32

type T [10]int
var sink *T

func f() (t T) {
	atomic.AddInt32(&x, 1)
	g(42, 42, 42, 42, 42, &t) // use int values that is invalid pointer to smash the stack slot of return value of runtime.newobject
	return
}

//go:noinline
func g(a, b, c, d, e int, p *T) {
	var t [10000]int // a large stack frame to trigger stack growing
	_ = t
	sink = p // force p (in caller) heap allocated
}
