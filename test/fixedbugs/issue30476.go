// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 30476: KeepAlive didn't keep stack object alive.

package main

import "runtime"

func main() {
	x := new([10]int)
	runtime.SetFinalizer(x, func(*[10]int) { panic("FAIL: finalizer runs") })
	p := &T{x, 0}
	use(p)
	runtime.GC()
	runtime.GC()
	runtime.GC()
	runtime.KeepAlive(p)
}

type T struct {
	x *[10]int
	y int
}

//go:noinline
func use(*T) {}
