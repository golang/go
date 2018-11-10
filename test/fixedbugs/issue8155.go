// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8155.
// Alignment of stack prologue zeroing was wrong on 64-bit Native Client
// (because of 32-bit pointers).

package main

import "runtime"

func bad(b bool) uintptr {
	var p **int
	var x1 uintptr
	x1 = 1
	if b {
		var x [11]*int
		p = &x[0]
	}
	if b {
		var x [1]*int
		p = &x[0]
	}
	runtime.GC()
	if p != nil {
		x1 = uintptr(**p)
	}
	return x1
}

func poison() uintptr {
	runtime.GC()
	var x [20]uintptr
	var s uintptr
	for i := range x {
		x[i] = uintptr(i+1)
		s += x[i]
	}
	return s
}

func main() {
	poison()
	bad(false)
}
