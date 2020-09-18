// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
)

var nilp *int
var forceHeap interface{}

func main() {
	// x is a pointer on the stack to heap-allocated memory.
	x := new([32]*int)
	forceHeap = x
	forceHeap = nil

	// Push a defer to be run when we panic below.
	defer func() {
		// Ignore the panic.
		recover()
		// Force a stack walk. Go 1.11 will fail because x is now
		// considered live again.
		runtime.GC()
	}()
	// Make x live at the defer's PC.
	runtime.KeepAlive(x)

	// x is no longer live. Garbage collect the [32]*int on the
	// heap.
	runtime.GC()
	// At this point x's dead stack slot points to dead memory.

	// Trigger a sigpanic. Since this is an implicit panic, we
	// don't have an explicit liveness map here.
	// Traceback used to use the liveness map of the most recent defer,
	// but in that liveness map, x will be live again even though
	// it points to dead memory. The fix is to use the liveness
	// map of a deferreturn call instead.
	*nilp = 0
}
