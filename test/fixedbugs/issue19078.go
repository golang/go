// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 19078: liveness & zero-initialization of results
// when there is a defer.
package main

import "unsafe"

func main() {
	// Construct an invalid pointer.  We do this by
	// making a pointer which points to the unused space
	// between the last 48-byte object in a span and the
	// end of the span (there are 32 unused bytes there).
	p := new([48]byte)              // make a 48-byte object
	sink = &p                       // escape it, so it allocates for real
	u := uintptr(unsafe.Pointer(p)) // get its address
	u = u >> 13 << 13               // round down to page size
	u += 1<<13 - 1                  // add almost a page

	for i := 0; i < 1000000; i++ {
		_ = identity(u)         // installs u at return slot
		_ = liveReturnSlot(nil) // incorrectly marks return slot as live
	}
}

//go:noinline
func liveReturnSlot(x *int) *int {
	defer func() {}() // causes return slot to be marked live
	sink = &x         // causes x to be moved to the heap, triggering allocation
	return x
}

//go:noinline
func identity(x uintptr) uintptr {
	return x
}

var sink interface{}
