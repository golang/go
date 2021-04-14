// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that we don't consider a Go'd function's
// arguments as pointers when they aren't.

package main

import (
	"unsafe"
)

var badPtr uintptr

var sink []byte

func init() {
	// Allocate large enough to use largeAlloc.
	b := make([]byte, 1<<16-1)
	sink = b // force heap allocation
	//  Any space between the object and the end of page is invalid to point to.
	badPtr = uintptr(unsafe.Pointer(&b[len(b)-1])) + 1
}

var throttle = make(chan struct{}, 10)

func noPointerArgs(a, b, c, d uintptr) {
	sink = make([]byte, 4096)
	<-throttle
}

func main() {
	const N = 1000
	for i := 0; i < N; i++ {
		throttle <- struct{}{}
		go noPointerArgs(badPtr, badPtr, badPtr, badPtr)
		sink = make([]byte, 4096)
	}
}
