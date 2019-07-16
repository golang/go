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

// There are 2 arg bitmaps for this function, each with 2 bits.
// In the first, p and q are both live, so that bitmap is 11.
// In the second, only p is live, so that bitmap is 10.
// Bitmaps are byte aligned, so if the first bitmap is interpreted as
// extending across the entire argument area, we incorrectly concatenate
// the bitmaps and end up using 110000001. That bad bitmap causes a6
// to be considered a pointer.
func noPointerArgs(p, q *byte, a0, a1, a2, a3, a4, a5, a6 uintptr) {
	sink = make([]byte, 4096)
	sinkptr = q
	<-throttle
	sinkptr = p
}

var sinkptr *byte

func main() {
	const N = 1000
	for i := 0; i < N; i++ {
		throttle <- struct{}{}
		go noPointerArgs(nil, nil, badPtr, badPtr, badPtr, badPtr, badPtr, badPtr, badPtr)
		sink = make([]byte, 4096)
	}
}
