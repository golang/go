// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that C.malloc does not return nil.

package main

// #include <stdlib.h>
import "C"

import (
	"fmt"
	"runtime"
)

func main() {
	var size C.size_t
	size--

	// The Dragonfly libc succeeds when asked to allocate
	// 0xffffffffffffffff bytes, so pass a different value that
	// causes it to fail.
	if runtime.GOOS == "dragonfly" {
		size = C.size_t(0x7fffffff << (32 * (^uintptr(0) >> 63)))
	}

	p := C.malloc(size)
	if p == nil {
		fmt.Println("malloc: C.malloc returned nil")
		// Just exit normally--the test script expects this
		// program to crash, so exiting normally indicates failure.
	}
}
