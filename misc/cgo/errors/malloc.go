// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that C.malloc does not return nil.

package main

// #include <stdlib.h>
import "C"

import (
	"fmt"
)

func main() {
	p := C.malloc(C.size_t(^uintptr(0)))
	if p == nil {
		fmt.Println("malloc: C.malloc returned nil")
		// Just exit normally--the test script expects this
		// program to crash, so exiting normally indicates failure.
	}
}
