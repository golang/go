// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that GC data is generated correctly for global
// variables with types defined in a shared library.
// See issue 39927.

// This test run under GODEBUG=clobberfree=1. The check
// *x[i] == 12345 depends on this debug mode to clobber
// the value if the object is freed prematurely.

package main

import (
	"fmt"
	"runtime"
	"testshared/gcdata/p"
)

var x p.T

func main() {
	for i := range x {
		x[i] = new(int)
		*x[i] = 12345
	}
	runtime.GC()
	runtime.GC()
	runtime.GC()
	for i := range x {
		if *x[i] != 12345 {
			fmt.Printf("x[%d] == %d, want 12345\n", i, *x[i])
			panic("FAIL")
		}
	}
}
