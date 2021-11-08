// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure FuncForPC won't panic when given a pc which
// lies between two functions.

package main

import (
	"runtime"
)

func main() {
	var stack [1]uintptr
	runtime.Callers(1, stack[:])
	f() // inlined function, to give main some inlining info
	for i := uintptr(0); true; i++ {
		f := runtime.FuncForPC(stack[0] + i)
		if f.Name() != "main.main" && f.Name() != "main.f" {
			// Reached next function successfully.
			break
		}
	}
}

func f() {
	sink = 0 // one instruction which can't be removed
}

var sink int
