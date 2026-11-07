// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// typedef int (*intfn)(int);
// int twice(int);
// int apply(intfn, int);
// void contextFunc(void *);
// int contextCalls(void);
import "C"

import (
	"runtime"
	"unsafe"
)

func init() {
	runtime.SetCgoTraceback(0, nil, unsafe.Pointer(C.contextFunc), nil)
}

// Apply returns twice(x), computed by C through a pointer to twice from Go.
//
//export Apply
func Apply(x C.int) C.int {
	return C.apply(C.intfn(C.twice), x)
}

// ContextCalls reports how many times C called the context function.
//
//export ContextCalls
func ContextCalls() C.int {
	return C.contextCalls()
}

func main() {}
