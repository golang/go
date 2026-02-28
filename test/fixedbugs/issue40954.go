// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

package main

import (
	"runtime/cgo"
	"unsafe"
)

type S struct {
	_ cgo.Incomplete
	x int
}

func main() {
	var i int
	p := (*S)(unsafe.Pointer(uintptr(unsafe.Pointer(&i))))
	v := uintptr(unsafe.Pointer(p))
	// p is a pointer to a not-in-heap type. Like some C libraries,
	// we stored an integer in that pointer. That integer just happens
	// to be the address of i.
	// v is also the address of i.
	// p has a base type which is marked not-in-heap, so it
	// should not be adjusted when the stack is copied.
	recurse(100, p, v)
}

//go:noinline
func recurse(n int, p *S, v uintptr) {
	if n > 0 {
		recurse(n-1, p, v)
	}
	if uintptr(unsafe.Pointer(p)) != v {
		panic("adjusted notinheap pointer")
	}
}
