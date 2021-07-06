// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A test for partial liveness / partial spilling / compiler-induced GC failure

package main

import "runtime"
import "unsafe"

//go:registerparams
func F(s []int) {
	for i, x := range s {
		G(i, x)
	}
	GC()
	H(&s[0]) // It's possible that this will make the spill redundant, but there's a bug in spill slot allocation.
	G(len(s), cap(s))
	GC()
}

//go:noinline
//go:registerparams
func G(int, int) {}

//go:noinline
//go:registerparams
func H(*int) {}

//go:registerparams
func GC() { runtime.GC(); runtime.GC() }

func main() {
	s := make([]int, 3)
	escape(s)
	p := int(uintptr(unsafe.Pointer(&s[2])) + 42) // likely point to unallocated memory
	poison([3]int{p, p, p})
	F(s)
}

//go:noinline
//go:registerparams
func poison([3]int) {}

//go:noinline
//go:registerparams
func escape(s []int) {
	g = s
}
var g []int
