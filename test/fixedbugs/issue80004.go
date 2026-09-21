// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"unsafe"
)

// 1 of these is 28 bytes.
// When allocating 1 of them in a 32-byte size class,
// we accidentally write 2 pointer bitmasks, which
// marks the first 6 fields of the following object
// as pointers.
type E struct {
	a [7]*byte
}

// 32 bytes, and has pointers.
// First field is not a pointer, but will contain
// a badPtr value.
type Victim struct {
	a uintptr
	b [6]int32
	c *byte
}

//go:noinline
func f(n int) []E {
	var r []E
	for range n {
		r = append(r, E{})
	}
	return r
}

//go:noinline
func newVictim() *Victim {
	return &Victim{a: badPtr}
}

var badPtr uintptr

func init() {
	x := make([]byte, 1<<18-8)
	sink = &x[0]
	badPtr = uintptr(unsafe.Pointer(&x[len(x)-1])) + 5
}

func main() {
	fs := make([]*Victim, 1000)

	// Allocate a bunch of Victims.
	for i := range fs {
		fs[i] = newVictim()
	}
	// Deallocate every other one.
	for i := range fs {
		if i%2 == 1 {
			fs[i] = nil
		}
	}
	runtime.GC()

	// Allocate Es in the deallocated slots.
	// Those allocations will incorrectly set the
	// pointer bit for the first field of all the
	// Victims we allocated.
	for range len(fs) / 2 {
		_ = f(1)
	}
	runtime.GC()

	// Keep fs alive.
	sink = &fs[0]
}

var sink any
