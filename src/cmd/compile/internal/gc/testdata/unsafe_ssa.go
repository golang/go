// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
	"unsafe"
)

// global pointer slot
var a *[8]uint

// unfoldable true
var b = true

// Test to make sure that a pointer value which is alive
// across a call is retained, even when there are matching
// conversions to/from uintptr around the call.
// We arrange things very carefully to have to/from
// conversions on either side of the call which cannot be
// combined with any other conversions.
func f_ssa() *[8]uint {
	// Make x a uintptr pointing to where a points.
	var x uintptr
	if b {
		x = uintptr(unsafe.Pointer(a))
	} else {
		x = 0
	}
	// Clobber the global pointer. The only live ref
	// to the allocated object is now x.
	a = nil

	// Convert to pointer so it should hold
	// the object live across GC call.
	p := unsafe.Pointer(x)

	// Call gc.
	runtime.GC()

	// Convert back to uintptr.
	y := uintptr(p)

	// Mess with y so that the subsequent cast
	// to unsafe.Pointer can't be combined with the
	// uintptr cast above.
	var z uintptr
	if b {
		z = y
	} else {
		z = 0
	}
	return (*[8]uint)(unsafe.Pointer(z))
}

// g_ssa is the same as f_ssa, but with a bit of pointer
// arithmetic for added insanity.
func g_ssa() *[7]uint {
	// Make x a uintptr pointing to where a points.
	var x uintptr
	if b {
		x = uintptr(unsafe.Pointer(a))
	} else {
		x = 0
	}
	// Clobber the global pointer. The only live ref
	// to the allocated object is now x.
	a = nil

	// Offset x by one int.
	x += unsafe.Sizeof(int(0))

	// Convert to pointer so it should hold
	// the object live across GC call.
	p := unsafe.Pointer(x)

	// Call gc.
	runtime.GC()

	// Convert back to uintptr.
	y := uintptr(p)

	// Mess with y so that the subsequent cast
	// to unsafe.Pointer can't be combined with the
	// uintptr cast above.
	var z uintptr
	if b {
		z = y
	} else {
		z = 0
	}
	return (*[7]uint)(unsafe.Pointer(z))
}

func testf() {
	a = new([8]uint)
	for i := 0; i < 8; i++ {
		a[i] = 0xabcd
	}
	c := f_ssa()
	for i := 0; i < 8; i++ {
		if c[i] != 0xabcd {
			fmt.Printf("%d:%x\n", i, c[i])
			panic("bad c")
		}
	}
}

func testg() {
	a = new([8]uint)
	for i := 0; i < 8; i++ {
		a[i] = 0xabcd
	}
	c := g_ssa()
	for i := 0; i < 7; i++ {
		if c[i] != 0xabcd {
			fmt.Printf("%d:%x\n", i, c[i])
			panic("bad c")
		}
	}
}

func alias_ssa(ui64 *uint64, ui32 *uint32) uint32 {
	*ui32 = 0xffffffff
	*ui64 = 0                  // store
	ret := *ui32               // load from same address, should be zero
	*ui64 = 0xffffffffffffffff // store
	return ret
}
func testdse() {
	x := int64(-1)
	// construct two pointers that alias one another
	ui64 := (*uint64)(unsafe.Pointer(&x))
	ui32 := (*uint32)(unsafe.Pointer(&x))
	if want, got := uint32(0), alias_ssa(ui64, ui32); got != want {
		fmt.Printf("alias_ssa: wanted %d, got %d\n", want, got)
		panic("alias_ssa")
	}
}

func main() {
	testf()
	testg()
	testdse()
}
