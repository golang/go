// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// These functions are called from C code via cgo/callbacks.c.

// Allocate memory.  This allocates the requested number of bytes in
// memory controlled by the Go runtime.  The allocated memory will be
// zeroed.  You are responsible for ensuring that the Go garbage
// collector can see a pointer to the allocated memory for as long as
// it is valid, e.g., by storing a pointer in a local variable in your
// C function, or in memory allocated by the Go runtime.  If the only
// pointers are in a C global variable or in memory allocated via
// malloc, then the Go garbage collector may collect the memory.
//
// TODO(rsc,iant): This memory is untyped.
// Either we need to add types or we need to stop using it.

func _cgo_allocate_internal(len uintptr) unsafe.Pointer {
	if len == 0 {
		len = 1
	}
	ret := unsafe.Pointer(&make([]unsafe.Pointer, (len+ptrSize-1)/ptrSize)[0])
	c := new(cgomal)
	c.alloc = ret
	gp := getg()
	c.next = gp.m.cgomal
	gp.m.cgomal = c
	return ret
}

// Panic.

func _cgo_panic_internal(p *byte) {
	panic(gostringnocopy(p))
}
