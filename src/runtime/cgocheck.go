// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Code to check that pointer writes follow the cgo rules.
// These functions are invoked when GOEXPERIMENT=cgocheck2 is enabled.

package runtime

import (
	"internal/goarch"
	"unsafe"
)

const cgoWriteBarrierFail = "unpinned Go pointer stored into non-Go memory"

// cgoCheckPtrWrite is called whenever a pointer is stored into memory.
// It throws if the program is storing an unpinned Go pointer into non-Go
// memory.
//
// This is called from generated code when GOEXPERIMENT=cgocheck2 is enabled.
//
//go:nosplit
//go:nowritebarrier
func cgoCheckPtrWrite(dst *unsafe.Pointer, src unsafe.Pointer) {
	if !mainStarted {
		// Something early in startup hates this function.
		// Don't start doing any actual checking until the
		// runtime has set itself up.
		return
	}
	if !cgoIsGoPointer(src) {
		return
	}
	if cgoIsGoPointer(unsafe.Pointer(dst)) {
		return
	}

	// If we are running on the system stack then dst might be an
	// address on the stack, which is OK.
	gp := getg()
	if gp == gp.m.g0 || gp == gp.m.gsignal {
		return
	}

	// Allocating memory can write to various mfixalloc structs
	// that look like they are non-Go memory.
	if gp.m.mallocing != 0 {
		return
	}

	// If the object is pinned, it's safe to store it in C memory. The GC
	// ensures it will not be moved or freed.
	if isPinned(src) {
		return
	}

	// It's OK if writing to memory allocated by persistentalloc.
	// Do this check last because it is more expensive and rarely true.
	// If it is false the expense doesn't matter since we are crashing.
	if inPersistentAlloc(uintptr(unsafe.Pointer(dst))) {
		return
	}

	systemstack(func() {
		println("write of unpinned Go pointer", hex(uintptr(src)), "to non-Go memory", hex(uintptr(unsafe.Pointer(dst))))
		throw(cgoWriteBarrierFail)
	})
}

// cgoCheckMemmove is called when moving a block of memory.
// It throws if the program is copying a block that contains an unpinned Go
// pointer into non-Go memory.
//
// This is called from generated code when GOEXPERIMENT=cgocheck2 is enabled.
//
//go:nosplit
//go:nowritebarrier
func cgoCheckMemmove(typ *_type, dst, src unsafe.Pointer) {
	cgoCheckMemmove2(typ, dst, src, 0, typ.Size_)
}

// cgoCheckMemmove2 is called when moving a block of memory.
// dst and src point off bytes into the value to copy.
// size is the number of bytes to copy.
// It throws if the program is copying a block that contains an unpinned Go
// pointer into non-Go memory.
//
//go:nosplit
//go:nowritebarrier
func cgoCheckMemmove2(typ *_type, dst, src unsafe.Pointer, off, size uintptr) {
	if !typ.Pointers() {
		return
	}
	if !cgoIsGoPointer(src) {
		return
	}
	if cgoIsGoPointer(dst) {
		return
	}
	cgoCheckTypedBlock(typ, src, off, size)
}

// cgoCheckSliceCopy is called when copying n elements of a slice.
// src and dst are pointers to the first element of the slice.
// typ is the element type of the slice.
// It throws if the program is copying slice elements that contain unpinned Go
// pointers into non-Go memory.
//
//go:nosplit
//go:nowritebarrier
func cgoCheckSliceCopy(typ *_type, dst, src unsafe.Pointer, n int) {
	if !typ.Pointers() {
		return
	}
	if !cgoIsGoPointer(src) {
		return
	}
	if cgoIsGoPointer(dst) {
		return
	}
	p := src
	for i := 0; i < n; i++ {
		cgoCheckTypedBlock(typ, p, 0, typ.Size_)
		p = add(p, typ.Size_)
	}
}

// cgoCheckTypedBlock checks the block of memory at src, for up to size bytes,
// and throws if it finds an unpinned Go pointer. The type of the memory is typ,
// and src is off bytes into that type.
//
//go:nosplit
//go:nowritebarrier
func cgoCheckTypedBlock(typ *_type, src unsafe.Pointer, off, size uintptr) {
	// Anything past typ.PtrBytes is not a pointer.
	if typ.PtrBytes <= off {
		return
	}
	if ptrdataSize := typ.PtrBytes - off; size > ptrdataSize {
		size = ptrdataSize
	}

	cgoCheckBits(src, getGCMask(typ), off, size)
}

// cgoCheckBits checks the block of memory at src, for up to size
// bytes, and throws if it finds an unpinned Go pointer. The gcbits mark each
// pointer value. The src pointer is off bytes into the gcbits.
//
//go:nosplit
//go:nowritebarrier
func cgoCheckBits(src unsafe.Pointer, gcbits *byte, off, size uintptr) {
	skipMask := off / goarch.PtrSize / 8
	skipBytes := skipMask * goarch.PtrSize * 8
	ptrmask := addb(gcbits, skipMask)
	src = add(src, skipBytes)
	off -= skipBytes
	size += off
	var bits uint32
	for i := uintptr(0); i < size; i += goarch.PtrSize {
		if i&(goarch.PtrSize*8-1) == 0 {
			bits = uint32(*ptrmask)
			ptrmask = addb(ptrmask, 1)
		} else {
			bits >>= 1
		}
		if off > 0 {
			off -= goarch.PtrSize
		} else {
			if bits&1 != 0 {
				v := *(*unsafe.Pointer)(add(src, i))
				if cgoIsGoPointer(v) && !isPinned(v) {
					throw(cgoWriteBarrierFail)
				}
			}
		}
	}
}

// cgoCheckUsingType is like cgoCheckTypedBlock, but is a last ditch
// fall back to look for pointers in src using the type information.
// We only use this when looking at a value on the stack when the type
// uses a GC program, because otherwise it's more efficient to use the
// GC bits. This is called on the system stack.
//
//go:nowritebarrier
//go:systemstack
func cgoCheckUsingType(typ *_type, src unsafe.Pointer, off, size uintptr) {
	if !typ.Pointers() {
		return
	}

	// Anything past typ.PtrBytes is not a pointer.
	if typ.PtrBytes <= off {
		return
	}
	if ptrdataSize := typ.PtrBytes - off; size > ptrdataSize {
		size = ptrdataSize
	}

	cgoCheckBits(src, getGCMask(typ), off, size)
}
