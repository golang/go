// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Code to check that pointer writes follow the cgo rules.
// These functions are invoked via the write barrier when debug.cgocheck > 1.

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

const cgoWriteBarrierFail = "Go pointer stored into non-Go memory"

// cgoCheckWriteBarrier is called whenever a pointer is stored into memory.
// It throws if the program is storing a Go pointer into non-Go memory.
//go:nosplit
//go:nowritebarrier
func cgoCheckWriteBarrier(dst *uintptr, src uintptr) {
	if !cgoIsGoPointer(unsafe.Pointer(src)) {
		return
	}
	if cgoIsGoPointer(unsafe.Pointer(dst)) {
		return
	}

	// If we are running on the system stack then dst might be an
	// address on the stack, which is OK.
	g := getg()
	if g == g.m.g0 || g == g.m.gsignal {
		return
	}

	// Allocating memory can write to various mfixalloc structs
	// that look like they are non-Go memory.
	if g.m.mallocing != 0 {
		return
	}

	systemstack(func() {
		println("write of Go pointer", hex(src), "to non-Go memory", hex(uintptr(unsafe.Pointer(dst))))
		throw(cgoWriteBarrierFail)
	})
}

// cgoCheckMemmove is called when moving a block of memory.
// dst and src point off bytes into the value to copy.
// size is the number of bytes to copy.
// It throws if the program is copying a block that contains a Go pointer
// into non-Go memory.
//go:nosplit
//go:nowritebarrier
func cgoCheckMemmove(typ *_type, dst, src unsafe.Pointer, off, size uintptr) {
	if typ.kind&kindNoPointers != 0 {
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

// cgoCheckSliceCopy is called when copying n elements of a slice from
// src to dst.  typ is the element type of the slice.
// It throws if the program is copying slice elements that contain Go pointers
// into non-Go memory.
//go:nosplit
//go:nowritebarrier
func cgoCheckSliceCopy(typ *_type, dst, src slice, n int) {
	if typ.kind&kindNoPointers != 0 {
		return
	}
	if !cgoIsGoPointer(src.array) {
		return
	}
	if cgoIsGoPointer(dst.array) {
		return
	}
	p := src.array
	for i := 0; i < n; i++ {
		cgoCheckTypedBlock(typ, p, 0, typ.size)
		p = add(p, typ.size)
	}
}

// cgoCheckTypedBlock checks the block of memory at src, for up to size bytes,
// and throws if it finds a Go pointer. The type of the memory is typ,
// and src is off bytes into that type.
//go:nosplit
//go:nowritebarrier
func cgoCheckTypedBlock(typ *_type, src unsafe.Pointer, off, size uintptr) {
	// Anything past typ.ptrdata is not a pointer.
	if typ.ptrdata <= off {
		return
	}
	if ptrdataSize := typ.ptrdata - off; size > ptrdataSize {
		size = ptrdataSize
	}

	if typ.kind&kindGCProg == 0 {
		cgoCheckBits(src, typ.gcdata, off, size)
		return
	}

	// The type has a GC program. Try to find GC bits somewhere else.
	for _, datap := range activeModules() {
		if cgoInRange(src, datap.data, datap.edata) {
			doff := uintptr(src) - datap.data
			cgoCheckBits(add(src, -doff), datap.gcdatamask.bytedata, off+doff, size)
			return
		}
		if cgoInRange(src, datap.bss, datap.ebss) {
			boff := uintptr(src) - datap.bss
			cgoCheckBits(add(src, -boff), datap.gcbssmask.bytedata, off+boff, size)
			return
		}
	}

	aoff := uintptr(src) - mheap_.arena_start
	idx := aoff >> _PageShift
	s := mheap_.spans[idx]
	if s.state == _MSpanStack {
		// There are no heap bits for value stored on the stack.
		// For a channel receive src might be on the stack of some
		// other goroutine, so we can't unwind the stack even if
		// we wanted to.
		// We can't expand the GC program without extra storage
		// space we can't easily get.
		// Fortunately we have the type information.
		systemstack(func() {
			cgoCheckUsingType(typ, src, off, size)
		})
		return
	}

	// src must be in the regular heap.

	hbits := heapBitsForAddr(uintptr(src))
	for i := uintptr(0); i < off+size; i += sys.PtrSize {
		bits := hbits.bits()
		if i >= off && bits&bitPointer != 0 {
			v := *(*unsafe.Pointer)(add(src, i))
			if cgoIsGoPointer(v) {
				systemstack(func() {
					throw(cgoWriteBarrierFail)
				})
			}
		}
		hbits = hbits.next()
	}
}

// cgoCheckBits checks the block of memory at src, for up to size
// bytes, and throws if it finds a Go pointer. The gcbits mark each
// pointer value. The src pointer is off bytes into the gcbits.
//go:nosplit
//go:nowritebarrier
func cgoCheckBits(src unsafe.Pointer, gcbits *byte, off, size uintptr) {
	skipMask := off / sys.PtrSize / 8
	skipBytes := skipMask * sys.PtrSize * 8
	ptrmask := addb(gcbits, skipMask)
	src = add(src, skipBytes)
	off -= skipBytes
	size += off
	var bits uint32
	for i := uintptr(0); i < size; i += sys.PtrSize {
		if i&(sys.PtrSize*8-1) == 0 {
			bits = uint32(*ptrmask)
			ptrmask = addb(ptrmask, 1)
		} else {
			bits >>= 1
		}
		if off > 0 {
			off -= sys.PtrSize
		} else {
			if bits&1 != 0 {
				v := *(*unsafe.Pointer)(add(src, i))
				if cgoIsGoPointer(v) {
					systemstack(func() {
						throw(cgoWriteBarrierFail)
					})
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
//go:nowritebarrier
//go:systemstack
func cgoCheckUsingType(typ *_type, src unsafe.Pointer, off, size uintptr) {
	if typ.kind&kindNoPointers != 0 {
		return
	}

	// Anything past typ.ptrdata is not a pointer.
	if typ.ptrdata <= off {
		return
	}
	if ptrdataSize := typ.ptrdata - off; size > ptrdataSize {
		size = ptrdataSize
	}

	if typ.kind&kindGCProg == 0 {
		cgoCheckBits(src, typ.gcdata, off, size)
		return
	}
	switch typ.kind & kindMask {
	default:
		throw("can't happen")
	case kindArray:
		at := (*arraytype)(unsafe.Pointer(typ))
		for i := uintptr(0); i < at.len; i++ {
			if off < at.elem.size {
				cgoCheckUsingType(at.elem, src, off, size)
			}
			src = add(src, at.elem.size)
			skipped := off
			if skipped > at.elem.size {
				skipped = at.elem.size
			}
			checked := at.elem.size - skipped
			off -= skipped
			if size <= checked {
				return
			}
			size -= checked
		}
	case kindStruct:
		st := (*structtype)(unsafe.Pointer(typ))
		for _, f := range st.fields {
			if off < f.typ.size {
				cgoCheckUsingType(f.typ, src, off, size)
			}
			src = add(src, f.typ.size)
			skipped := off
			if skipped > f.typ.size {
				skipped = f.typ.size
			}
			checked := f.typ.size - skipped
			off -= skipped
			if size <= checked {
				return
			}
			size -= checked
		}
	}
}
