// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// Pinner represents a set of pinned Go objects. An object can be pinned with
// the Pin method and all pinned objects of a Pinner can be unpinned with the
// Unpin method.
type Pinner struct {
	*pinner
}

// Pin a Go object. The object will not be moved or freed by the garbage
// collector until the Unpin method has been called. The pointer to a pinned
// object can be directly stored in C memory or can be contained in Go memory
// passed to C functions. If the pinned object iftself contains pointers to Go
// objects, these objects must be pinned separately if they are going to be
// accessed from C code. The argument must be a pointer of any type or an
// unsafe.Pointer. It must be a pointer to an object allocated by calling new,
// by taking the address of a composite literal, or by taking the address of a
// local variable. If one of these conditions is not met, Pin will panic.
func (p *Pinner) Pin(pointer any) {
	if p.pinner == nil {
		p.pinner = new(pinner)
		SetFinalizer(p.pinner, func(i *pinner) {
			if i.refs != nil {
				i.unpin() // only required to make the test idempotent
				pinnerLeakPanic()
			}
		})
	}
	ptr := pinnerGetPtr(&pointer)

	setPinned(ptr, true)
	p.refs = append(p.refs, ptr)
}

// Unpin all pinned objects of the Pinner.
func (p *Pinner) Unpin() {
	p.pinner.unpin()
}

type pinner struct {
	refs []unsafe.Pointer
}

func (p *pinner) unpin() {
	if p == nil || p.refs == nil {
		return
	}
	for i := range p.refs {
		setPinned(p.refs[i], false)
		p.refs[i] = nil
	}
	p.refs = nil
}

func pinnerGetPtr(i *any) unsafe.Pointer {
	e := efaceOf(i)
	etyp := e._type
	if etyp == nil {
		panic(errorString("runtime.Pinner: argument is nil"))
	}
	if kind := etyp.Kind_ & kindMask; kind != kindPtr && kind != kindUnsafePointer {
		panic(errorString("runtime.Pinner: argument is not a pointer: " + toRType(etyp).string()))
	}
	if inUserArenaChunk(uintptr(e.data)) {
		// Arena-allocated objects are not eligible for pinning.
		panic(errorString("runtime.Pinner: object was allocated into an arena"))
	}
	return e.data
}

// isPinned checks if a Go pointer is pinned.
// nosplit, because it's called from nosplit code in cgocheck.
//
//go:nosplit
func isPinned(ptr unsafe.Pointer) bool {
	span := spanOfHeap(uintptr(ptr))
	if span == nil {
		// this code is only called for Go pointer, so this must be a
		// linker-allocated global object.
		return true
	}
	pinnerBits := span.getPinnerBits()
	if pinnerBits == nil {
		return false
	}
	objIndex := span.objIndex(uintptr(ptr))
	bytep := &pinnerBits.x[objIndex/8]
	mask := byte(1 << (objIndex % 8))
	result := (bytep.Load() & mask) != 0
	KeepAlive(ptr) // make sure ptr is alive until we are done so the span can't be freed
	return result
}

// setPinned marks or unmarks a Go pointer as pinned.
func setPinned(ptr unsafe.Pointer, pin bool) {
	span := spanOfHeap(uintptr(ptr))
	if span == nil {
		if isGoPointerWithoutSpan(ptr) {
			// this is a linker-allocated or zero size object, nothing to do.
			return
		}
		panic(errorString("runtime.Pinner.Pin: argument is not a Go pointer"))
	}

	// ensure that the span is swept, b/c sweeping accesses the specials list
	// w/o locks.
	mp := acquirem()
	span.ensureSwept()
	KeepAlive(ptr) // make sure ptr is still alive after span is swept

	objIndex := span.objIndex(uintptr(ptr))
	mask := byte(1 << (objIndex % 8))

	lock(&span.speciallock) // guard against concurrent calls of setPinned on same span

	pinnerBits := span.getPinnerBits()
	if pinnerBits == nil {
		pinnerBits = mheap_.newPinnerBits()
		span.setPinnerBits(pinnerBits)
	}
	bytep := &pinnerBits.x[objIndex/8]
	alreadySet := pin == ((bytep.Load() & mask) != 0)
	if pin {
		if alreadySet {
			// multiple pin on same object, record it in counter
			offset := objIndex * span.elemsize
			// TODO(mknyszek): investigate if systemstack is necessary here
			systemstack(func() {
				span.incPinCounter(offset)
			})
		} else {
			bytep.Or(mask)
		}
	} else {
		if alreadySet {
			// unpinning unpinned object, bail out
			throw("runtime.Pinner: object already unpinned")
		} else {
			multipin := false
			if pinnerBits.specialCnt.Load() != 0 {
				// TODO(mknyszek): investigate if systemstack is necessary here
				systemstack(func() {
					offset := objIndex * span.elemsize
					multipin = span.decPinCounter(offset)
				})
			}
			if !multipin {
				// no multiple pins recoded. unpin object.
				bytep.And(^mask)
			}
		}
	}
	unlock(&span.speciallock)
	releasem(mp)
	return
}

// pinBits is a bitmap for pinned objects. This is always used as pinBits.x.
type pinBits struct {
	_          sys.NotInHeap
	x          [(maxObjsPerSpan + 7) / 8]atomic.Uint8
	specialCnt atomic.Int32
}

func (h *mheap) newPinnerBits() *pinBits {
	lock(&h.speciallock)
	pinnerBits := (*pinBits)(h.pinnerBitsAlloc.alloc())
	unlock(&h.speciallock)
	return pinnerBits
}

func (h *mheap) freePinnerBits(p *pinBits) {
	lock(&h.speciallock)
	h.pinnerBitsAlloc.free(unsafe.Pointer(p))
	unlock(&h.speciallock)
}

// nosplit, because it's called by isPinned, which is nosplit
//
//go:nosplit
func (s *mspan) getPinnerBits() *pinBits {
	return (*pinBits)(atomic.Loadp(unsafe.Pointer(&s.pinnerBits)))
}

func (s *mspan) setPinnerBits(p *pinBits) {
	atomicstorep(unsafe.Pointer(&s.pinnerBits), unsafe.Pointer(p))
}

// incPinCounter is only called for multiple pins of the same object and records
// the _additional_ pins.
func (span *mspan) incPinCounter(offset uintptr) {
	var rec *specialPinCounter

	ref, exists := span.specialFindSplicePoint(offset, _KindSpecialPinCounter)
	if !exists {
		lock(&mheap_.speciallock)
		rec = (*specialPinCounter)(mheap_.specialPinCounterAlloc.alloc())
		unlock(&mheap_.speciallock)
		// splice in record, fill in offset.
		rec.special.offset = uint16(offset)
		rec.special.kind = _KindSpecialPinCounter
		rec.special.next = *ref
		*ref = (*special)(unsafe.Pointer(rec))
		spanHasSpecials(span)
		span.pinnerBits.specialCnt.Add(1)
	} else {
		rec = (*specialPinCounter)(unsafe.Pointer(*ref))
	}
	rec.counter++
}

// decPinCounter is always called for unpins and returns false if no multiple
// pins are recorded. If multiple pins are recorded, it decreases the counter
// and returns true.
func (span *mspan) decPinCounter(offset uintptr) bool {
	ref, exists := span.specialFindSplicePoint(offset, _KindSpecialPinCounter)
	if exists {
		counter := (*specialPinCounter)(unsafe.Pointer(*ref))
		if counter.counter > 1 {
			counter.counter--
		} else {
			span.pinnerBits.specialCnt.Add(-1)
			*ref = counter.special.next
			if span.specials == nil {
				spanHasNoSpecials(span)
			}
			lock(&mheap_.speciallock)
			mheap_.specialPinCounterAlloc.free(unsafe.Pointer(counter))
			unlock(&mheap_.speciallock)
		}
	}
	return exists
}

// only for tests
func pinnerGetPinCounter(addr unsafe.Pointer) *uintptr {
	_, span, objIndex := findObject(uintptr(addr), 0, 0)
	offset := objIndex * span.elemsize
	t, exists := span.specialFindSplicePoint(offset, _KindSpecialPinCounter)
	if !exists {
		return nil
	}
	counter := (*specialPinCounter)(unsafe.Pointer(*t))
	return &counter.counter
}

// to be able to test that the GC panics when a pinned pointer is leaking, this
// panic function is a variable, that can be overwritten by a test.
var pinnerLeakPanic = func() {
	panic(errorString("runtime.Pinner: found leaking pinned pointer; forgot to call Unpin()?"))
}
