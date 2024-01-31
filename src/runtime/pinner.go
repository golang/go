// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"unsafe"
)

// A Pinner is a set of Go objects each pinned to a fixed location in memory. The
// [Pinner.Pin] method pins one object, while [Pinner.Unpin] unpins all pinned
// objects. See their comments for more information.
type Pinner struct {
	*pinner
}

// Pin pins a Go object, preventing it from being moved or freed by the garbage
// collector until the [Pinner.Unpin] method has been called.
//
// A pointer to a pinned object can be directly stored in C memory or can be
// contained in Go memory passed to C functions. If the pinned object itself
// contains pointers to Go objects, these objects must be pinned separately if they
// are going to be accessed from C code.
//
// The argument must be a pointer of any type, an [unsafe.Pointer] ,
// a string, or a slice of any type.
// It's safe to call Pin on non-Go pointers, in which case Pin will do nothing.
func (p *Pinner) Pin(pointer any) {
	if p.pinner == nil {
		// Check the pinner cache first.
		mp := acquirem()
		if pp := mp.p.ptr(); pp != nil {
			p.pinner = pp.pinnerCache
			pp.pinnerCache = nil
		}
		releasem(mp)

		if p.pinner == nil {
			// Didn't get anything from the pinner cache.
			p.pinner = new(pinner)
			p.refs = p.refStore[:0]

			// We set this finalizer once and never clear it. Thus, if the
			// pinner gets cached, we'll reuse it, along with its finalizer.
			// This lets us avoid the relatively expensive SetFinalizer call
			// when reusing from the cache. The finalizer however has to be
			// resilient to an empty pinner being finalized, which is done
			// by checking p.refs' length.
			SetFinalizer(p.pinner, func(i *pinner) {
				if len(i.refs) != 0 {
					i.unpin() // only required to make the test idempotent
					pinnerLeakPanic()
				}
			})
		}
	}
	ptr := pinnerGetPtr(&pointer)
	if setPinned(ptr, true) {
		p.refs = append(p.refs, ptr)
	}
}

// Unpin unpins all pinned objects of the [Pinner].
func (p *Pinner) Unpin() {
	p.pinner.unpin()

	mp := acquirem()
	if pp := mp.p.ptr(); pp != nil && pp.pinnerCache == nil {
		// Put the pinner back in the cache, but only if the
		// cache is empty. If application code is reusing Pinners
		// on its own, we want to leave the backing store in place
		// so reuse is more efficient.
		pp.pinnerCache = p.pinner
		p.pinner = nil
	}
	releasem(mp)
}

const (
	pinnerSize         = 64
	pinnerRefStoreSize = (pinnerSize - unsafe.Sizeof([]unsafe.Pointer{})) / unsafe.Sizeof(unsafe.Pointer(nil))
)

type pinner struct {
	refs     []unsafe.Pointer
	refStore [pinnerRefStoreSize]unsafe.Pointer
}

func (p *pinner) unpin() {
	if p == nil || p.refs == nil {
		return
	}
	for i := range p.refs {
		setPinned(p.refs[i], false)
	}
	// The following two lines make all pointers to references
	// in p.refs unreachable, either by deleting them or dropping
	// p.refs' backing store (if it was not backed by refStore).
	p.refStore = [pinnerRefStoreSize]unsafe.Pointer{}
	p.refs = p.refStore[:0]
}

func pinnerGetPtr(i *any) unsafe.Pointer {
	e := efaceOf(i)
	etyp := e._type
	if etyp == nil {
		panic(errorString("runtime.Pinner: argument is nil"))
	}
	var data unsafe.Pointer
	kind := etyp.Kind_ & kindMask
	switch kind {
	case kindPtr, kindUnsafePointer:
		data = e.data
	case kindString:
		data = unsafe.Pointer(unsafe.StringData(*(*string)(e.data)))
	case kindSlice:
		data = ((*slice)(e.data)).array
	default:
		panic(errorString("runtime.Pinner: argument is not a pointer, string, or slice: " + toRType(etyp).string()))
	}
	if inUserArenaChunk(uintptr(e.data)) {
		// Arena-allocated objects are not eligible for pinning.
		panic(errorString("runtime.Pinner: object was allocated into an arena"))
	}
	return data
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
	// these pinnerBits might get unlinked by a concurrently running sweep, but
	// that's OK because gcBits don't get cleared until the following GC cycle
	// (nextMarkBitArenaEpoch)
	if pinnerBits == nil {
		return false
	}
	objIndex := span.objIndex(uintptr(ptr))
	pinState := pinnerBits.ofObject(objIndex)
	KeepAlive(ptr) // make sure ptr is alive until we are done so the span can't be freed
	return pinState.isPinned()
}

// setPinned marks or unmarks a Go pointer as pinned, when the ptr is a Go pointer.
// It will be ignored while try to pin a non-Go pointer,
// and it will be panic while try to unpin a non-Go pointer,
// which should not happen in normal usage.
func setPinned(ptr unsafe.Pointer, pin bool) bool {
	span := spanOfHeap(uintptr(ptr))
	if span == nil {
		if !pin {
			panic(errorString("tried to unpin non-Go pointer"))
		}
		// This is a linker-allocated, zero size object or other object,
		// nothing to do, silently ignore it.
		return false
	}

	// ensure that the span is swept, b/c sweeping accesses the specials list
	// w/o locks.
	mp := acquirem()
	span.ensureSwept()
	KeepAlive(ptr) // make sure ptr is still alive after span is swept

	objIndex := span.objIndex(uintptr(ptr))

	lock(&span.speciallock) // guard against concurrent calls of setPinned on same span

	pinnerBits := span.getPinnerBits()
	if pinnerBits == nil {
		pinnerBits = span.newPinnerBits()
		span.setPinnerBits(pinnerBits)
	}
	pinState := pinnerBits.ofObject(objIndex)
	if pin {
		if pinState.isPinned() {
			// multiple pins on same object, set multipin bit
			pinState.setMultiPinned(true)
			// and increase the pin counter
			// TODO(mknyszek): investigate if systemstack is necessary here
			systemstack(func() {
				offset := objIndex * span.elemsize
				span.incPinCounter(offset)
			})
		} else {
			// set pin bit
			pinState.setPinned(true)
		}
	} else {
		// unpin
		if pinState.isPinned() {
			if pinState.isMultiPinned() {
				var exists bool
				// TODO(mknyszek): investigate if systemstack is necessary here
				systemstack(func() {
					offset := objIndex * span.elemsize
					exists = span.decPinCounter(offset)
				})
				if !exists {
					// counter is 0, clear multipin bit
					pinState.setMultiPinned(false)
				}
			} else {
				// no multipins recorded. unpin object.
				pinState.setPinned(false)
			}
		} else {
			// unpinning unpinned object, bail out
			throw("runtime.Pinner: object already unpinned")
		}
	}
	unlock(&span.speciallock)
	releasem(mp)
	return true
}

type pinState struct {
	bytep   *uint8
	byteVal uint8
	mask    uint8
}

// nosplit, because it's called by isPinned, which is nosplit
//
//go:nosplit
func (v *pinState) isPinned() bool {
	return (v.byteVal & v.mask) != 0
}

func (v *pinState) isMultiPinned() bool {
	return (v.byteVal & (v.mask << 1)) != 0
}

func (v *pinState) setPinned(val bool) {
	v.set(val, false)
}

func (v *pinState) setMultiPinned(val bool) {
	v.set(val, true)
}

// set sets the pin bit of the pinState to val. If multipin is true, it
// sets/unsets the multipin bit instead.
func (v *pinState) set(val bool, multipin bool) {
	mask := v.mask
	if multipin {
		mask <<= 1
	}
	if val {
		atomic.Or8(v.bytep, mask)
	} else {
		atomic.And8(v.bytep, ^mask)
	}
}

// pinnerBits is the same type as gcBits but has different methods.
type pinnerBits gcBits

// ofObject returns the pinState of the n'th object.
// nosplit, because it's called by isPinned, which is nosplit
//
//go:nosplit
func (p *pinnerBits) ofObject(n uintptr) pinState {
	bytep, mask := (*gcBits)(p).bitp(n * 2)
	byteVal := atomic.Load8(bytep)
	return pinState{bytep, byteVal, mask}
}

func (s *mspan) pinnerBitSize() uintptr {
	return divRoundUp(uintptr(s.nelems)*2, 8)
}

// newPinnerBits returns a pointer to 8 byte aligned bytes to be used for this
// span's pinner bits. newPinneBits is used to mark objects that are pinned.
// They are copied when the span is swept.
func (s *mspan) newPinnerBits() *pinnerBits {
	return (*pinnerBits)(newMarkBits(uintptr(s.nelems) * 2))
}

// nosplit, because it's called by isPinned, which is nosplit
//
//go:nosplit
func (s *mspan) getPinnerBits() *pinnerBits {
	return (*pinnerBits)(atomic.Loadp(unsafe.Pointer(&s.pinnerBits)))
}

func (s *mspan) setPinnerBits(p *pinnerBits) {
	atomicstorep(unsafe.Pointer(&s.pinnerBits), unsafe.Pointer(p))
}

// refreshPinnerBits replaces pinnerBits with a fresh copy in the arenas for the
// next GC cycle. If it does not contain any pinned objects, pinnerBits of the
// span is set to nil.
func (s *mspan) refreshPinnerBits() {
	p := s.getPinnerBits()
	if p == nil {
		return
	}

	hasPins := false
	bytes := alignUp(s.pinnerBitSize(), 8)

	// Iterate over each 8-byte chunk and check for pins. Note that
	// newPinnerBits guarantees that pinnerBits will be 8-byte aligned, so we
	// don't have to worry about edge cases, irrelevant bits will simply be
	// zero.
	for _, x := range unsafe.Slice((*uint64)(unsafe.Pointer(&p.x)), bytes/8) {
		if x != 0 {
			hasPins = true
			break
		}
	}

	if hasPins {
		newPinnerBits := s.newPinnerBits()
		memmove(unsafe.Pointer(&newPinnerBits.x), unsafe.Pointer(&p.x), bytes)
		s.setPinnerBits(newPinnerBits)
	} else {
		s.setPinnerBits(nil)
	}
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
	} else {
		rec = (*specialPinCounter)(unsafe.Pointer(*ref))
	}
	rec.counter++
}

// decPinCounter decreases the counter. If the counter reaches 0, the counter
// special is deleted and false is returned. Otherwise true is returned.
func (span *mspan) decPinCounter(offset uintptr) bool {
	ref, exists := span.specialFindSplicePoint(offset, _KindSpecialPinCounter)
	if !exists {
		throw("runtime.Pinner: decreased non-existing pin counter")
	}
	counter := (*specialPinCounter)(unsafe.Pointer(*ref))
	counter.counter--
	if counter.counter == 0 {
		*ref = counter.special.next
		if span.specials == nil {
			spanHasNoSpecials(span)
		}
		lock(&mheap_.speciallock)
		mheap_.specialPinCounterAlloc.free(unsafe.Pointer(counter))
		unlock(&mheap_.speciallock)
		return false
	}
	return true
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
