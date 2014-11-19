// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

const (
	debugMalloc = false

	flagNoScan = _FlagNoScan
	flagNoZero = _FlagNoZero

	maxTinySize   = _TinySize
	tinySizeClass = _TinySizeClass
	maxSmallSize  = _MaxSmallSize

	pageShift = _PageShift
	pageSize  = _PageSize
	pageMask  = _PageMask

	bitsPerPointer  = _BitsPerPointer
	bitsMask        = _BitsMask
	pointersPerByte = _PointersPerByte
	maxGCMask       = _MaxGCMask
	bitsDead        = _BitsDead
	bitsPointer     = _BitsPointer

	mSpanInUse = _MSpanInUse

	concurrentSweep = _ConcurrentSweep != 0
)

// Page number (address>>pageShift)
type pageID uintptr

// base address for all 0-byte allocations
var zerobase uintptr

// Allocate an object of size bytes.
// Small objects are allocated from the per-P cache's free lists.
// Large objects (> 32 kB) are allocated straight from the heap.
func mallocgc(size uintptr, typ *_type, flags uint32) unsafe.Pointer {
	if size == 0 {
		return unsafe.Pointer(&zerobase)
	}
	size0 := size

	if flags&flagNoScan == 0 && typ == nil {
		gothrow("malloc missing type")
	}

	// This function must be atomic wrt GC, but for performance reasons
	// we don't acquirem/releasem on fast path. The code below does not have
	// split stack checks, so it can't be preempted by GC.
	// Functions like roundup/add are inlined. And onM/racemalloc are nosplit.
	// If debugMalloc = true, these assumptions are checked below.
	if debugMalloc {
		mp := acquirem()
		if mp.mallocing != 0 {
			gothrow("malloc deadlock")
		}
		mp.mallocing = 1
		if mp.curg != nil {
			mp.curg.stackguard0 = ^uintptr(0xfff) | 0xbad
		}
	}

	c := gomcache()
	var s *mspan
	var x unsafe.Pointer
	if size <= maxSmallSize {
		if flags&flagNoScan != 0 && size < maxTinySize {
			// Tiny allocator.
			//
			// Tiny allocator combines several tiny allocation requests
			// into a single memory block. The resulting memory block
			// is freed when all subobjects are unreachable. The subobjects
			// must be FlagNoScan (don't have pointers), this ensures that
			// the amount of potentially wasted memory is bounded.
			//
			// Size of the memory block used for combining (maxTinySize) is tunable.
			// Current setting is 16 bytes, which relates to 2x worst case memory
			// wastage (when all but one subobjects are unreachable).
			// 8 bytes would result in no wastage at all, but provides less
			// opportunities for combining.
			// 32 bytes provides more opportunities for combining,
			// but can lead to 4x worst case wastage.
			// The best case winning is 8x regardless of block size.
			//
			// Objects obtained from tiny allocator must not be freed explicitly.
			// So when an object will be freed explicitly, we ensure that
			// its size >= maxTinySize.
			//
			// SetFinalizer has a special case for objects potentially coming
			// from tiny allocator, it such case it allows to set finalizers
			// for an inner byte of a memory block.
			//
			// The main targets of tiny allocator are small strings and
			// standalone escaping variables. On a json benchmark
			// the allocator reduces number of allocations by ~12% and
			// reduces heap size by ~20%.
			tinysize := uintptr(c.tinysize)
			if size <= tinysize {
				tiny := unsafe.Pointer(c.tiny)
				// Align tiny pointer for required (conservative) alignment.
				if size&7 == 0 {
					tiny = roundup(tiny, 8)
				} else if size&3 == 0 {
					tiny = roundup(tiny, 4)
				} else if size&1 == 0 {
					tiny = roundup(tiny, 2)
				}
				size1 := size + (uintptr(tiny) - uintptr(unsafe.Pointer(c.tiny)))
				if size1 <= tinysize {
					// The object fits into existing tiny block.
					x = tiny
					c.tiny = (*byte)(add(x, size))
					c.tinysize -= uintptr(size1)
					c.local_tinyallocs++
					if debugMalloc {
						mp := acquirem()
						if mp.mallocing == 0 {
							gothrow("bad malloc")
						}
						mp.mallocing = 0
						if mp.curg != nil {
							mp.curg.stackguard0 = mp.curg.stack.lo + _StackGuard
						}
						// Note: one releasem for the acquirem just above.
						// The other for the acquirem at start of malloc.
						releasem(mp)
						releasem(mp)
					}
					return x
				}
			}
			// Allocate a new maxTinySize block.
			s = c.alloc[tinySizeClass]
			v := s.freelist
			if v == nil {
				mp := acquirem()
				mp.scalararg[0] = tinySizeClass
				onM(mcacheRefill_m)
				releasem(mp)
				s = c.alloc[tinySizeClass]
				v = s.freelist
			}
			s.freelist = v.next
			s.ref++
			//TODO: prefetch v.next
			x = unsafe.Pointer(v)
			(*[2]uint64)(x)[0] = 0
			(*[2]uint64)(x)[1] = 0
			// See if we need to replace the existing tiny block with the new one
			// based on amount of remaining free space.
			if maxTinySize-size > tinysize {
				c.tiny = (*byte)(add(x, size))
				c.tinysize = uintptr(maxTinySize - size)
			}
			size = maxTinySize
		} else {
			var sizeclass int8
			if size <= 1024-8 {
				sizeclass = size_to_class8[(size+7)>>3]
			} else {
				sizeclass = size_to_class128[(size-1024+127)>>7]
			}
			size = uintptr(class_to_size[sizeclass])
			s = c.alloc[sizeclass]
			v := s.freelist
			if v == nil {
				mp := acquirem()
				mp.scalararg[0] = uintptr(sizeclass)
				onM(mcacheRefill_m)
				releasem(mp)
				s = c.alloc[sizeclass]
				v = s.freelist
			}
			s.freelist = v.next
			s.ref++
			//TODO: prefetch
			x = unsafe.Pointer(v)
			if flags&flagNoZero == 0 {
				v.next = nil
				if size > 2*ptrSize && ((*[2]uintptr)(x))[1] != 0 {
					memclr(unsafe.Pointer(v), size)
				}
			}
		}
		c.local_cachealloc += intptr(size)
	} else {
		mp := acquirem()
		mp.scalararg[0] = uintptr(size)
		mp.scalararg[1] = uintptr(flags)
		onM(largeAlloc_m)
		s = (*mspan)(mp.ptrarg[0])
		mp.ptrarg[0] = nil
		releasem(mp)
		x = unsafe.Pointer(uintptr(s.start << pageShift))
		size = uintptr(s.elemsize)
	}

	if flags&flagNoScan != 0 {
		// All objects are pre-marked as noscan.
		goto marked
	}

	// If allocating a defer+arg block, now that we've picked a malloc size
	// large enough to hold everything, cut the "asked for" size down to
	// just the defer header, so that the GC bitmap will record the arg block
	// as containing nothing at all (as if it were unused space at the end of
	// a malloc block caused by size rounding).
	// The defer arg areas are scanned as part of scanstack.
	if typ == deferType {
		size0 = unsafe.Sizeof(_defer{})
	}

	// From here till marked label marking the object as allocated
	// and storing type info in the GC bitmap.
	{
		arena_start := uintptr(unsafe.Pointer(mheap_.arena_start))
		off := (uintptr(x) - arena_start) / ptrSize
		xbits := (*uint8)(unsafe.Pointer(arena_start - off/wordsPerBitmapByte - 1))
		shift := (off % wordsPerBitmapByte) * gcBits
		if debugMalloc && ((*xbits>>shift)&(bitMask|bitPtrMask)) != bitBoundary {
			println("runtime: bits =", (*xbits>>shift)&(bitMask|bitPtrMask))
			gothrow("bad bits in markallocated")
		}

		var ti, te uintptr
		var ptrmask *uint8
		if size == ptrSize {
			// It's one word and it has pointers, it must be a pointer.
			*xbits |= (bitsPointer << 2) << shift
			goto marked
		}
		if typ.kind&kindGCProg != 0 {
			nptr := (uintptr(typ.size) + ptrSize - 1) / ptrSize
			masksize := nptr
			if masksize%2 != 0 {
				masksize *= 2 // repeated
			}
			masksize = masksize * pointersPerByte / 8 // 4 bits per word
			masksize++                                // unroll flag in the beginning
			if masksize > maxGCMask && typ.gc[1] != 0 {
				// If the mask is too large, unroll the program directly
				// into the GC bitmap. It's 7 times slower than copying
				// from the pre-unrolled mask, but saves 1/16 of type size
				// memory for the mask.
				mp := acquirem()
				mp.ptrarg[0] = x
				mp.ptrarg[1] = unsafe.Pointer(typ)
				mp.scalararg[0] = uintptr(size)
				mp.scalararg[1] = uintptr(size0)
				onM(unrollgcproginplace_m)
				releasem(mp)
				goto marked
			}
			ptrmask = (*uint8)(unsafe.Pointer(uintptr(typ.gc[0])))
			// Check whether the program is already unrolled.
			if uintptr(atomicloadp(unsafe.Pointer(ptrmask)))&0xff == 0 {
				mp := acquirem()
				mp.ptrarg[0] = unsafe.Pointer(typ)
				onM(unrollgcprog_m)
				releasem(mp)
			}
			ptrmask = (*uint8)(add(unsafe.Pointer(ptrmask), 1)) // skip the unroll flag byte
		} else {
			ptrmask = (*uint8)(unsafe.Pointer(typ.gc[0])) // pointer to unrolled mask
		}
		if size == 2*ptrSize {
			*xbits = *ptrmask | bitBoundary
			goto marked
		}
		te = uintptr(typ.size) / ptrSize
		// If the type occupies odd number of words, its mask is repeated.
		if te%2 == 0 {
			te /= 2
		}
		// Copy pointer bitmask into the bitmap.
		for i := uintptr(0); i < size0; i += 2 * ptrSize {
			v := *(*uint8)(add(unsafe.Pointer(ptrmask), ti))
			ti++
			if ti == te {
				ti = 0
			}
			if i == 0 {
				v |= bitBoundary
			}
			if i+ptrSize == size0 {
				v &^= uint8(bitPtrMask << 4)
			}

			*xbits = v
			xbits = (*byte)(add(unsafe.Pointer(xbits), ^uintptr(0)))
		}
		if size0%(2*ptrSize) == 0 && size0 < size {
			// Mark the word after last object's word as bitsDead.
			*xbits = bitsDead << 2
		}
	}
marked:
	if raceenabled {
		racemalloc(x, size)
	}

	if debugMalloc {
		mp := acquirem()
		if mp.mallocing == 0 {
			gothrow("bad malloc")
		}
		mp.mallocing = 0
		if mp.curg != nil {
			mp.curg.stackguard0 = mp.curg.stack.lo + _StackGuard
		}
		// Note: one releasem for the acquirem just above.
		// The other for the acquirem at start of malloc.
		releasem(mp)
		releasem(mp)
	}

	if debug.allocfreetrace != 0 {
		tracealloc(x, size, typ)
	}

	if rate := MemProfileRate; rate > 0 {
		if size < uintptr(rate) && int32(size) < c.next_sample {
			c.next_sample -= int32(size)
		} else {
			mp := acquirem()
			profilealloc(mp, x, size)
			releasem(mp)
		}
	}

	if memstats.heap_alloc >= memstats.next_gc {
		gogc(0)
	}

	return x
}

// implementation of new builtin
func newobject(typ *_type) unsafe.Pointer {
	flags := uint32(0)
	if typ.kind&kindNoPointers != 0 {
		flags |= flagNoScan
	}
	return mallocgc(uintptr(typ.size), typ, flags)
}

// implementation of make builtin for slices
func newarray(typ *_type, n uintptr) unsafe.Pointer {
	flags := uint32(0)
	if typ.kind&kindNoPointers != 0 {
		flags |= flagNoScan
	}
	if int(n) < 0 || (typ.size > 0 && n > maxmem/uintptr(typ.size)) {
		panic("runtime: allocation size out of range")
	}
	return mallocgc(uintptr(typ.size)*n, typ, flags)
}

// rawmem returns a chunk of pointerless memory.  It is
// not zeroed.
func rawmem(size uintptr) unsafe.Pointer {
	return mallocgc(size, nil, flagNoScan|flagNoZero)
}

// round size up to next size class
func goroundupsize(size uintptr) uintptr {
	if size < maxSmallSize {
		if size <= 1024-8 {
			return uintptr(class_to_size[size_to_class8[(size+7)>>3]])
		}
		return uintptr(class_to_size[size_to_class128[(size-1024+127)>>7]])
	}
	if size+pageSize < size {
		return size
	}
	return (size + pageSize - 1) &^ pageMask
}

func profilealloc(mp *m, x unsafe.Pointer, size uintptr) {
	c := mp.mcache
	rate := MemProfileRate
	if size < uintptr(rate) {
		// pick next profile time
		// If you change this, also change allocmcache.
		if rate > 0x3fffffff { // make 2*rate not overflow
			rate = 0x3fffffff
		}
		next := int32(fastrand1()) % (2 * int32(rate))
		// Subtract the "remainder" of the current allocation.
		// Otherwise objects that are close in size to sampling rate
		// will be under-sampled, because we consistently discard this remainder.
		next -= (int32(size) - c.next_sample)
		if next < 0 {
			next = 0
		}
		c.next_sample = next
	}

	mProf_Malloc(x, size)
}

// force = 1 - do GC regardless of current heap usage
// force = 2 - go GC and eager sweep
func gogc(force int32) {
	// The gc is turned off (via enablegc) until the bootstrap has completed.
	// Also, malloc gets called in the guts of a number of libraries that might be
	// holding locks. To avoid deadlocks during stoptheworld, don't bother
	// trying to run gc while holding a lock. The next mallocgc without a lock
	// will do the gc instead.
	mp := acquirem()
	if gp := getg(); gp == mp.g0 || mp.locks > 1 || !memstats.enablegc || panicking != 0 || gcpercent < 0 {
		releasem(mp)
		return
	}
	releasem(mp)
	mp = nil

	semacquire(&worldsema, false)

	if force == 0 && memstats.heap_alloc < memstats.next_gc {
		// typically threads which lost the race to grab
		// worldsema exit here when gc is done.
		semrelease(&worldsema)
		return
	}

	// Ok, we're doing it!  Stop everybody else
	startTime := nanotime()
	mp = acquirem()
	mp.gcing = 1
	releasem(mp)
	onM(stoptheworld)
	if mp != acquirem() {
		gothrow("gogc: rescheduled")
	}

	clearpools()

	// Run gc on the g0 stack.  We do this so that the g stack
	// we're currently running on will no longer change.  Cuts
	// the root set down a bit (g0 stacks are not scanned, and
	// we don't need to scan gc's internal state).  We also
	// need to switch to g0 so we can shrink the stack.
	n := 1
	if debug.gctrace > 1 {
		n = 2
	}
	for i := 0; i < n; i++ {
		if i > 0 {
			startTime = nanotime()
		}
		// switch to g0, call gc, then switch back
		mp.scalararg[0] = uintptr(uint32(startTime)) // low 32 bits
		mp.scalararg[1] = uintptr(startTime >> 32)   // high 32 bits
		if force >= 2 {
			mp.scalararg[2] = 1 // eagersweep
		} else {
			mp.scalararg[2] = 0
		}
		onM(gc_m)
	}

	// all done
	mp.gcing = 0
	semrelease(&worldsema)
	onM(starttheworld)
	releasem(mp)
	mp = nil

	// now that gc is done, kick off finalizer thread if needed
	if !concurrentSweep {
		// give the queued finalizers, if any, a chance to run
		Gosched()
	}
}

// GC runs a garbage collection.
func GC() {
	gogc(2)
}

// linker-provided
var noptrdata struct{}
var enoptrdata struct{}
var noptrbss struct{}
var enoptrbss struct{}

// SetFinalizer sets the finalizer associated with x to f.
// When the garbage collector finds an unreachable block
// with an associated finalizer, it clears the association and runs
// f(x) in a separate goroutine.  This makes x reachable again, but
// now without an associated finalizer.  Assuming that SetFinalizer
// is not called again, the next time the garbage collector sees
// that x is unreachable, it will free x.
//
// SetFinalizer(x, nil) clears any finalizer associated with x.
//
// The argument x must be a pointer to an object allocated by
// calling new or by taking the address of a composite literal.
// The argument f must be a function that takes a single argument
// to which x's type can be assigned, and can have arbitrary ignored return
// values. If either of these is not true, SetFinalizer aborts the
// program.
//
// Finalizers are run in dependency order: if A points at B, both have
// finalizers, and they are otherwise unreachable, only the finalizer
// for A runs; once A is freed, the finalizer for B can run.
// If a cyclic structure includes a block with a finalizer, that
// cycle is not guaranteed to be garbage collected and the finalizer
// is not guaranteed to run, because there is no ordering that
// respects the dependencies.
//
// The finalizer for x is scheduled to run at some arbitrary time after
// x becomes unreachable.
// There is no guarantee that finalizers will run before a program exits,
// so typically they are useful only for releasing non-memory resources
// associated with an object during a long-running program.
// For example, an os.File object could use a finalizer to close the
// associated operating system file descriptor when a program discards
// an os.File without calling Close, but it would be a mistake
// to depend on a finalizer to flush an in-memory I/O buffer such as a
// bufio.Writer, because the buffer would not be flushed at program exit.
//
// It is not guaranteed that a finalizer will run if the size of *x is
// zero bytes.
//
// It is not guaranteed that a finalizer will run for objects allocated
// in initializers for package-level variables. Such objects may be
// linker-allocated, not heap-allocated.
//
// A single goroutine runs all finalizers for a program, sequentially.
// If a finalizer must run for a long time, it should do so by starting
// a new goroutine.
func SetFinalizer(obj interface{}, finalizer interface{}) {
	e := (*eface)(unsafe.Pointer(&obj))
	etyp := e._type
	if etyp == nil {
		gothrow("runtime.SetFinalizer: first argument is nil")
	}
	if etyp.kind&kindMask != kindPtr {
		gothrow("runtime.SetFinalizer: first argument is " + *etyp._string + ", not pointer")
	}
	ot := (*ptrtype)(unsafe.Pointer(etyp))
	if ot.elem == nil {
		gothrow("nil elem type!")
	}

	// find the containing object
	_, base, _ := findObject(e.data)

	if base == nil {
		// 0-length objects are okay.
		if e.data == unsafe.Pointer(&zerobase) {
			return
		}

		// Global initializers might be linker-allocated.
		//	var Foo = &Object{}
		//	func main() {
		//		runtime.SetFinalizer(Foo, nil)
		//	}
		// The relevant segments are: noptrdata, data, bss, noptrbss.
		// We cannot assume they are in any order or even contiguous,
		// due to external linking.
		if uintptr(unsafe.Pointer(&noptrdata)) <= uintptr(e.data) && uintptr(e.data) < uintptr(unsafe.Pointer(&enoptrdata)) ||
			uintptr(unsafe.Pointer(&data)) <= uintptr(e.data) && uintptr(e.data) < uintptr(unsafe.Pointer(&edata)) ||
			uintptr(unsafe.Pointer(&bss)) <= uintptr(e.data) && uintptr(e.data) < uintptr(unsafe.Pointer(&ebss)) ||
			uintptr(unsafe.Pointer(&noptrbss)) <= uintptr(e.data) && uintptr(e.data) < uintptr(unsafe.Pointer(&enoptrbss)) {
			return
		}
		gothrow("runtime.SetFinalizer: pointer not in allocated block")
	}

	if e.data != base {
		// As an implementation detail we allow to set finalizers for an inner byte
		// of an object if it could come from tiny alloc (see mallocgc for details).
		if ot.elem == nil || ot.elem.kind&kindNoPointers == 0 || ot.elem.size >= maxTinySize {
			gothrow("runtime.SetFinalizer: pointer not at beginning of allocated block")
		}
	}

	f := (*eface)(unsafe.Pointer(&finalizer))
	ftyp := f._type
	if ftyp == nil {
		// switch to M stack and remove finalizer
		mp := acquirem()
		mp.ptrarg[0] = e.data
		onM(removeFinalizer_m)
		releasem(mp)
		return
	}

	if ftyp.kind&kindMask != kindFunc {
		gothrow("runtime.SetFinalizer: second argument is " + *ftyp._string + ", not a function")
	}
	ft := (*functype)(unsafe.Pointer(ftyp))
	ins := *(*[]*_type)(unsafe.Pointer(&ft.in))
	if ft.dotdotdot || len(ins) != 1 {
		gothrow("runtime.SetFinalizer: cannot pass " + *etyp._string + " to finalizer " + *ftyp._string)
	}
	fint := ins[0]
	switch {
	case fint == etyp:
		// ok - same type
		goto okarg
	case fint.kind&kindMask == kindPtr:
		if (fint.x == nil || fint.x.name == nil || etyp.x == nil || etyp.x.name == nil) && (*ptrtype)(unsafe.Pointer(fint)).elem == ot.elem {
			// ok - not same type, but both pointers,
			// one or the other is unnamed, and same element type, so assignable.
			goto okarg
		}
	case fint.kind&kindMask == kindInterface:
		ityp := (*interfacetype)(unsafe.Pointer(fint))
		if len(ityp.mhdr) == 0 {
			// ok - satisfies empty interface
			goto okarg
		}
		if _, ok := assertE2I2(ityp, obj); ok {
			goto okarg
		}
	}
	gothrow("runtime.SetFinalizer: cannot pass " + *etyp._string + " to finalizer " + *ftyp._string)
okarg:
	// compute size needed for return parameters
	nret := uintptr(0)
	for _, t := range *(*[]*_type)(unsafe.Pointer(&ft.out)) {
		nret = round(nret, uintptr(t.align)) + uintptr(t.size)
	}
	nret = round(nret, ptrSize)

	// make sure we have a finalizer goroutine
	createfing()

	// switch to M stack to add finalizer record
	mp := acquirem()
	mp.ptrarg[0] = f.data
	mp.ptrarg[1] = e.data
	mp.scalararg[0] = nret
	mp.ptrarg[2] = unsafe.Pointer(fint)
	mp.ptrarg[3] = unsafe.Pointer(ot)
	onM(setFinalizer_m)
	if mp.scalararg[0] != 1 {
		gothrow("runtime.SetFinalizer: finalizer already set")
	}
	releasem(mp)
}

// round n up to a multiple of a.  a must be a power of 2.
func round(n, a uintptr) uintptr {
	return (n + a - 1) &^ (a - 1)
}

// Look up pointer v in heap.  Return the span containing the object,
// the start of the object, and the size of the object.  If the object
// does not exist, return nil, nil, 0.
func findObject(v unsafe.Pointer) (s *mspan, x unsafe.Pointer, n uintptr) {
	c := gomcache()
	c.local_nlookup++
	if ptrSize == 4 && c.local_nlookup >= 1<<30 {
		// purge cache stats to prevent overflow
		lock(&mheap_.lock)
		purgecachedstats(c)
		unlock(&mheap_.lock)
	}

	// find span
	arena_start := uintptr(unsafe.Pointer(mheap_.arena_start))
	arena_used := uintptr(unsafe.Pointer(mheap_.arena_used))
	if uintptr(v) < arena_start || uintptr(v) >= arena_used {
		return
	}
	p := uintptr(v) >> pageShift
	q := p - arena_start>>pageShift
	s = *(**mspan)(add(unsafe.Pointer(mheap_.spans), q*ptrSize))
	if s == nil {
		return
	}
	x = unsafe.Pointer(uintptr(s.start) << pageShift)

	if uintptr(v) < uintptr(x) || uintptr(v) >= uintptr(unsafe.Pointer(s.limit)) || s.state != mSpanInUse {
		s = nil
		x = nil
		return
	}

	n = uintptr(s.elemsize)
	if s.sizeclass != 0 {
		x = add(x, (uintptr(v)-uintptr(x))/n*n)
	}
	return
}

var fingCreate uint32

func createfing() {
	// start the finalizer goroutine exactly once
	if fingCreate == 0 && cas(&fingCreate, 0, 1) {
		go runfinq()
	}
}

// This is the goroutine that runs all of the finalizers
func runfinq() {
	var (
		frame    unsafe.Pointer
		framecap uintptr
	)

	for {
		lock(&finlock)
		fb := finq
		finq = nil
		if fb == nil {
			gp := getg()
			fing = gp
			fingwait = true
			gp.issystem = true
			goparkunlock(&finlock, "finalizer wait")
			gp.issystem = false
			continue
		}
		unlock(&finlock)
		if raceenabled {
			racefingo()
		}
		for fb != nil {
			for i := int32(0); i < fb.cnt; i++ {
				f := (*finalizer)(add(unsafe.Pointer(&fb.fin), uintptr(i)*unsafe.Sizeof(finalizer{})))

				framesz := unsafe.Sizeof((interface{})(nil)) + uintptr(f.nret)
				if framecap < framesz {
					// The frame does not contain pointers interesting for GC,
					// all not yet finalized objects are stored in finq.
					// If we do not mark it as FlagNoScan,
					// the last finalized object is not collected.
					frame = mallocgc(framesz, nil, flagNoScan)
					framecap = framesz
				}

				if f.fint == nil {
					gothrow("missing type in runfinq")
				}
				switch f.fint.kind & kindMask {
				case kindPtr:
					// direct use of pointer
					*(*unsafe.Pointer)(frame) = f.arg
				case kindInterface:
					ityp := (*interfacetype)(unsafe.Pointer(f.fint))
					// set up with empty interface
					(*eface)(frame)._type = &f.ot.typ
					(*eface)(frame).data = f.arg
					if len(ityp.mhdr) != 0 {
						// convert to interface with methods
						// this conversion is guaranteed to succeed - we checked in SetFinalizer
						*(*fInterface)(frame) = assertE2I(ityp, *(*interface{})(frame))
					}
				default:
					gothrow("bad kind in runfinq")
				}
				reflectcall(unsafe.Pointer(f.fn), frame, uint32(framesz), uint32(framesz))

				// drop finalizer queue references to finalized object
				f.fn = nil
				f.arg = nil
				f.ot = nil
			}
			fb.cnt = 0
			next := fb.next
			lock(&finlock)
			fb.next = finc
			finc = fb
			unlock(&finlock)
			fb = next
		}
	}
}

var persistent struct {
	lock mutex
	pos  unsafe.Pointer
	end  unsafe.Pointer
}

// Wrapper around sysAlloc that can allocate small chunks.
// There is no associated free operation.
// Intended for things like function/type/debug-related persistent data.
// If align is 0, uses default align (currently 8).
func persistentalloc(size, align uintptr, stat *uint64) unsafe.Pointer {
	const (
		chunk    = 256 << 10
		maxBlock = 64 << 10 // VM reservation granularity is 64K on windows
	)

	if align != 0 {
		if align&(align-1) != 0 {
			gothrow("persistentalloc: align is not a power of 2")
		}
		if align > _PageSize {
			gothrow("persistentalloc: align is too large")
		}
	} else {
		align = 8
	}

	if size >= maxBlock {
		return sysAlloc(size, stat)
	}

	lock(&persistent.lock)
	persistent.pos = roundup(persistent.pos, align)
	if uintptr(persistent.pos)+size > uintptr(persistent.end) {
		persistent.pos = sysAlloc(chunk, &memstats.other_sys)
		if persistent.pos == nil {
			unlock(&persistent.lock)
			gothrow("runtime: cannot allocate memory")
		}
		persistent.end = add(persistent.pos, chunk)
	}
	p := persistent.pos
	persistent.pos = add(persistent.pos, size)
	unlock(&persistent.lock)

	if stat != &memstats.other_sys {
		xadd64(stat, int64(size))
		xadd64(&memstats.other_sys, -int64(size))
	}
	return p
}
