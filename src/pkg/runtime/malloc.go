// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

const (
	flagNoScan      = 1 << 0 // GC doesn't have to scan object
	flagNoProfiling = 1 << 1 // must not profile
	flagNoZero      = 1 << 3 // don't zero memory
	flagNoInvokeGC  = 1 << 4 // don't invoke GC

	kindArray      = 17
	kindFunc       = 19
	kindInterface  = 20
	kindPtr        = 22
	kindStruct     = 25
	kindMask       = 1<<6 - 1
	kindGCProg     = 1 << 6
	kindNoPointers = 1 << 7

	maxTinySize   = 16
	tinySizeClass = 2
	maxSmallSize  = 32 << 10

	pageShift = 13
	pageSize  = 1 << pageShift
	pageMask  = pageSize - 1
)

// All zero-sized allocations return a pointer to this byte.
var zeroObject byte

// Maximum possible heap size.
var maxMem uintptr

// Allocate an object of size bytes.
// Small objects are allocated from the per-P cache's free lists.
// Large objects (> 32 kB) are allocated straight from the heap.
func gomallocgc(size uintptr, typ *_type, flags int) unsafe.Pointer {
	if size == 0 {
		return unsafe.Pointer(&zeroObject)
	}
	mp := acquirem()
	if mp.mallocing != 0 {
		gothrow("malloc/free - deadlock")
	}
	mp.mallocing = 1
	size0 := size

	c := mp.mcache
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
					c.tinysize -= uint(size1)
					mp.mallocing = 0
					releasem(mp)
					return x
				}
			}
			// Allocate a new maxTinySize block.
			s = c.alloc[tinySizeClass]
			v := s.freelist
			if v == nil {
				mp.scalararg[0] = tinySizeClass
				onM(&mcacheRefill_m)
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
				c.tinysize = uint(maxTinySize - size)
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
				mp.scalararg[0] = uint(sizeclass)
				onM(&mcacheRefill_m)
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
		c.local_cachealloc += int(size)
	} else {
		mp.scalararg[0] = uint(size)
		mp.scalararg[1] = uint(flags)
		onM(&largeAlloc_m)
		s = (*mspan)(mp.ptrarg[0])
		mp.ptrarg[0] = nil
		x = unsafe.Pointer(uintptr(s.start << pageShift))
		size = uintptr(s.elemsize)
	}

	// TODO: write markallocated in Go
	mp.ptrarg[0] = x
	mp.scalararg[0] = uint(size)
	mp.scalararg[1] = uint(size0)
	mp.ptrarg[1] = unsafe.Pointer(typ)
	mp.scalararg[2] = uint(flags & flagNoScan)
	onM(&markallocated_m)

	mp.mallocing = 0

	if raceenabled {
		racemalloc(x, size)
	}
	if debug.allocfreetrace != 0 {
		tracealloc(x, size, typ)
	}
	if flags&flagNoProfiling == 0 {
		rate := MemProfileRate
		if rate > 0 {
			if size < uintptr(rate) && int32(size) < c.next_sample {
				c.next_sample -= int32(size)
			} else {
				profilealloc(mp, x, size)
			}
		}
	}

	releasem(mp)

	if flags&flagNoInvokeGC == 0 && memstats.heap_alloc >= memstats.next_gc {
		gogc(0)
	}

	return x
}

// cmallocgc is a trampoline used to call the Go malloc from C.
func cmallocgc(size uintptr, typ *_type, flags int, ret *unsafe.Pointer) {
	*ret = gomallocgc(size, typ, flags)
}

// implementation of new builtin
func newobject(typ *_type) unsafe.Pointer {
	flags := 0
	if typ.kind&kindNoPointers != 0 {
		flags |= flagNoScan
	}
	return gomallocgc(uintptr(typ.size), typ, flags)
}

// implementation of make builtin for slices
func newarray(typ *_type, n uintptr) unsafe.Pointer {
	flags := 0
	if typ.kind&kindNoPointers != 0 {
		flags |= flagNoScan
	}
	if int(n) < 0 || (typ.size > 0 && n > maxMem/uintptr(typ.size)) {
		panic("runtime: allocation size out of range")
	}
	return gomallocgc(uintptr(typ.size)*n, typ, flags)
}

// rawmem returns a chunk of pointerless memory.  It is
// not zeroed.
func rawmem(size uintptr) unsafe.Pointer {
	return gomallocgc(size, nil, flagNoScan|flagNoZero)
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
		next := int32(fastrand2()) % (2 * int32(rate))
		// Subtract the "remainder" of the current allocation.
		// Otherwise objects that are close in size to sampling rate
		// will be under-sampled, because we consistently discard this remainder.
		next -= (int32(size) - c.next_sample)
		if next < 0 {
			next = 0
		}
		c.next_sample = next
	}
	mp.scalararg[0] = uint(size)
	mp.ptrarg[0] = x
	onM(&mprofMalloc_m)
}

// force = 1 - do GC regardless of current heap usage
// force = 2 - go GC and eager sweep
func gogc(force int32) {
	if memstats.enablegc == 0 {
		return
	}

	// TODO: should never happen?  Only C calls malloc while holding a lock?
	mp := acquirem()
	if mp.locks > 1 {
		releasem(mp)
		return
	}
	releasem(mp)

	if panicking != 0 {
		return
	}
	if gcpercent == gcpercentUnknown {
		golock(&mheap_.lock)
		if gcpercent == gcpercentUnknown {
			gcpercent = goreadgogc()
		}
		gounlock(&mheap_.lock)
	}
	if gcpercent < 0 {
		return
	}

	semacquire(&worldsema, false)

	if force == 0 && memstats.heap_alloc < memstats.next_gc {
		// typically threads which lost the race to grab
		// worldsema exit here when gc is done.
		semrelease(&worldsema)
		return
	}

	// Ok, we're doing it!  Stop everybody else
	startTime := gonanotime()
	mp = acquirem()
	mp.gcing = 1
	stoptheworld()

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
			startTime = gonanotime()
		}
		// switch to g0, call gc, then switch back
		mp.scalararg[0] = uint(startTime)
		if force >= 2 {
			mp.scalararg[1] = 1 // eagersweep
		} else {
			mp.scalararg[1] = 0
		}
		onM(&gc_m)
	}

	// all done
	mp.gcing = 0
	semrelease(&worldsema)
	starttheworld()
	releasem(mp)

	// now that gc is done, kick off finalizer thread if needed
	if !concurrentSweep {
		// give the queued finalizers, if any, a chance to run
		gosched()
	}
}

// GC runs a garbage collection.
func GC() {
	gogc(2)
}

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
// A single goroutine runs all finalizers for a program, sequentially.
// If a finalizer must run for a long time, it should do so by starting
// a new goroutine.
func SetFinalizer(obj interface{}, finalizer interface{}) {
	// We do just enough work here to make the mcall type safe.
	// The rest is done on the M stack.
	e := (*eface)(unsafe.Pointer(&obj))
	typ := e._type
	if typ == nil {
		gothrow("runtime.SetFinalizer: first argument is nil")
	}
	if typ.kind&kindMask != kindPtr {
		gothrow("runtime.SetFinalizer: first argument is " + *typ._string + ", not pointer")
	}

	f := (*eface)(unsafe.Pointer(&finalizer))
	ftyp := f._type
	if ftyp != nil && ftyp.kind&kindMask != kindFunc {
		gothrow("runtime.SetFinalizer: second argument is " + *ftyp._string + ", not a function")
	}
	mp := acquirem()
	mp.ptrarg[0] = unsafe.Pointer(typ)
	mp.ptrarg[1] = e.data
	mp.ptrarg[2] = unsafe.Pointer(ftyp)
	mp.ptrarg[3] = f.data
	onM(&setFinalizer_m)
	releasem(mp)
}
