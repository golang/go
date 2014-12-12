// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector: finalizers and block profiling.

package runtime

import "unsafe"

var finlock mutex  // protects the following variables
var fing *g        // goroutine that runs finalizers
var finq *finblock // list of finalizers that are to be executed
var finc *finblock // cache of free blocks
var finptrmask [_FinBlockSize / typeBitmapScale]byte
var fingwait bool
var fingwake bool
var allfin *finblock // list of all blocks

var finalizer1 = [...]byte{
	// Each Finalizer is 5 words, ptr ptr uintptr ptr ptr.
	// Each byte describes 4 words.
	// Need 4 Finalizers described by 5 bytes before pattern repeats:
	//	ptr ptr uintptr ptr ptr
	//	ptr ptr uintptr ptr ptr
	//	ptr ptr uintptr ptr ptr
	//	ptr ptr uintptr ptr ptr
	// aka
	//	ptr ptr uintptr ptr
	//	ptr ptr ptr uintptr
	//	ptr ptr ptr ptr
	//	uintptr ptr ptr ptr
	//	ptr uintptr ptr ptr
	// Assumptions about Finalizer layout checked below.
	typePointer | typePointer<<2 | typeScalar<<4 | typePointer<<6,
	typePointer | typePointer<<2 | typePointer<<4 | typeScalar<<6,
	typePointer | typePointer<<2 | typePointer<<4 | typePointer<<6,
	typeScalar | typePointer<<2 | typePointer<<4 | typePointer<<6,
	typePointer | typeScalar<<2 | typePointer<<4 | typePointer<<6,
}

func queuefinalizer(p unsafe.Pointer, fn *funcval, nret uintptr, fint *_type, ot *ptrtype) {
	lock(&finlock)
	if finq == nil || finq.cnt == int32(len(finq.fin)) {
		if finc == nil {
			// Note: write barrier here, assigning to finc, but should be okay.
			finc = (*finblock)(persistentalloc(_FinBlockSize, 0, &memstats.gc_sys))
			finc.alllink = allfin
			allfin = finc
			if finptrmask[0] == 0 {
				// Build pointer mask for Finalizer array in block.
				// Check assumptions made in finalizer1 array above.
				if (unsafe.Sizeof(finalizer{}) != 5*ptrSize ||
					unsafe.Offsetof(finalizer{}.fn) != 0 ||
					unsafe.Offsetof(finalizer{}.arg) != ptrSize ||
					unsafe.Offsetof(finalizer{}.nret) != 2*ptrSize ||
					unsafe.Offsetof(finalizer{}.fint) != 3*ptrSize ||
					unsafe.Offsetof(finalizer{}.ot) != 4*ptrSize ||
					typeBitsWidth != 2) {
					throw("finalizer out of sync")
				}
				for i := range finptrmask {
					finptrmask[i] = finalizer1[i%len(finalizer1)]
				}
			}
		}
		block := finc
		finc = block.next
		block.next = finq
		finq = block
	}
	f := &finq.fin[finq.cnt]
	finq.cnt++
	f.fn = fn
	f.nret = nret
	f.fint = fint
	f.ot = ot
	f.arg = p
	fingwake = true
	unlock(&finlock)
}

//go:nowritebarrier
func iterate_finq(callback func(*funcval, unsafe.Pointer, uintptr, *_type, *ptrtype)) {
	for fb := allfin; fb != nil; fb = fb.alllink {
		for i := int32(0); i < fb.cnt; i++ {
			f := &fb.fin[i]
			callback(f.fn, f.arg, f.nret, f.fint, f.ot)
		}
	}
}

func wakefing() *g {
	var res *g
	lock(&finlock)
	if fingwait && fingwake {
		fingwait = false
		fingwake = false
		res = fing
	}
	unlock(&finlock)
	return res
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
			goparkunlock(&finlock, "finalizer wait", traceEvGoBlock)
			gp.issystem = false
			continue
		}
		unlock(&finlock)
		if raceenabled {
			racefingo()
		}
		for fb != nil {
			for i := fb.cnt; i > 0; i-- {
				f := (*finalizer)(add(unsafe.Pointer(&fb.fin), uintptr(i-1)*unsafe.Sizeof(finalizer{})))

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
					throw("missing type in runfinq")
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
						assertE2I(ityp, *(*interface{})(frame), (*fInterface)(frame))
					}
				default:
					throw("bad kind in runfinq")
				}
				reflectcall(nil, unsafe.Pointer(f.fn), frame, uint32(framesz), uint32(framesz))

				// drop finalizer queue references to finalized object
				f.fn = nil
				f.arg = nil
				f.ot = nil
				fb.cnt = i - 1
			}
			next := fb.next
			lock(&finlock)
			fb.next = finc
			finc = fb
			unlock(&finlock)
			fb = next
		}
	}
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
		throw("runtime.SetFinalizer: first argument is nil")
	}
	if etyp.kind&kindMask != kindPtr {
		throw("runtime.SetFinalizer: first argument is " + *etyp._string + ", not pointer")
	}
	ot := (*ptrtype)(unsafe.Pointer(etyp))
	if ot.elem == nil {
		throw("nil elem type!")
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
		throw("runtime.SetFinalizer: pointer not in allocated block")
	}

	if e.data != base {
		// As an implementation detail we allow to set finalizers for an inner byte
		// of an object if it could come from tiny alloc (see mallocgc for details).
		if ot.elem == nil || ot.elem.kind&kindNoPointers == 0 || ot.elem.size >= maxTinySize {
			throw("runtime.SetFinalizer: pointer not at beginning of allocated block")
		}
	}

	f := (*eface)(unsafe.Pointer(&finalizer))
	ftyp := f._type
	if ftyp == nil {
		// switch to system stack and remove finalizer
		systemstack(func() {
			removefinalizer(e.data)
		})
		return
	}

	if ftyp.kind&kindMask != kindFunc {
		throw("runtime.SetFinalizer: second argument is " + *ftyp._string + ", not a function")
	}
	ft := (*functype)(unsafe.Pointer(ftyp))
	ins := *(*[]*_type)(unsafe.Pointer(&ft.in))
	if ft.dotdotdot || len(ins) != 1 {
		throw("runtime.SetFinalizer: cannot pass " + *etyp._string + " to finalizer " + *ftyp._string)
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
		if assertE2I2(ityp, obj, nil) {
			goto okarg
		}
	}
	throw("runtime.SetFinalizer: cannot pass " + *etyp._string + " to finalizer " + *ftyp._string)
okarg:
	// compute size needed for return parameters
	nret := uintptr(0)
	for _, t := range *(*[]*_type)(unsafe.Pointer(&ft.out)) {
		nret = round(nret, uintptr(t.align)) + uintptr(t.size)
	}
	nret = round(nret, ptrSize)

	// make sure we have a finalizer goroutine
	createfing()

	systemstack(func() {
		if !addfinalizer(e.data, (*funcval)(f.data), nret, fint, ot) {
			throw("runtime.SetFinalizer: finalizer already set")
		}
	})
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
