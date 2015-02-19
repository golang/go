// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector: marking and scanning

package runtime

import "unsafe"

// Scan all of the stacks, greying (or graying if in America) the referents
// but not blackening them since the mark write barrier isn't installed.
//go:nowritebarrier
func gcscan_m() {
	_g_ := getg()

	// Grab the g that called us and potentially allow rescheduling.
	// This allows it to be scanned like other goroutines.
	mastergp := _g_.m.curg
	casgstatus(mastergp, _Grunning, _Gwaiting)
	mastergp.waitreason = "garbage collection scan"

	// Span sweeping has been done by finishsweep_m.
	// Long term we will want to make this goroutine runnable
	// by placing it onto a scanenqueue state and then calling
	// runtime·restartg(mastergp) to make it Grunnable.
	// At the bottom we will want to return this p back to the scheduler.

	// Prepare flag indicating that the scan has not been completed.
	lock(&allglock)
	local_allglen := allglen
	for i := uintptr(0); i < local_allglen; i++ {
		gp := allgs[i]
		gp.gcworkdone = false  // set to true in gcphasework
		gp.gcscanvalid = false // stack has not been scanned
	}
	unlock(&allglock)

	work.nwait = 0
	work.ndone = 0
	work.nproc = 1 // For now do not do this in parallel.
	//	ackgcphase is not needed since we are not scanning running goroutines.
	parforsetup(work.markfor, work.nproc, uint32(_RootCount+local_allglen), false, markroot)
	parfordo(work.markfor)

	lock(&allglock)
	// Check that gc work is done.
	for i := uintptr(0); i < local_allglen; i++ {
		gp := allgs[i]
		if !gp.gcworkdone {
			throw("scan missed a g")
		}
	}
	unlock(&allglock)

	casgstatus(mastergp, _Gwaiting, _Grunning)
	// Let the g that called us continue to run.
}

// ptrmask for an allocation containing a single pointer.
var oneptr = [...]uint8{typePointer}

//go:nowritebarrier
func markroot(desc *parfor, i uint32) {
	var gcw gcWorkProducer
	gcw.initFromCache()

	// Note: if you add a case here, please also update heapdump.c:dumproots.
	switch i {
	case _RootData:
		scanblock(uintptr(unsafe.Pointer(&data)), uintptr(unsafe.Pointer(&edata))-uintptr(unsafe.Pointer(&data)), gcdatamask.bytedata, &gcw)

	case _RootBss:
		scanblock(uintptr(unsafe.Pointer(&bss)), uintptr(unsafe.Pointer(&ebss))-uintptr(unsafe.Pointer(&bss)), gcbssmask.bytedata, &gcw)

	case _RootFinalizers:
		for fb := allfin; fb != nil; fb = fb.alllink {
			scanblock(uintptr(unsafe.Pointer(&fb.fin[0])), uintptr(fb.cnt)*unsafe.Sizeof(fb.fin[0]), &finptrmask[0], &gcw)
		}

	case _RootSpans:
		// mark MSpan.specials
		sg := mheap_.sweepgen
		for spanidx := uint32(0); spanidx < uint32(len(work.spans)); spanidx++ {
			s := work.spans[spanidx]
			if s.state != mSpanInUse {
				continue
			}
			if !useCheckmark && s.sweepgen != sg {
				// sweepgen was updated (+2) during non-checkmark GC pass
				print("sweep ", s.sweepgen, " ", sg, "\n")
				throw("gc: unswept span")
			}
			for sp := s.specials; sp != nil; sp = sp.next {
				if sp.kind != _KindSpecialFinalizer {
					continue
				}
				// don't mark finalized object, but scan it so we
				// retain everything it points to.
				spf := (*specialfinalizer)(unsafe.Pointer(sp))
				// A finalizer can be set for an inner byte of an object, find object beginning.
				p := uintptr(s.start<<_PageShift) + uintptr(spf.special.offset)/s.elemsize*s.elemsize
				if gcphase != _GCscan {
					scanblock(p, s.elemsize, nil, &gcw) // scanned during mark phase
				}
				scanblock(uintptr(unsafe.Pointer(&spf.fn)), ptrSize, &oneptr[0], &gcw)
			}
		}

	case _RootFlushCaches:
		if gcphase != _GCscan { // Do not flush mcaches during GCscan phase.
			flushallmcaches()
		}

	default:
		// the rest is scanning goroutine stacks
		if uintptr(i-_RootCount) >= allglen {
			throw("markroot: bad index")
		}
		gp := allgs[i-_RootCount]

		// remember when we've first observed the G blocked
		// needed only to output in traceback
		status := readgstatus(gp) // We are not in a scan state
		if (status == _Gwaiting || status == _Gsyscall) && gp.waitsince == 0 {
			gp.waitsince = work.tstart
		}

		// Shrink a stack if not much of it is being used but not in the scan phase.
		if gcphase == _GCmarktermination {
			// Shrink during STW GCmarktermination phase thus avoiding
			// complications introduced by shrinking during
			// non-STW phases.
			shrinkstack(gp)
		}
		if readgstatus(gp) == _Gdead {
			gp.gcworkdone = true
		} else {
			gp.gcworkdone = false
		}
		restart := stopg(gp)

		// goroutine will scan its own stack when it stops running.
		// Wait until it has.
		for readgstatus(gp) == _Grunning && !gp.gcworkdone {
		}

		// scanstack(gp) is done as part of gcphasework
		// But to make sure we finished we need to make sure that
		// the stack traps have all responded so drop into
		// this while loop until they respond.
		for !gp.gcworkdone {
			status = readgstatus(gp)
			if status == _Gdead {
				gp.gcworkdone = true // scan is a noop
				break
			}
			if status == _Gwaiting || status == _Grunnable {
				restart = stopg(gp)
			}
		}
		if restart {
			restartg(gp)
		}
	}
	gcw.dispose()
}

// gchelpwork does a small bounded amount of gc work. The purpose is to
// shorten the time (as measured by allocations) spent doing a concurrent GC.
// The number of mutator calls is roughly propotional to the number of allocations
// made by that mutator. This slows down the allocation while speeding up the GC.
//go:nowritebarrier
func gchelpwork() {
	switch gcphase {
	default:
		throw("gcphasework in bad gcphase")
	case _GCoff, _GCquiesce, _GCstw:
		// No work.
	case _GCsweep:
		// We could help by calling sweepone to sweep a single span.
		// _ = sweepone()
	case _GCscan:
		// scan the stack, mark the objects, put pointers in work buffers
		// hanging off the P where this is being run.
		// scanstack(gp)
	case _GCmark:
		// Get a full work buffer and empty it.
		// drain your own currentwbuf first in the hopes that it will
		// be more cache friendly.
		var gcw gcWork
		gcw.initFromCache()
		const n = len(workbuf{}.obj)
		gcDrainN(&gcw, n) // drain upto one buffer's worth of objects
		gcw.dispose()
	case _GCmarktermination:
		// We should never be here since the world is stopped.
		// All available mark work will be emptied before returning.
		throw("gcphasework in bad gcphase")
	}
}

// The gp has been moved to a GC safepoint. GC phase specific
// work is done here.
//go:nowritebarrier
func gcphasework(gp *g) {
	switch gcphase {
	default:
		throw("gcphasework in bad gcphase")
	case _GCoff, _GCquiesce, _GCstw, _GCsweep:
		// No work.
	case _GCscan:
		// scan the stack, mark the objects, put pointers in work buffers
		// hanging off the P where this is being run.
		// Indicate that the scan is valid until the goroutine runs again
		scanstack(gp)
	case _GCmark:
		// No work.
	case _GCmarktermination:
		scanstack(gp)
		// All available mark work will be emptied before returning.
	}
	gp.gcworkdone = true
}

//go:nowritebarrier
func scanstack(gp *g) {
	if gp.gcscanvalid {
		return
	}

	if readgstatus(gp)&_Gscan == 0 {
		print("runtime:scanstack: gp=", gp, ", goid=", gp.goid, ", gp->atomicstatus=", hex(readgstatus(gp)), "\n")
		throw("scanstack - bad status")
	}

	switch readgstatus(gp) &^ _Gscan {
	default:
		print("runtime: gp=", gp, ", goid=", gp.goid, ", gp->atomicstatus=", readgstatus(gp), "\n")
		throw("mark - bad status")
	case _Gdead:
		return
	case _Grunning:
		print("runtime: gp=", gp, ", goid=", gp.goid, ", gp->atomicstatus=", readgstatus(gp), "\n")
		throw("scanstack: goroutine not stopped")
	case _Grunnable, _Gsyscall, _Gwaiting:
		// ok
	}

	if gp == getg() {
		throw("can't scan our own stack")
	}
	mp := gp.m
	if mp != nil && mp.helpgc != 0 {
		throw("can't scan gchelper stack")
	}

	var gcw gcWorkProducer
	gcw.initFromCache()
	scanframe := func(frame *stkframe, unused unsafe.Pointer) bool {
		// Pick up gcw as free variable so gentraceback and friends can
		// keep the same signature.
		scanframeworker(frame, unused, &gcw)
		return true
	}
	gentraceback(^uintptr(0), ^uintptr(0), 0, gp, 0, nil, 0x7fffffff, scanframe, nil, 0)
	tracebackdefers(gp, scanframe, nil)
	gcw.disposeToCache()
	gp.gcscanvalid = true
}

// Scan a stack frame: local variables and function arguments/results.
//go:nowritebarrier
func scanframeworker(frame *stkframe, unused unsafe.Pointer, gcw *gcWorkProducer) {

	f := frame.fn
	targetpc := frame.continpc
	if targetpc == 0 {
		// Frame is dead.
		return
	}
	if _DebugGC > 1 {
		print("scanframe ", funcname(f), "\n")
	}
	if targetpc != f.entry {
		targetpc--
	}
	pcdata := pcdatavalue(f, _PCDATA_StackMapIndex, targetpc)
	if pcdata == -1 {
		// We do not have a valid pcdata value but there might be a
		// stackmap for this function.  It is likely that we are looking
		// at the function prologue, assume so and hope for the best.
		pcdata = 0
	}

	// Scan local variables if stack frame has been allocated.
	size := frame.varp - frame.sp
	var minsize uintptr
	if thechar != '6' && thechar != '8' {
		minsize = ptrSize
	} else {
		minsize = 0
	}
	if size > minsize {
		stkmap := (*stackmap)(funcdata(f, _FUNCDATA_LocalsPointerMaps))
		if stkmap == nil || stkmap.n <= 0 {
			print("runtime: frame ", funcname(f), " untyped locals ", hex(frame.varp-size), "+", hex(size), "\n")
			throw("missing stackmap")
		}

		// Locals bitmap information, scan just the pointers in locals.
		if pcdata < 0 || pcdata >= stkmap.n {
			// don't know where we are
			print("runtime: pcdata is ", pcdata, " and ", stkmap.n, " locals stack map entries for ", funcname(f), " (targetpc=", targetpc, ")\n")
			throw("scanframe: bad symbol table")
		}
		bv := stackmapdata(stkmap, pcdata)
		size = (uintptr(bv.n) / typeBitsWidth) * ptrSize
		scanblock(frame.varp-size, size, bv.bytedata, gcw)
	}

	// Scan arguments.
	if frame.arglen > 0 {
		var bv bitvector
		if frame.argmap != nil {
			bv = *frame.argmap
		} else {
			stkmap := (*stackmap)(funcdata(f, _FUNCDATA_ArgsPointerMaps))
			if stkmap == nil || stkmap.n <= 0 {
				print("runtime: frame ", funcname(f), " untyped args ", hex(frame.argp), "+", hex(frame.arglen), "\n")
				throw("missing stackmap")
			}
			if pcdata < 0 || pcdata >= stkmap.n {
				// don't know where we are
				print("runtime: pcdata is ", pcdata, " and ", stkmap.n, " args stack map entries for ", funcname(f), " (targetpc=", targetpc, ")\n")
				throw("scanframe: bad symbol table")
			}
			bv = stackmapdata(stkmap, pcdata)
		}
		scanblock(frame.argp, uintptr(bv.n)/typeBitsWidth*ptrSize, bv.bytedata, gcw)
	}
}

// gcDrain scans objects in work buffers (starting with wbuf), blackening grey
// objects until all work buffers have been drained.
//go:nowritebarrier
func gcDrain(gcw *gcWork) {
	if gcphase != _GCmark && gcphase != _GCmarktermination {
		throw("scanblock phase incorrect")
	}

	for {
		// If another proc wants a pointer, give it some.
		if work.nwait > 0 && work.full == 0 {
			gcw.balance()
		}

		b := gcw.get()
		if b == 0 {
			// work barrier reached
			break
		}
		// If the current wbuf is filled by the scan a new wbuf might be
		// returned that could possibly hold only a single object. This
		// could result in each iteration draining only a single object
		// out of the wbuf passed in + a single object placed
		// into an empty wbuf in scanobject so there could be
		// a performance hit as we keep fetching fresh wbufs.
		scanobject(b, 0, nil, &gcw.gcWorkProducer)
	}
	checknocurrentwbuf()
}

// gcDrainN scans n objects, blackening grey objects.
//go:nowritebarrier
func gcDrainN(gcw *gcWork, n int) {
	checknocurrentwbuf()
	for i := 0; i < n; i++ {
		// This might be a good place to add prefetch code...
		// if(wbuf.nobj > 4) {
		//         PREFETCH(wbuf->obj[wbuf.nobj - 3];
		//  }
		b := gcw.tryGet()
		if b == 0 {
			return
		}
		scanobject(b, 0, nil, &gcw.gcWorkProducer)
	}
}

// scanblock scans b as scanobject would.
// If the gcphase is GCscan, scanblock performs additional checks.
//go:nowritebarrier
func scanblock(b0, n0 uintptr, ptrmask *uint8, gcw *gcWorkProducer) {
	// Use local copies of original parameters, so that a stack trace
	// due to one of the throws below shows the original block
	// base and extent.
	b := b0
	n := n0

	// ptrmask can have 2 possible values:
	// 1. nil - obtain pointer mask from GC bitmap.
	// 2. pointer to a compact mask (for stacks and data).

	scanobject(b, n, ptrmask, gcw)
	if gcphase == _GCscan {
		if inheap(b) && ptrmask == nil {
			// b is in heap, we are in GCscan so there should be a ptrmask.
			throw("scanblock: In GCscan phase and inheap is true.")
		}
	}
}

// Scan the object b of size n, adding pointers to wbuf.
// Return possibly new wbuf to use.
// If ptrmask != nil, it specifies where pointers are in b.
// If ptrmask == nil, the GC bitmap should be consulted.
// In this case, n may be an overestimate of the size; the GC bitmap
// must also be used to make sure the scan stops at the end of b.
//go:nowritebarrier
func scanobject(b, n uintptr, ptrmask *uint8, gcw *gcWorkProducer) {
	arena_start := mheap_.arena_start
	arena_used := mheap_.arena_used

	// Find bits of the beginning of the object.
	var hbits heapBits
	if ptrmask == nil {
		b, hbits = heapBitsForObject(b)
		if b == 0 {
			return
		}
		if n == 0 {
			n = mheap_.arena_used - b
		}
	}
	for i := uintptr(0); i < n; i += ptrSize {
		// Find bits for this word.
		var bits uintptr
		if ptrmask != nil {
			// dense mask (stack or data)
			bits = (uintptr(*(*byte)(add(unsafe.Pointer(ptrmask), (i/ptrSize)/4))) >> (((i / ptrSize) % 4) * typeBitsWidth)) & typeMask
		} else {
			// Check if we have reached end of span.
			// n is an overestimate of the size of the object.
			if (b+i)%_PageSize == 0 && h_spans[(b-arena_start)>>_PageShift] != h_spans[(b+i-arena_start)>>_PageShift] {
				break
			}

			bits = uintptr(hbits.typeBits())
			if i > 0 && (hbits.isBoundary() || bits == typeDead) {
				break // reached beginning of the next object
			}
			hbits = hbits.next()
		}

		if bits <= typeScalar { // typeScalar, typeDead, typeScalarMarked
			continue
		}

		if bits&typePointer != typePointer {
			print("gc useCheckmark=", useCheckmark, " b=", hex(b), " ptrmask=", ptrmask, "\n")
			throw("unexpected garbage collection bits")
		}

		obj := *(*uintptr)(unsafe.Pointer(b + i))

		// At this point we have extracted the next potential pointer.
		// Check if it points into heap.
		if obj == 0 || obj < arena_start || obj >= arena_used {
			continue
		}

		if mheap_.shadow_enabled && debug.wbshadow >= 2 && debug.gccheckmark > 0 && useCheckmark {
			checkwbshadow((*uintptr)(unsafe.Pointer(b + i)))
		}

		// Mark the object.
		if obj, hbits := heapBitsForObject(obj); obj != 0 {
			greyobject(obj, b, i, hbits, gcw)
		}
	}
}

// Shade the object if it isn't already.
// The object is not nil and known to be in the heap.
//go:nowritebarrier
func shade(b uintptr) {
	if !inheap(b) {
		throw("shade: passed an address not in the heap")
	}
	if obj, hbits := heapBitsForObject(b); obj != 0 {
		// TODO: this would be a great place to put a check to see
		// if we are harvesting and if we are then we should
		// figure out why there is a call to shade when the
		// harvester thinks we are in a STW.
		// if atomicload(&harvestingwbufs) == uint32(1) {
		//	// Throw here to discover write barriers
		//	// being executed during a STW.
		//	throw("shade during harvest")
		// }

		var gcw gcWorkProducer
		greyobject(obj, 0, 0, hbits, &gcw)
		// This is part of the write barrier so put the wbuf back.
		if gcphase == _GCmarktermination {
			gcw.dispose()
		} else {
			// If we added any pointers to the gcw, then
			// currentwbuf must be nil because 1)
			// greyobject got its wbuf from currentwbuf
			// and 2) shade runs on the systemstack, so
			// we're still on the same M.  If either of
			// these becomes no longer true, we need to
			// rethink this.
			gcw.disposeToCache()
		}
	}
}

// obj is the start of an object with mark mbits.
// If it isn't already marked, mark it and enqueue into workbuf.
// Return possibly new workbuf to use.
// base and off are for debugging only and could be removed.
//go:nowritebarrier
func greyobject(obj, base, off uintptr, hbits heapBits, gcw *gcWorkProducer) {
	// obj should be start of allocation, and so must be at least pointer-aligned.
	if obj&(ptrSize-1) != 0 {
		throw("greyobject: obj not pointer-aligned")
	}

	if useCheckmark {
		if !hbits.isMarked() {
			print("runtime:greyobject: checkmarks finds unexpected unmarked object obj=", hex(obj), "\n")
			print("runtime: found obj at *(", hex(base), "+", hex(off), ")\n")

			// Dump the source (base) object

			kb := base >> _PageShift
			xb := kb
			xb -= mheap_.arena_start >> _PageShift
			sb := h_spans[xb]
			printlock()
			print("runtime:greyobject Span: base=", hex(base), " kb=", hex(kb))
			if sb == nil {
				print(" sb=nil\n")
			} else {
				print(" sb.start*_PageSize=", hex(sb.start*_PageSize), " sb.limit=", hex(sb.limit), " sb.sizeclass=", sb.sizeclass, " sb.elemsize=", sb.elemsize, "\n")
				// base is (a pointer to) the source object holding the reference to object. Create a pointer to each of the fields
				// fields in base and print them out as hex values.
				for i := 0; i < int(sb.elemsize/ptrSize); i++ {
					print(" *(base+", i*ptrSize, ") = ", hex(*(*uintptr)(unsafe.Pointer(base + uintptr(i)*ptrSize))), "\n")
				}
			}

			// Dump the object

			k := obj >> _PageShift
			x := k
			x -= mheap_.arena_start >> _PageShift
			s := h_spans[x]
			print("runtime:greyobject Span: obj=", hex(obj), " k=", hex(k))
			if s == nil {
				print(" s=nil\n")
			} else {
				print(" s.start=", hex(s.start*_PageSize), " s.limit=", hex(s.limit), " s.sizeclass=", s.sizeclass, " s.elemsize=", s.elemsize, "\n")
				// NOTE(rsc): This code is using s.sizeclass as an approximation of the
				// number of pointer-sized words in an object. Perhaps not what was intended.
				for i := 0; i < int(s.sizeclass); i++ {
					print(" *(obj+", i*ptrSize, ") = ", hex(*(*uintptr)(unsafe.Pointer(obj + uintptr(i)*ptrSize))), "\n")
				}
			}
			throw("checkmark found unmarked object")
		}
		if !hbits.isCheckmarked() {
			return
		}
		hbits.setCheckmarked()
		if !hbits.isCheckmarked() {
			throw("setCheckmarked and isCheckmarked disagree")
		}
	} else {
		// If marked we have nothing to do.
		if hbits.isMarked() {
			return
		}

		// Each byte of GC bitmap holds info for two words.
		// Might be racing with other updates, so use atomic update always.
		// We used to be clever here and use a non-atomic update in certain
		// cases, but it's not worth the risk.
		hbits.setMarked()
	}

	if !useCheckmark && hbits.typeBits() == typeDead {
		return // noscan object
	}

	// Queue the obj for scanning. The PREFETCH(obj) logic has been removed but
	// seems like a nice optimization that can be added back in.
	// There needs to be time between the PREFETCH and the use.
	// Previously we put the obj in an 8 element buffer that is drained at a rate
	// to give the PREFETCH time to do its work.
	// Use of PREFETCHNTA might be more appropriate than PREFETCH

	gcw.put(obj)
}

// When in GCmarkterminate phase we allocate black.
//go:nowritebarrier
func gcmarknewobject_m(obj uintptr) {
	if gcphase != _GCmarktermination {
		throw("marking new object while not in mark termination phase")
	}
	if useCheckmark { // The world should be stopped so this should not happen.
		throw("gcmarknewobject called while doing checkmark")
	}

	heapBitsForAddr(obj).setMarked()
}

// Checkmarking

// To help debug the concurrent GC we remark with the world
// stopped ensuring that any object encountered has their normal
// mark bit set. To do this we use an orthogonal bit
// pattern to indicate the object is marked. The following pattern
// uses the upper two bits in the object's bounday nibble.
// 01: scalar  not marked
// 10: pointer not marked
// 11: pointer     marked
// 00: scalar      marked
// Xoring with 01 will flip the pattern from marked to unmarked and vica versa.
// The higher bit is 1 for pointers and 0 for scalars, whether the object
// is marked or not.
// The first nibble no longer holds the typeDead pattern indicating that the
// there are no more pointers in the object. This information is held
// in the second nibble.

// If useCheckmark is true, marking of an object uses the
// checkmark bits (encoding above) instead of the standard
// mark bits.
var useCheckmark = false

//go:nowritebarrier
func initCheckmarks() {
	useCheckmark = true
	for _, s := range work.spans {
		if s.state == _MSpanInUse {
			heapBitsForSpan(s.base()).initCheckmarkSpan(s.layout())
		}
	}
}

func clearCheckmarks() {
	useCheckmark = false
	for _, s := range work.spans {
		if s.state == _MSpanInUse {
			heapBitsForSpan(s.base()).clearCheckmarkSpan(s.layout())
		}
	}
}
