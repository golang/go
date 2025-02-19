// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector: finalizers and block profiling.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"internal/runtime/atomic"
	"internal/runtime/gc"
	"internal/runtime/sys"
	"unsafe"
)

const finBlockSize = 4 * 1024

// finBlock is an block of finalizers to be executed. finBlocks
// are arranged in a linked list for the finalizer queue.
//
// finBlock is allocated from non-GC'd memory, so any heap pointers
// must be specially handled. GC currently assumes that the finalizer
// queue does not grow during marking (but it can shrink).
type finBlock struct {
	_       sys.NotInHeap
	alllink *finBlock
	next    *finBlock
	cnt     uint32
	_       int32
	fin     [(finBlockSize - 2*goarch.PtrSize - 2*4) / unsafe.Sizeof(finalizer{})]finalizer
}

var fingStatus atomic.Uint32

// finalizer goroutine status.
const (
	fingUninitialized uint32 = iota
	fingCreated       uint32 = 1 << (iota - 1)
	fingRunningFinalizer
	fingWait
	fingWake
)

var (
	finlock    mutex     // protects the following variables
	fing       *g        // goroutine that runs finalizers
	finq       *finBlock // list of finalizers that are to be executed
	finc       *finBlock // cache of free blocks
	finptrmask [finBlockSize / goarch.PtrSize / 8]byte
)

var allfin *finBlock // list of all blocks

// NOTE: Layout known to queuefinalizer.
type finalizer struct {
	fn   *funcval       // function to call (may be a heap pointer)
	arg  unsafe.Pointer // ptr to object (may be a heap pointer)
	nret uintptr        // bytes of return values from fn
	fint *_type         // type of first argument of fn
	ot   *ptrtype       // type of ptr to object (may be a heap pointer)
}

var finalizer1 = [...]byte{
	// Each Finalizer is 5 words, ptr ptr INT ptr ptr (INT = uintptr here)
	// Each byte describes 8 words.
	// Need 8 Finalizers described by 5 bytes before pattern repeats:
	//	ptr ptr INT ptr ptr
	//	ptr ptr INT ptr ptr
	//	ptr ptr INT ptr ptr
	//	ptr ptr INT ptr ptr
	//	ptr ptr INT ptr ptr
	//	ptr ptr INT ptr ptr
	//	ptr ptr INT ptr ptr
	//	ptr ptr INT ptr ptr
	// aka
	//
	//	ptr ptr INT ptr ptr ptr ptr INT
	//	ptr ptr ptr ptr INT ptr ptr ptr
	//	ptr INT ptr ptr ptr ptr INT ptr
	//	ptr ptr ptr INT ptr ptr ptr ptr
	//	INT ptr ptr ptr ptr INT ptr ptr
	//
	// Assumptions about Finalizer layout checked below.
	1<<0 | 1<<1 | 0<<2 | 1<<3 | 1<<4 | 1<<5 | 1<<6 | 0<<7,
	1<<0 | 1<<1 | 1<<2 | 1<<3 | 0<<4 | 1<<5 | 1<<6 | 1<<7,
	1<<0 | 0<<1 | 1<<2 | 1<<3 | 1<<4 | 1<<5 | 0<<6 | 1<<7,
	1<<0 | 1<<1 | 1<<2 | 0<<3 | 1<<4 | 1<<5 | 1<<6 | 1<<7,
	0<<0 | 1<<1 | 1<<2 | 1<<3 | 1<<4 | 0<<5 | 1<<6 | 1<<7,
}

// lockRankMayQueueFinalizer records the lock ranking effects of a
// function that may call queuefinalizer.
func lockRankMayQueueFinalizer() {
	lockWithRankMayAcquire(&finlock, getLockRank(&finlock))
}

func queuefinalizer(p unsafe.Pointer, fn *funcval, nret uintptr, fint *_type, ot *ptrtype) {
	if gcphase != _GCoff {
		// Currently we assume that the finalizer queue won't
		// grow during marking so we don't have to rescan it
		// during mark termination. If we ever need to lift
		// this assumption, we can do it by adding the
		// necessary barriers to queuefinalizer (which it may
		// have automatically).
		throw("queuefinalizer during GC")
	}

	lock(&finlock)
	if finq == nil || finq.cnt == uint32(len(finq.fin)) {
		if finc == nil {
			finc = (*finBlock)(persistentalloc(finBlockSize, 0, &memstats.gcMiscSys))
			finc.alllink = allfin
			allfin = finc
			if finptrmask[0] == 0 {
				// Build pointer mask for Finalizer array in block.
				// Check assumptions made in finalizer1 array above.
				if (unsafe.Sizeof(finalizer{}) != 5*goarch.PtrSize ||
					unsafe.Offsetof(finalizer{}.fn) != 0 ||
					unsafe.Offsetof(finalizer{}.arg) != goarch.PtrSize ||
					unsafe.Offsetof(finalizer{}.nret) != 2*goarch.PtrSize ||
					unsafe.Offsetof(finalizer{}.fint) != 3*goarch.PtrSize ||
					unsafe.Offsetof(finalizer{}.ot) != 4*goarch.PtrSize) {
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
	atomic.Xadd(&finq.cnt, +1) // Sync with markroots
	f.fn = fn
	f.nret = nret
	f.fint = fint
	f.ot = ot
	f.arg = p
	unlock(&finlock)
	fingStatus.Or(fingWake)
}

//go:nowritebarrier
func iterate_finq(callback func(*funcval, unsafe.Pointer, uintptr, *_type, *ptrtype)) {
	for fb := allfin; fb != nil; fb = fb.alllink {
		for i := uint32(0); i < fb.cnt; i++ {
			f := &fb.fin[i]
			callback(f.fn, f.arg, f.nret, f.fint, f.ot)
		}
	}
}

func wakefing() *g {
	if ok := fingStatus.CompareAndSwap(fingCreated|fingWait|fingWake, fingCreated); ok {
		return fing
	}
	return nil
}

func createfing() {
	// start the finalizer goroutine exactly once
	if fingStatus.Load() == fingUninitialized && fingStatus.CompareAndSwap(fingUninitialized, fingCreated) {
		go runFinalizers()
	}
}

func finalizercommit(gp *g, lock unsafe.Pointer) bool {
	unlock((*mutex)(lock))
	// fingStatus should be modified after fing is put into a waiting state
	// to avoid waking fing in running state, even if it is about to be parked.
	fingStatus.Or(fingWait)
	return true
}

// This is the goroutine that runs all of the finalizers.
func runFinalizers() {
	var (
		frame    unsafe.Pointer
		framecap uintptr
		argRegs  int
	)

	gp := getg()
	lock(&finlock)
	fing = gp
	unlock(&finlock)

	for {
		lock(&finlock)
		fb := finq
		finq = nil
		if fb == nil {
			gopark(finalizercommit, unsafe.Pointer(&finlock), waitReasonFinalizerWait, traceBlockSystemGoroutine, 1)
			continue
		}
		argRegs = intArgRegs
		unlock(&finlock)
		if raceenabled {
			racefingo()
		}
		for fb != nil {
			for i := fb.cnt; i > 0; i-- {
				f := &fb.fin[i-1]

				var regs abi.RegArgs
				// The args may be passed in registers or on stack. Even for
				// the register case, we still need the spill slots.
				// TODO: revisit if we remove spill slots.
				//
				// Unfortunately because we can have an arbitrary
				// amount of returns and it would be complex to try and
				// figure out how many of those can get passed in registers,
				// just conservatively assume none of them do.
				framesz := unsafe.Sizeof((any)(nil)) + f.nret
				if framecap < framesz {
					// The frame does not contain pointers interesting for GC,
					// all not yet finalized objects are stored in finq.
					// If we do not mark it as FlagNoScan,
					// the last finalized object is not collected.
					frame = mallocgc(framesz, nil, true)
					framecap = framesz
				}
				if f.fint == nil {
					throw("missing type in finalizer")
				}
				r := frame
				if argRegs > 0 {
					r = unsafe.Pointer(&regs.Ints)
				} else {
					// frame is effectively uninitialized
					// memory. That means we have to clear
					// it before writing to it to avoid
					// confusing the write barrier.
					*(*[2]uintptr)(frame) = [2]uintptr{}
				}
				switch f.fint.Kind_ & abi.KindMask {
				case abi.Pointer:
					// direct use of pointer
					*(*unsafe.Pointer)(r) = f.arg
				case abi.Interface:
					ityp := (*interfacetype)(unsafe.Pointer(f.fint))
					// set up with empty interface
					(*eface)(r)._type = &f.ot.Type
					(*eface)(r).data = f.arg
					if len(ityp.Methods) != 0 {
						// convert to interface with methods
						// this conversion is guaranteed to succeed - we checked in SetFinalizer
						(*iface)(r).tab = assertE2I(ityp, (*eface)(r)._type)
					}
				default:
					throw("bad type kind in finalizer")
				}
				fingStatus.Or(fingRunningFinalizer)
				reflectcall(nil, unsafe.Pointer(f.fn), frame, uint32(framesz), uint32(framesz), uint32(framesz), &regs)
				fingStatus.And(^fingRunningFinalizer)

				// Drop finalizer queue heap references
				// before hiding them from markroot.
				// This also ensures these will be
				// clear if we reuse the finalizer.
				f.fn = nil
				f.arg = nil
				f.ot = nil
				atomic.Store(&fb.cnt, i-1)
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

func isGoPointerWithoutSpan(p unsafe.Pointer) bool {
	// 0-length objects are okay.
	if p == unsafe.Pointer(&zerobase) {
		return true
	}

	// Global initializers might be linker-allocated.
	//	var Foo = &Object{}
	//	func main() {
	//		runtime.SetFinalizer(Foo, nil)
	//	}
	// The relevant segments are: noptrdata, data, bss, noptrbss.
	// We cannot assume they are in any order or even contiguous,
	// due to external linking.
	for datap := &firstmoduledata; datap != nil; datap = datap.next {
		if datap.noptrdata <= uintptr(p) && uintptr(p) < datap.enoptrdata ||
			datap.data <= uintptr(p) && uintptr(p) < datap.edata ||
			datap.bss <= uintptr(p) && uintptr(p) < datap.ebss ||
			datap.noptrbss <= uintptr(p) && uintptr(p) < datap.enoptrbss {
			return true
		}
	}
	return false
}

// blockUntilEmptyFinalizerQueue blocks until either the finalizer
// queue is emptied (and the finalizers have executed) or the timeout
// is reached. Returns true if the finalizer queue was emptied.
// This is used by the runtime, sync, and unique tests.
func blockUntilEmptyFinalizerQueue(timeout int64) bool {
	start := nanotime()
	for nanotime()-start < timeout {
		lock(&finlock)
		// We know the queue has been drained when both finq is nil
		// and the finalizer g has stopped executing.
		empty := finq == nil
		empty = empty && readgstatus(fing) == _Gwaiting && fing.waitreason == waitReasonFinalizerWait
		unlock(&finlock)
		if empty {
			return true
		}
		Gosched()
	}
	return false
}

//go:linkname unique_runtime_blockUntilEmptyFinalizerQueue unique.runtime_blockUntilEmptyFinalizerQueue
func unique_runtime_blockUntilEmptyFinalizerQueue(timeout int64) bool {
	return blockUntilEmptyFinalizerQueue(timeout)
}

// SetFinalizer sets the finalizer associated with obj to the provided
// finalizer function. When the garbage collector finds an unreachable block
// with an associated finalizer, it clears the association and runs
// finalizer(obj) in a separate goroutine. This makes obj reachable again,
// but now without an associated finalizer. Assuming that SetFinalizer
// is not called again, the next time the garbage collector sees
// that obj is unreachable, it will free obj.
//
// SetFinalizer(obj, nil) clears any finalizer associated with obj.
//
// New Go code should consider using [AddCleanup] instead, which is much
// less error-prone than SetFinalizer.
//
// The argument obj must be a pointer to an object allocated by calling
// new, by taking the address of a composite literal, or by taking the
// address of a local variable.
// The argument finalizer must be a function that takes a single argument
// to which obj's type can be assigned, and can have arbitrary ignored return
// values. If either of these is not true, SetFinalizer may abort the
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
// The finalizer is scheduled to run at some arbitrary time after the
// program can no longer reach the object to which obj points.
// There is no guarantee that finalizers will run before a program exits,
// so typically they are useful only for releasing non-memory resources
// associated with an object during a long-running program.
// For example, an [os.File] object could use a finalizer to close the
// associated operating system file descriptor when a program discards
// an os.File without calling Close, but it would be a mistake
// to depend on a finalizer to flush an in-memory I/O buffer such as a
// [bufio.Writer], because the buffer would not be flushed at program exit.
//
// It is not guaranteed that a finalizer will run if the size of *obj is
// zero bytes, because it may share same address with other zero-size
// objects in memory. See https://go.dev/ref/spec#Size_and_alignment_guarantees.
//
// It is not guaranteed that a finalizer will run for objects allocated
// in initializers for package-level variables. Such objects may be
// linker-allocated, not heap-allocated.
//
// Note that because finalizers may execute arbitrarily far into the future
// after an object is no longer referenced, the runtime is allowed to perform
// a space-saving optimization that batches objects together in a single
// allocation slot. The finalizer for an unreferenced object in such an
// allocation may never run if it always exists in the same batch as a
// referenced object. Typically, this batching only happens for tiny
// (on the order of 16 bytes or less) and pointer-free objects.
//
// A finalizer may run as soon as an object becomes unreachable.
// In order to use finalizers correctly, the program must ensure that
// the object is reachable until it is no longer required.
// Objects stored in global variables, or that can be found by tracing
// pointers from a global variable, are reachable. A function argument or
// receiver may become unreachable at the last point where the function
// mentions it. To make an unreachable object reachable, pass the object
// to a call of the [KeepAlive] function to mark the last point in the
// function where the object must be reachable.
//
// For example, if p points to a struct, such as os.File, that contains
// a file descriptor d, and p has a finalizer that closes that file
// descriptor, and if the last use of p in a function is a call to
// syscall.Write(p.d, buf, size), then p may be unreachable as soon as
// the program enters [syscall.Write]. The finalizer may run at that moment,
// closing p.d, causing syscall.Write to fail because it is writing to
// a closed file descriptor (or, worse, to an entirely different
// file descriptor opened by a different goroutine). To avoid this problem,
// call KeepAlive(p) after the call to syscall.Write.
//
// A single goroutine runs all finalizers for a program, sequentially.
// If a finalizer must run for a long time, it should do so by starting
// a new goroutine.
//
// In the terminology of the Go memory model, a call
// SetFinalizer(x, f) “synchronizes before” the finalization call f(x).
// However, there is no guarantee that KeepAlive(x) or any other use of x
// “synchronizes before” f(x), so in general a finalizer should use a mutex
// or other synchronization mechanism if it needs to access mutable state in x.
// For example, consider a finalizer that inspects a mutable field in x
// that is modified from time to time in the main program before x
// becomes unreachable and the finalizer is invoked.
// The modifications in the main program and the inspection in the finalizer
// need to use appropriate synchronization, such as mutexes or atomic updates,
// to avoid read-write races.
func SetFinalizer(obj any, finalizer any) {
	e := efaceOf(&obj)
	etyp := e._type
	if etyp == nil {
		throw("runtime.SetFinalizer: first argument is nil")
	}
	if etyp.Kind_&abi.KindMask != abi.Pointer {
		throw("runtime.SetFinalizer: first argument is " + toRType(etyp).string() + ", not pointer")
	}
	ot := (*ptrtype)(unsafe.Pointer(etyp))
	if ot.Elem == nil {
		throw("nil elem type!")
	}
	if inUserArenaChunk(uintptr(e.data)) {
		// Arena-allocated objects are not eligible for finalizers.
		throw("runtime.SetFinalizer: first argument was allocated into an arena")
	}
	if debug.sbrk != 0 {
		// debug.sbrk never frees memory, so no finalizers run
		// (and we don't have the data structures to record them).
		return
	}

	// find the containing object
	base, span, _ := findObject(uintptr(e.data), 0, 0)

	if base == 0 {
		if isGoPointerWithoutSpan(e.data) {
			return
		}
		throw("runtime.SetFinalizer: pointer not in allocated block")
	}

	// Move base forward if we've got an allocation header.
	if !span.spanclass.noscan() && !heapBitsInSpan(span.elemsize) && span.spanclass.sizeclass() != 0 {
		base += gc.MallocHeaderSize
	}

	if uintptr(e.data) != base {
		// As an implementation detail we allow to set finalizers for an inner byte
		// of an object if it could come from tiny alloc (see mallocgc for details).
		if ot.Elem == nil || ot.Elem.Pointers() || ot.Elem.Size_ >= maxTinySize {
			throw("runtime.SetFinalizer: pointer not at beginning of allocated block")
		}
	}

	f := efaceOf(&finalizer)
	ftyp := f._type
	if ftyp == nil {
		// switch to system stack and remove finalizer
		systemstack(func() {
			removefinalizer(e.data)
		})
		return
	}

	if ftyp.Kind_&abi.KindMask != abi.Func {
		throw("runtime.SetFinalizer: second argument is " + toRType(ftyp).string() + ", not a function")
	}
	ft := (*functype)(unsafe.Pointer(ftyp))
	if ft.IsVariadic() {
		throw("runtime.SetFinalizer: cannot pass " + toRType(etyp).string() + " to finalizer " + toRType(ftyp).string() + " because dotdotdot")
	}
	if ft.InCount != 1 {
		throw("runtime.SetFinalizer: cannot pass " + toRType(etyp).string() + " to finalizer " + toRType(ftyp).string())
	}
	fint := ft.InSlice()[0]
	switch {
	case fint == etyp:
		// ok - same type
		goto okarg
	case fint.Kind_&abi.KindMask == abi.Pointer:
		if (fint.Uncommon() == nil || etyp.Uncommon() == nil) && (*ptrtype)(unsafe.Pointer(fint)).Elem == ot.Elem {
			// ok - not same type, but both pointers,
			// one or the other is unnamed, and same element type, so assignable.
			goto okarg
		}
	case fint.Kind_&abi.KindMask == abi.Interface:
		ityp := (*interfacetype)(unsafe.Pointer(fint))
		if len(ityp.Methods) == 0 {
			// ok - satisfies empty interface
			goto okarg
		}
		if itab := assertE2I2(ityp, efaceOf(&obj)._type); itab != nil {
			goto okarg
		}
	}
	throw("runtime.SetFinalizer: cannot pass " + toRType(etyp).string() + " to finalizer " + toRType(ftyp).string())
okarg:
	// compute size needed for return parameters
	nret := uintptr(0)
	for _, t := range ft.OutSlice() {
		nret = alignUp(nret, uintptr(t.Align_)) + t.Size_
	}
	nret = alignUp(nret, goarch.PtrSize)

	// make sure we have a finalizer goroutine
	createfing()

	systemstack(func() {
		if !addfinalizer(e.data, (*funcval)(f.data), nret, fint, ot) {
			throw("runtime.SetFinalizer: finalizer already set")
		}
	})
}

// Mark KeepAlive as noinline so that it is easily detectable as an intrinsic.
//
//go:noinline

// KeepAlive marks its argument as currently reachable.
// This ensures that the object is not freed, and its finalizer is not run,
// before the point in the program where KeepAlive is called.
//
// A very simplified example showing where KeepAlive is required:
//
//	type File struct { d int }
//	d, err := syscall.Open("/file/path", syscall.O_RDONLY, 0)
//	// ... do something if err != nil ...
//	p := &File{d}
//	runtime.SetFinalizer(p, func(p *File) { syscall.Close(p.d) })
//	var buf [10]byte
//	n, err := syscall.Read(p.d, buf[:])
//	// Ensure p is not finalized until Read returns.
//	runtime.KeepAlive(p)
//	// No more uses of p after this point.
//
// Without the KeepAlive call, the finalizer could run at the start of
// [syscall.Read], closing the file descriptor before syscall.Read makes
// the actual system call.
//
// Note: KeepAlive should only be used to prevent finalizers from
// running prematurely. In particular, when used with [unsafe.Pointer],
// the rules for valid uses of unsafe.Pointer still apply.
func KeepAlive(x any) {
	// Introduce a use of x that the compiler can't eliminate.
	// This makes sure x is alive on entry. We need x to be alive
	// on entry for "defer runtime.KeepAlive(x)"; see issue 21402.
	if cgoAlwaysFalse {
		println(x)
	}
}
