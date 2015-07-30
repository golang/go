// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	_Debugwbufs  = false   // if true check wbufs consistency
	_WorkbufSize = 1 * 256 // in bytes - if small wbufs are passed to GC in a timely fashion.
)

// Garbage collector work pool abstraction.
//
// This implements a producer/consumer model for pointers to grey
// objects.  A grey object is one that is marked and on a work
// queue.  A black object is marked and not on a work queue.
//
// Write barriers, root discovery, stack scanning, and object scanning
// produce pointers to grey objects.  Scanning consumes pointers to
// grey objects, thus blackening them, and then scans them,
// potentially producing new pointers to grey objects.

// A wbufptr holds a workbuf*, but protects it from write barriers.
// workbufs never live on the heap, so write barriers are unnecessary.
// Write barriers on workbuf pointers may also be dangerous in the GC.
type wbufptr uintptr

func wbufptrOf(w *workbuf) wbufptr {
	return wbufptr(unsafe.Pointer(w))
}

func (wp wbufptr) ptr() *workbuf {
	return (*workbuf)(unsafe.Pointer(wp))
}

// A gcWork provides the interface to produce and consume work for the
// garbage collector.
//
// A gcWork can be used on the stack as follows:
//
//     var gcw gcWork
//     disable preemption
//     .. call gcw.put() to produce and gcw.get() to consume ..
//     gcw.dispose()
//     enable preemption
//
// Or from the per-P gcWork cache:
//
//     (preemption must be disabled)
//     gcw := &getg().m.p.ptr().gcw
//     .. call gcw.put() to produce and gcw.get() to consume ..
//     if gcphase == _GCmarktermination {
//         gcw.dispose()
//     }
//
// It's important that any use of gcWork during the mark phase prevent
// the garbage collector from transitioning to mark termination since
// gcWork may locally hold GC work buffers. This can be done by
// disabling preemption (systemstack or acquirem).
type gcWork struct {
	// Invariant: wbuf is never full or empty
	wbuf wbufptr

	// Bytes marked (blackened) on this gcWork. This is aggregated
	// into work.bytesMarked by dispose.
	bytesMarked uint64

	// Scan work performed on this gcWork. This is aggregated into
	// gcController by dispose.
	scanWork int64
}

// put enqueues a pointer for the garbage collector to trace.
// obj must point to the beginning of a heap object.
//go:nowritebarrier
func (ww *gcWork) put(obj uintptr) {
	w := (*gcWork)(noescape(unsafe.Pointer(ww))) // TODO: remove when escape analysis is fixed

	wbuf := w.wbuf.ptr()
	if wbuf == nil {
		wbuf = getpartialorempty(42)
		w.wbuf = wbufptrOf(wbuf)
	}

	wbuf.obj[wbuf.nobj] = obj
	wbuf.nobj++

	if wbuf.nobj == len(wbuf.obj) {
		putfull(wbuf, 50)
		w.wbuf = 0
	}
}

// tryGet dequeues a pointer for the garbage collector to trace.
//
// If there are no pointers remaining in this gcWork or in the global
// queue, tryGet returns 0.  Note that there may still be pointers in
// other gcWork instances or other caches.
//go:nowritebarrier
func (ww *gcWork) tryGet() uintptr {
	w := (*gcWork)(noescape(unsafe.Pointer(ww))) // TODO: remove when escape analysis is fixed

	wbuf := w.wbuf.ptr()
	if wbuf == nil {
		wbuf = trygetfull(74)
		if wbuf == nil {
			return 0
		}
		w.wbuf = wbufptrOf(wbuf)
	}

	wbuf.nobj--
	obj := wbuf.obj[wbuf.nobj]

	if wbuf.nobj == 0 {
		putempty(wbuf, 86)
		w.wbuf = 0
	}

	return obj
}

// get dequeues a pointer for the garbage collector to trace, blocking
// if necessary to ensure all pointers from all queues and caches have
// been retrieved.  get returns 0 if there are no pointers remaining.
//go:nowritebarrier
func (ww *gcWork) get() uintptr {
	w := (*gcWork)(noescape(unsafe.Pointer(ww))) // TODO: remove when escape analysis is fixed

	wbuf := w.wbuf.ptr()
	if wbuf == nil {
		wbuf = getfull(103)
		if wbuf == nil {
			return 0
		}
		wbuf.checknonempty()
		w.wbuf = wbufptrOf(wbuf)
	}

	// TODO: This might be a good place to add prefetch code

	wbuf.nobj--
	obj := wbuf.obj[wbuf.nobj]

	if wbuf.nobj == 0 {
		putempty(wbuf, 115)
		w.wbuf = 0
	}

	return obj
}

// dispose returns any cached pointers to the global queue.
// The buffers are being put on the full queue so that the
// write barriers will not simply reacquire them before the
// GC can inspect them. This helps reduce the mutator's
// ability to hide pointers during the concurrent mark phase.
//
//go:nowritebarrier
func (w *gcWork) dispose() {
	if wbuf := w.wbuf; wbuf != 0 {
		if wbuf.ptr().nobj == 0 {
			throw("dispose: workbuf is empty")
		}
		putfull(wbuf.ptr(), 166)
		w.wbuf = 0
	}
	if w.bytesMarked != 0 {
		// dispose happens relatively infrequently. If this
		// atomic becomes a problem, we should first try to
		// dispose less and if necessary aggregate in a per-P
		// counter.
		xadd64(&work.bytesMarked, int64(w.bytesMarked))
		w.bytesMarked = 0
	}
	if w.scanWork != 0 {
		xaddint64(&gcController.scanWork, w.scanWork)
		w.scanWork = 0
	}
}

// balance moves some work that's cached in this gcWork back on the
// global queue.
//go:nowritebarrier
func (w *gcWork) balance() {
	if wbuf := w.wbuf; wbuf != 0 && wbuf.ptr().nobj > 4 {
		w.wbuf = wbufptrOf(handoff(wbuf.ptr()))
	}
}

// empty returns true if w has no mark work available.
//go:nowritebarrier
func (w *gcWork) empty() bool {
	wbuf := w.wbuf
	return wbuf == 0 || wbuf.ptr().nobj == 0
}

// Internally, the GC work pool is kept in arrays in work buffers.
// The gcWork interface caches a work buffer until full (or empty) to
// avoid contending on the global work buffer lists.

type workbufhdr struct {
	node  lfnode // must be first
	nobj  int
	inuse bool   // This workbuf is in use by some gorotuine and is not on the work.empty/partial/full queues.
	log   [4]int // line numbers forming a history of ownership changes to workbuf
}

type workbuf struct {
	workbufhdr
	// account for the above fields
	obj [(_WorkbufSize - unsafe.Sizeof(workbufhdr{})) / ptrSize]uintptr
}

// workbuf factory routines. These funcs are used to manage the
// workbufs.
// If the GC asks for some work these are the only routines that
// make partially full wbufs available to the GC.
// Each of the gets and puts also take an distinct integer that is used
// to record a brief history of changes to ownership of the workbuf.
// The convention is to use a unique line number but any encoding
// is permissible. For example if you want to pass in 2 bits of information
// you could simple add lineno1*100000+lineno2.

// logget records the past few values of entry to aid in debugging.
// logget checks the buffer b is not currently in use.
func (b *workbuf) logget(entry int) {
	if !_Debugwbufs {
		return
	}
	if b.inuse {
		println("runtime: logget fails log entry=", entry,
			"b.log[0]=", b.log[0], "b.log[1]=", b.log[1],
			"b.log[2]=", b.log[2], "b.log[3]=", b.log[3])
		throw("logget: get not legal")
	}
	b.inuse = true
	copy(b.log[1:], b.log[:])
	b.log[0] = entry
}

// logput records the past few values of entry to aid in debugging.
// logput checks the buffer b is currently in use.
func (b *workbuf) logput(entry int) {
	if !_Debugwbufs {
		return
	}
	if !b.inuse {
		println("runtime: logput fails log entry=", entry,
			"b.log[0]=", b.log[0], "b.log[1]=", b.log[1],
			"b.log[2]=", b.log[2], "b.log[3]=", b.log[3])
		throw("logput: put not legal")
	}
	b.inuse = false
	copy(b.log[1:], b.log[:])
	b.log[0] = entry
}

func (b *workbuf) checknonempty() {
	if b.nobj == 0 {
		println("runtime: nonempty check fails",
			"b.log[0]=", b.log[0], "b.log[1]=", b.log[1],
			"b.log[2]=", b.log[2], "b.log[3]=", b.log[3])
		throw("workbuf is empty")
	}
}

func (b *workbuf) checkempty() {
	if b.nobj != 0 {
		println("runtime: empty check fails",
			"b.log[0]=", b.log[0], "b.log[1]=", b.log[1],
			"b.log[2]=", b.log[2], "b.log[3]=", b.log[3])
		throw("workbuf is not empty")
	}
}

// getempty pops an empty work buffer off the work.empty list,
// allocating new buffers if none are available.
// entry is used to record a brief history of ownership.
//go:nowritebarrier
func getempty(entry int) *workbuf {
	var b *workbuf
	if work.empty != 0 {
		b = (*workbuf)(lfstackpop(&work.empty))
		if b != nil {
			b.checkempty()
		}
	}
	if b == nil {
		b = (*workbuf)(persistentalloc(unsafe.Sizeof(*b), _CacheLineSize, &memstats.gc_sys))
	}
	b.logget(entry)
	return b
}

// putempty puts a workbuf onto the work.empty list.
// Upon entry this go routine owns b. The lfstackpush relinquishes ownership.
//go:nowritebarrier
func putempty(b *workbuf, entry int) {
	b.checkempty()
	b.logput(entry)
	lfstackpush(&work.empty, &b.node)
}

// putfull puts the workbuf on the work.full list for the GC.
// putfull accepts partially full buffers so the GC can avoid competing
// with the mutators for ownership of partially full buffers.
//go:nowritebarrier
func putfull(b *workbuf, entry int) {
	b.checknonempty()
	b.logput(entry)
	lfstackpush(&work.full, &b.node)
}

// getpartialorempty tries to return a partially empty
// and if none are available returns an empty one.
// entry is used to provide a brief history of ownership
// using entry + xxx00000 to
// indicating that two line numbers in the call chain.
//go:nowritebarrier
func getpartialorempty(entry int) *workbuf {
	b := (*workbuf)(lfstackpop(&work.partial))
	if b != nil {
		b.logget(entry)
		return b
	}
	// Let getempty do the logget check but
	// use the entry to encode that it passed
	// through this routine.
	b = getempty(entry + 80700000)
	return b
}

// putpartial puts empty buffers on the work.empty queue,
// full buffers on the work.full queue and
// others on the work.partial queue.
// entry is used to provide a brief history of ownership
// using entry + xxx00000 to
// indicating that two call chain line numbers.
//go:nowritebarrier
func putpartial(b *workbuf, entry int) {
	if b.nobj == 0 {
		putempty(b, entry+81500000)
	} else if b.nobj < len(b.obj) {
		b.logput(entry)
		lfstackpush(&work.partial, &b.node)
	} else if b.nobj == len(b.obj) {
		b.logput(entry)
		lfstackpush(&work.full, &b.node)
	} else {
		throw("putpartial: bad Workbuf b.nobj")
	}
}

// trygetfull tries to get a full or partially empty workbuffer.
// If one is not immediately available return nil
//go:nowritebarrier
func trygetfull(entry int) *workbuf {
	b := (*workbuf)(lfstackpop(&work.full))
	if b == nil {
		b = (*workbuf)(lfstackpop(&work.partial))
	}
	if b != nil {
		b.logget(entry)
		b.checknonempty()
		return b
	}
	return b
}

// Get a full work buffer off the work.full or a partially
// filled one off the work.partial list. If nothing is available
// wait until all the other gc helpers have finished and then
// return nil.
// getfull acts as a barrier for work.nproc helpers. As long as one
// gchelper is actively marking objects it
// may create a workbuffer that the other helpers can work on.
// The for loop either exits when a work buffer is found
// or when _all_ of the work.nproc GC helpers are in the loop
// looking for work and thus not capable of creating new work.
// This is in fact the termination condition for the STW mark
// phase.
//go:nowritebarrier
func getfull(entry int) *workbuf {
	b := (*workbuf)(lfstackpop(&work.full))
	if b != nil {
		b.logget(entry)
		b.checknonempty()
		return b
	}
	b = (*workbuf)(lfstackpop(&work.partial))
	if b != nil {
		b.logget(entry)
		return b
	}

	incnwait := xadd(&work.nwait, +1)
	if incnwait > work.nproc {
		println("runtime: work.nwait=", incnwait, "work.nproc=", work.nproc)
		throw("work.nwait > work.nproc")
	}
	for i := 0; ; i++ {
		if work.full != 0 || work.partial != 0 {
			decnwait := xadd(&work.nwait, -1)
			if decnwait == work.nproc {
				println("runtime: work.nwait=", decnwait, "work.nproc=", work.nproc)
				throw("work.nwait > work.nproc")
			}
			b = (*workbuf)(lfstackpop(&work.full))
			if b == nil {
				b = (*workbuf)(lfstackpop(&work.partial))
			}
			if b != nil {
				b.logget(entry)
				b.checknonempty()
				return b
			}
			incnwait := xadd(&work.nwait, +1)
			if incnwait > work.nproc {
				println("runtime: work.nwait=", incnwait, "work.nproc=", work.nproc)
				throw("work.nwait > work.nproc")
			}
		}
		if work.nwait == work.nproc {
			return nil
		}
		_g_ := getg()
		if i < 10 {
			_g_.m.gcstats.nprocyield++
			procyield(20)
		} else if i < 20 {
			_g_.m.gcstats.nosyield++
			osyield()
		} else {
			_g_.m.gcstats.nsleep++
			usleep(100)
		}
	}
}

//go:nowritebarrier
func handoff(b *workbuf) *workbuf {
	// Make new buffer with half of b's pointers.
	b1 := getempty(915)
	n := b.nobj / 2
	b.nobj -= n
	b1.nobj = n
	memmove(unsafe.Pointer(&b1.obj[0]), unsafe.Pointer(&b.obj[b.nobj]), uintptr(n)*unsafe.Sizeof(b1.obj[0]))
	_g_ := getg()
	_g_.m.gcstats.nhandoff++
	_g_.m.gcstats.nhandoffcnt += uint64(n)

	// Put b on full list - let first half of b get stolen.
	putfull(b, 942)
	return b1
}
