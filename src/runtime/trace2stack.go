// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.exectracer2

// Trace stack table and acquisition.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"unsafe"
)

const (
	// Maximum number of PCs in a single stack trace.
	// Since events contain only stack id rather than whole stack trace,
	// we can allow quite large values here.
	traceStackSize = 128

	// logicalStackSentinel is a sentinel value at pcBuf[0] signifying that
	// pcBuf[1:] holds a logical stack requiring no further processing. Any other
	// value at pcBuf[0] represents a skip value to apply to the physical stack in
	// pcBuf[1:] after inline expansion.
	logicalStackSentinel = ^uintptr(0)
)

// traceStack captures a stack trace from a goroutine and registers it in the trace
// stack table. It then returns its unique ID. If gp == nil, then traceStack will
// attempt to use the current execution context.
//
// skip controls the number of leaf frames to omit in order to hide tracer internals
// from stack traces, see CL 5523.
//
// Avoid calling this function directly. gen needs to be the current generation
// that this stack trace is being written out for, which needs to be synchronized with
// generations moving forward. Prefer traceEventWriter.stack.
func traceStack(skip int, gp *g, gen uintptr) uint64 {
	var pcBuf [traceStackSize]uintptr

	// Figure out gp and mp for the backtrace.
	var mp *m
	if gp == nil {
		mp = getg().m
		gp = mp.curg
	}

	// Double-check that we own the stack we're about to trace.
	if debug.traceCheckStackOwnership != 0 && gp != nil {
		status := readgstatus(gp)
		// If the scan bit is set, assume we're the ones that acquired it.
		if status&_Gscan == 0 {
			// Use the trace status to check this. There are a number of cases
			// where a running goroutine might be in _Gwaiting, and these cases
			// are totally fine for taking a stack trace. They're captured
			// correctly in goStatusToTraceGoStatus.
			switch goStatusToTraceGoStatus(status, gp.waitreason) {
			case traceGoRunning, traceGoSyscall:
				if getg() == gp || mp.curg == gp {
					break
				}
				fallthrough
			default:
				print("runtime: gp=", unsafe.Pointer(gp), " gp.goid=", gp.goid, " status=", gStatusStrings[status], "\n")
				throw("attempted to trace stack of a goroutine this thread does not own")
			}
		}
	}

	if gp != nil && mp == nil {
		// We're getting the backtrace for a G that's not currently executing.
		// It may still have an M, if it's locked to some M.
		mp = gp.lockedm.ptr()
	}
	nstk := 1
	if tracefpunwindoff() || (mp != nil && mp.hasCgoOnStack()) {
		// Slow path: Unwind using default unwinder. Used when frame pointer
		// unwinding is unavailable or disabled (tracefpunwindoff), or might
		// produce incomplete results or crashes (hasCgoOnStack). Note that no
		// cgo callback related crashes have been observed yet. The main
		// motivation is to take advantage of a potentially registered cgo
		// symbolizer.
		pcBuf[0] = logicalStackSentinel
		if getg() == gp {
			nstk += callers(skip+1, pcBuf[1:])
		} else if gp != nil {
			nstk += gcallers(gp, skip, pcBuf[1:])
		}
	} else {
		// Fast path: Unwind using frame pointers.
		pcBuf[0] = uintptr(skip)
		if getg() == gp {
			nstk += fpTracebackPCs(unsafe.Pointer(getfp()), pcBuf[1:])
		} else if gp != nil {
			// Two cases:
			//
			// (1) We're called on the g0 stack through mcall(fn) or systemstack(fn). To
			// behave like gcallers above, we start unwinding from sched.bp, which
			// points to the caller frame of the leaf frame on g's stack. The return
			// address of the leaf frame is stored in sched.pc, which we manually
			// capture here.
			//
			// (2) We're called against a gp that we're not currently executing on, in
			// which case it's currently not executing. gp.sched contains the most up-to-date
			// information about where it stopped, and like case (1), we match gcallers here.
			pcBuf[1] = gp.sched.pc
			nstk += 1 + fpTracebackPCs(unsafe.Pointer(gp.sched.bp), pcBuf[2:])
		}
	}
	if nstk > 0 {
		nstk-- // skip runtime.goexit
	}
	if nstk > 0 && gp.goid == 1 {
		nstk-- // skip runtime.main
	}
	id := trace.stackTab[gen%2].put(pcBuf[:nstk])
	return id
}

// traceStackTable maps stack traces (arrays of PC's) to unique uint32 ids.
// It is lock-free for reading.
type traceStackTable struct {
	tab traceMap
}

// put returns a unique id for the stack trace pcs and caches it in the table,
// if it sees the trace for the first time.
func (t *traceStackTable) put(pcs []uintptr) uint64 {
	if len(pcs) == 0 {
		return 0
	}
	id, _ := t.tab.put(noescape(unsafe.Pointer(&pcs[0])), uintptr(len(pcs))*unsafe.Sizeof(uintptr(0)))
	return id
}

// dump writes all previously cached stacks to trace buffers,
// releases all memory and resets state. It must only be called once the caller
// can guarantee that there are no more writers to the table.
func (t *traceStackTable) dump(gen uintptr) {
	w := unsafeTraceWriter(gen, nil)

	// Iterate over the table.
	//
	// Do not acquire t.tab.lock. There's a conceptual lock cycle between acquiring this lock
	// here and allocation-related locks. Specifically, this lock may be acquired when an event
	// is emitted in allocation paths. Simultaneously, we might allocate here with the lock held,
	// creating a cycle. In practice, this cycle is never exercised. Because the table is only
	// dumped once there are no more writers, it's not possible for the cycle to occur. However
	// the lockrank mode is not sophisticated enough to identify this, and if it's not possible
	// for that cycle to happen, then it's also not possible for this to race with writers to
	// the table.
	for i := range t.tab.tab {
		stk := t.tab.bucket(i)
		for ; stk != nil; stk = stk.next() {
			stack := unsafe.Slice((*uintptr)(unsafe.Pointer(&stk.data[0])), uintptr(len(stk.data))/unsafe.Sizeof(uintptr(0)))

			// N.B. This might allocate, but that's OK because we're not writing to the M's buffer,
			// but one we're about to create (with ensure).
			frames := makeTraceFrames(gen, fpunwindExpand(stack))

			// Returns the maximum number of bytes required to hold the encoded stack, given that
			// it contains N frames.
			maxBytes := 1 + (2+4*len(frames))*traceBytesPerNumber

			// Estimate the size of this record. This
			// bound is pretty loose, but avoids counting
			// lots of varint sizes.
			//
			// Add 1 because we might also write traceEvStacks.
			var flushed bool
			w, flushed = w.ensure(1 + maxBytes)
			if flushed {
				w.byte(byte(traceEvStacks))
			}

			// Emit stack event.
			w.byte(byte(traceEvStack))
			w.varint(uint64(stk.id))
			w.varint(uint64(len(frames)))
			for _, frame := range frames {
				w.varint(uint64(frame.PC))
				w.varint(frame.funcID)
				w.varint(frame.fileID)
				w.varint(frame.line)
			}
		}
	}
	// Still, hold the lock over reset. The callee expects it, even though it's
	// not strictly necessary.
	systemstack(func() {
		lock(&t.tab.lock)
		t.tab.reset()
		unlock(&t.tab.lock)
	})

	w.flush().end()
}

// makeTraceFrames returns the frames corresponding to pcs. It may
// allocate and may emit trace events.
func makeTraceFrames(gen uintptr, pcs []uintptr) []traceFrame {
	frames := make([]traceFrame, 0, len(pcs))
	ci := CallersFrames(pcs)
	for {
		f, more := ci.Next()
		frames = append(frames, makeTraceFrame(gen, f))
		if !more {
			return frames
		}
	}
}

type traceFrame struct {
	PC     uintptr
	funcID uint64
	fileID uint64
	line   uint64
}

// makeTraceFrame sets up a traceFrame for a frame.
func makeTraceFrame(gen uintptr, f Frame) traceFrame {
	var frame traceFrame
	frame.PC = f.PC

	fn := f.Function
	const maxLen = 1 << 10
	if len(fn) > maxLen {
		fn = fn[len(fn)-maxLen:]
	}
	frame.funcID = trace.stringTab[gen%2].put(gen, fn)
	frame.line = uint64(f.Line)
	file := f.File
	if len(file) > maxLen {
		file = file[len(file)-maxLen:]
	}
	frame.fileID = trace.stringTab[gen%2].put(gen, file)
	return frame
}

// tracefpunwindoff returns true if frame pointer unwinding for the tracer is
// disabled via GODEBUG or not supported by the architecture.
func tracefpunwindoff() bool {
	return debug.tracefpunwindoff != 0 || (goarch.ArchFamily != goarch.AMD64 && goarch.ArchFamily != goarch.ARM64)
}

// fpTracebackPCs populates pcBuf with the return addresses for each frame and
// returns the number of PCs written to pcBuf. The returned PCs correspond to
// "physical frames" rather than "logical frames"; that is if A is inlined into
// B, this will return a PC for only B.
func fpTracebackPCs(fp unsafe.Pointer, pcBuf []uintptr) (i int) {
	for i = 0; i < len(pcBuf) && fp != nil; i++ {
		// return addr sits one word above the frame pointer
		pcBuf[i] = *(*uintptr)(unsafe.Pointer(uintptr(fp) + goarch.PtrSize))
		// follow the frame pointer to the next one
		fp = unsafe.Pointer(*(*uintptr)(fp))
	}
	return i
}

// fpunwindExpand checks if pcBuf contains logical frames (which include inlined
// frames) or physical frames (produced by frame pointer unwinding) using a
// sentinel value in pcBuf[0]. Logical frames are simply returned without the
// sentinel. Physical frames are turned into logical frames via inline unwinding
// and by applying the skip value that's stored in pcBuf[0].
func fpunwindExpand(pcBuf []uintptr) []uintptr {
	if len(pcBuf) > 0 && pcBuf[0] == logicalStackSentinel {
		// pcBuf contains logical rather than inlined frames, skip has already been
		// applied, just return it without the sentinel value in pcBuf[0].
		return pcBuf[1:]
	}

	var (
		lastFuncID = abi.FuncIDNormal
		newPCBuf   = make([]uintptr, 0, traceStackSize)
		skip       = pcBuf[0]
		// skipOrAdd skips or appends retPC to newPCBuf and returns true if more
		// pcs can be added.
		skipOrAdd = func(retPC uintptr) bool {
			if skip > 0 {
				skip--
			} else {
				newPCBuf = append(newPCBuf, retPC)
			}
			return len(newPCBuf) < cap(newPCBuf)
		}
	)

outer:
	for _, retPC := range pcBuf[1:] {
		callPC := retPC - 1
		fi := findfunc(callPC)
		if !fi.valid() {
			// There is no funcInfo if callPC belongs to a C function. In this case
			// we still keep the pc, but don't attempt to expand inlined frames.
			if more := skipOrAdd(retPC); !more {
				break outer
			}
			continue
		}

		u, uf := newInlineUnwinder(fi, callPC)
		for ; uf.valid(); uf = u.next(uf) {
			sf := u.srcFunc(uf)
			if sf.funcID == abi.FuncIDWrapper && elideWrapperCalling(lastFuncID) {
				// ignore wrappers
			} else if more := skipOrAdd(uf.pc + 1); !more {
				break outer
			}
			lastFuncID = sf.funcID
		}
	}
	return newPCBuf
}

// startPCForTrace returns the start PC of a goroutine for tracing purposes.
// If pc is a wrapper, it returns the PC of the wrapped function. Otherwise it
// returns pc.
func startPCForTrace(pc uintptr) uintptr {
	f := findfunc(pc)
	if !f.valid() {
		return pc // may happen for locked g in extra M since its pc is 0.
	}
	w := funcdata(f, abi.FUNCDATA_WrapInfo)
	if w == nil {
		return pc // not a wrapper
	}
	return f.datap.textAddr(*(*uint32)(w))
}
