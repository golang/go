// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Go execution tracer.
// The tracer captures a wide range of execution events like goroutine
// creation/blocking/unblocking, syscall enter/exit/block, GC-related events,
// changes of heap size, processor start/stop, etc and writes them to a buffer
// in a compact form. A precise nanosecond-precision timestamp and a stack
// trace is captured for most events.
// See https://golang.org/s/go15trace for more info.

package runtime

import (
	"internal/goarch"
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// Event types in the trace, args are given in square brackets.
const (
	traceEvNone              = 0  // unused
	traceEvBatch             = 1  // start of per-P batch of events [pid, timestamp]
	traceEvFrequency         = 2  // contains tracer timer frequency [frequency (ticks per second)]
	traceEvStack             = 3  // stack [stack id, number of PCs, array of {PC, func string ID, file string ID, line}]
	traceEvGomaxprocs        = 4  // current value of GOMAXPROCS [timestamp, GOMAXPROCS, stack id]
	traceEvProcStart         = 5  // start of P [timestamp, thread id]
	traceEvProcStop          = 6  // stop of P [timestamp]
	traceEvGCStart           = 7  // GC start [timestamp, seq, stack id]
	traceEvGCDone            = 8  // GC done [timestamp]
	traceEvGCSTWStart        = 9  // GC STW start [timestamp, kind]
	traceEvGCSTWDone         = 10 // GC STW done [timestamp]
	traceEvGCSweepStart      = 11 // GC sweep start [timestamp, stack id]
	traceEvGCSweepDone       = 12 // GC sweep done [timestamp, swept, reclaimed]
	traceEvGoCreate          = 13 // goroutine creation [timestamp, new goroutine id, new stack id, stack id]
	traceEvGoStart           = 14 // goroutine starts running [timestamp, goroutine id, seq]
	traceEvGoEnd             = 15 // goroutine ends [timestamp]
	traceEvGoStop            = 16 // goroutine stops (like in select{}) [timestamp, stack]
	traceEvGoSched           = 17 // goroutine calls Gosched [timestamp, stack]
	traceEvGoPreempt         = 18 // goroutine is preempted [timestamp, stack]
	traceEvGoSleep           = 19 // goroutine calls Sleep [timestamp, stack]
	traceEvGoBlock           = 20 // goroutine blocks [timestamp, stack]
	traceEvGoUnblock         = 21 // goroutine is unblocked [timestamp, goroutine id, seq, stack]
	traceEvGoBlockSend       = 22 // goroutine blocks on chan send [timestamp, stack]
	traceEvGoBlockRecv       = 23 // goroutine blocks on chan recv [timestamp, stack]
	traceEvGoBlockSelect     = 24 // goroutine blocks on select [timestamp, stack]
	traceEvGoBlockSync       = 25 // goroutine blocks on Mutex/RWMutex [timestamp, stack]
	traceEvGoBlockCond       = 26 // goroutine blocks on Cond [timestamp, stack]
	traceEvGoBlockNet        = 27 // goroutine blocks on network [timestamp, stack]
	traceEvGoSysCall         = 28 // syscall enter [timestamp, stack]
	traceEvGoSysExit         = 29 // syscall exit [timestamp, goroutine id, seq, real timestamp]
	traceEvGoSysBlock        = 30 // syscall blocks [timestamp]
	traceEvGoWaiting         = 31 // denotes that goroutine is blocked when tracing starts [timestamp, goroutine id]
	traceEvGoInSyscall       = 32 // denotes that goroutine is in syscall when tracing starts [timestamp, goroutine id]
	traceEvHeapAlloc         = 33 // gcController.heapLive change [timestamp, heap_alloc]
	traceEvHeapGoal          = 34 // gcController.heapGoal (formerly next_gc) change [timestamp, heap goal in bytes]
	traceEvTimerGoroutine    = 35 // not currently used; previously denoted timer goroutine [timer goroutine id]
	traceEvFutileWakeup      = 36 // denotes that the previous wakeup of this goroutine was futile [timestamp]
	traceEvString            = 37 // string dictionary entry [ID, length, string]
	traceEvGoStartLocal      = 38 // goroutine starts running on the same P as the last event [timestamp, goroutine id]
	traceEvGoUnblockLocal    = 39 // goroutine is unblocked on the same P as the last event [timestamp, goroutine id, stack]
	traceEvGoSysExitLocal    = 40 // syscall exit on the same P as the last event [timestamp, goroutine id, real timestamp]
	traceEvGoStartLabel      = 41 // goroutine starts running with label [timestamp, goroutine id, seq, label string id]
	traceEvGoBlockGC         = 42 // goroutine blocks on GC assist [timestamp, stack]
	traceEvGCMarkAssistStart = 43 // GC mark assist start [timestamp, stack]
	traceEvGCMarkAssistDone  = 44 // GC mark assist done [timestamp]
	traceEvUserTaskCreate    = 45 // trace.NewContext [timestamp, internal task id, internal parent task id, stack, name string]
	traceEvUserTaskEnd       = 46 // end of a task [timestamp, internal task id, stack]
	traceEvUserRegion        = 47 // trace.WithRegion [timestamp, internal task id, mode(0:start, 1:end), stack, name string]
	traceEvUserLog           = 48 // trace.Log [timestamp, internal task id, key string id, stack, value string]
	traceEvCount             = 49
	// Byte is used but only 6 bits are available for event type.
	// The remaining 2 bits are used to specify the number of arguments.
	// That means, the max event type value is 63.
)

const (
	// Timestamps in trace are cputicks/traceTickDiv.
	// This makes absolute values of timestamp diffs smaller,
	// and so they are encoded in less number of bytes.
	// 64 on x86 is somewhat arbitrary (one tick is ~20ns on a 3GHz machine).
	// The suggested increment frequency for PowerPC's time base register is
	// 512 MHz according to Power ISA v2.07 section 6.2, so we use 16 on ppc64
	// and ppc64le.
	// Tracing won't work reliably for architectures where cputicks is emulated
	// by nanotime, so the value doesn't matter for those architectures.
	traceTickDiv = 16 + 48*(goarch.Is386|goarch.IsAmd64)
	// Maximum number of PCs in a single stack trace.
	// Since events contain only stack id rather than whole stack trace,
	// we can allow quite large values here.
	traceStackSize = 128
	// Identifier of a fake P that is used when we trace without a real P.
	traceGlobProc = -1
	// Maximum number of bytes to encode uint64 in base-128.
	traceBytesPerNumber = 10
	// Shift of the number of arguments in the first event byte.
	traceArgCountShift = 6
	// Flag passed to traceGoPark to denote that the previous wakeup of this
	// goroutine was futile. For example, a goroutine was unblocked on a mutex,
	// but another goroutine got ahead and acquired the mutex before the first
	// goroutine is scheduled, so the first goroutine has to block again.
	// Such wakeups happen on buffered channels and sync.Mutex,
	// but are generally not interesting for end user.
	traceFutileWakeup byte = 128
)

// trace is global tracing context.
var trace struct {
	lock          mutex       // protects the following members
	lockOwner     *g          // to avoid deadlocks during recursive lock locks
	enabled       bool        // when set runtime traces events
	shutdown      bool        // set when we are waiting for trace reader to finish after setting enabled to false
	headerWritten bool        // whether ReadTrace has emitted trace header
	footerWritten bool        // whether ReadTrace has emitted trace footer
	shutdownSema  uint32      // used to wait for ReadTrace completion
	seqStart      uint64      // sequence number when tracing was started
	ticksStart    int64       // cputicks when tracing was started
	ticksEnd      int64       // cputicks when tracing was stopped
	timeStart     int64       // nanotime when tracing was started
	timeEnd       int64       // nanotime when tracing was stopped
	seqGC         uint64      // GC start/done sequencer
	reading       traceBufPtr // buffer currently handed off to user
	empty         traceBufPtr // stack of empty buffers
	fullHead      traceBufPtr // queue of full buffers
	fullTail      traceBufPtr
	reader        guintptr        // goroutine that called ReadTrace, or nil
	stackTab      traceStackTable // maps stack traces to unique ids

	// Dictionary for traceEvString.
	//
	// TODO: central lock to access the map is not ideal.
	//   option: pre-assign ids to all user annotation region names and tags
	//   option: per-P cache
	//   option: sync.Map like data structure
	stringsLock mutex
	strings     map[string]uint64
	stringSeq   uint64

	// markWorkerLabels maps gcMarkWorkerMode to string ID.
	markWorkerLabels [len(gcMarkWorkerModeStrings)]uint64

	bufLock mutex       // protects buf
	buf     traceBufPtr // global trace buffer, used when running without a p
}

// traceBufHeader is per-P tracing buffer.
type traceBufHeader struct {
	link      traceBufPtr             // in trace.empty/full
	lastTicks uint64                  // when we wrote the last event
	pos       int                     // next write offset in arr
	stk       [traceStackSize]uintptr // scratch buffer for traceback
}

// traceBuf is per-P tracing buffer.
//
//go:notinheap
type traceBuf struct {
	traceBufHeader
	arr [64<<10 - unsafe.Sizeof(traceBufHeader{})]byte // underlying buffer for traceBufHeader.buf
}

// traceBufPtr is a *traceBuf that is not traced by the garbage
// collector and doesn't have write barriers. traceBufs are not
// allocated from the GC'd heap, so this is safe, and are often
// manipulated in contexts where write barriers are not allowed, so
// this is necessary.
//
// TODO: Since traceBuf is now go:notinheap, this isn't necessary.
type traceBufPtr uintptr

func (tp traceBufPtr) ptr() *traceBuf   { return (*traceBuf)(unsafe.Pointer(tp)) }
func (tp *traceBufPtr) set(b *traceBuf) { *tp = traceBufPtr(unsafe.Pointer(b)) }
func traceBufPtrOf(b *traceBuf) traceBufPtr {
	return traceBufPtr(unsafe.Pointer(b))
}

// StartTrace enables tracing for the current process.
// While tracing, the data will be buffered and available via ReadTrace.
// StartTrace returns an error if tracing is already enabled.
// Most clients should use the runtime/trace package or the testing package's
// -test.trace flag instead of calling StartTrace directly.
func StartTrace() error {
	// Stop the world so that we can take a consistent snapshot
	// of all goroutines at the beginning of the trace.
	// Do not stop the world during GC so we ensure we always see
	// a consistent view of GC-related events (e.g. a start is always
	// paired with an end).
	stopTheWorldGC("start tracing")

	// Prevent sysmon from running any code that could generate events.
	lock(&sched.sysmonlock)

	// We are in stop-the-world, but syscalls can finish and write to trace concurrently.
	// Exitsyscall could check trace.enabled long before and then suddenly wake up
	// and decide to write to trace at a random point in time.
	// However, such syscall will use the global trace.buf buffer, because we've
	// acquired all p's by doing stop-the-world. So this protects us from such races.
	lock(&trace.bufLock)

	if trace.enabled || trace.shutdown {
		unlock(&trace.bufLock)
		unlock(&sched.sysmonlock)
		startTheWorldGC()
		return errorString("tracing is already enabled")
	}

	// Can't set trace.enabled yet. While the world is stopped, exitsyscall could
	// already emit a delayed event (see exitTicks in exitsyscall) if we set trace.enabled here.
	// That would lead to an inconsistent trace:
	// - either GoSysExit appears before EvGoInSyscall,
	// - or GoSysExit appears for a goroutine for which we don't emit EvGoInSyscall below.
	// To instruct traceEvent that it must not ignore events below, we set startingtrace.
	// trace.enabled is set afterwards once we have emitted all preliminary events.
	_g_ := getg()
	_g_.m.startingtrace = true

	// Obtain current stack ID to use in all traceEvGoCreate events below.
	mp := acquirem()
	stkBuf := make([]uintptr, traceStackSize)
	stackID := traceStackID(mp, stkBuf, 2)
	releasem(mp)

	// World is stopped, no need to lock.
	forEachGRace(func(gp *g) {
		status := readgstatus(gp)
		if status != _Gdead {
			gp.traceseq = 0
			gp.tracelastp = getg().m.p
			// +PCQuantum because traceFrameForPC expects return PCs and subtracts PCQuantum.
			id := trace.stackTab.put([]uintptr{startPCforTrace(gp.startpc) + sys.PCQuantum})
			traceEvent(traceEvGoCreate, -1, uint64(gp.goid), uint64(id), stackID)
		}
		if status == _Gwaiting {
			// traceEvGoWaiting is implied to have seq=1.
			gp.traceseq++
			traceEvent(traceEvGoWaiting, -1, uint64(gp.goid))
		}
		if status == _Gsyscall {
			gp.traceseq++
			traceEvent(traceEvGoInSyscall, -1, uint64(gp.goid))
		} else {
			gp.sysblocktraced = false
		}
	})
	traceProcStart()
	traceGoStart()
	// Note: ticksStart needs to be set after we emit traceEvGoInSyscall events.
	// If we do it the other way around, it is possible that exitsyscall will
	// query sysexitticks after ticksStart but before traceEvGoInSyscall timestamp.
	// It will lead to a false conclusion that cputicks is broken.
	trace.ticksStart = cputicks()
	trace.timeStart = nanotime()
	trace.headerWritten = false
	trace.footerWritten = false

	// string to id mapping
	//  0 : reserved for an empty string
	//  remaining: other strings registered by traceString
	trace.stringSeq = 0
	trace.strings = make(map[string]uint64)

	trace.seqGC = 0
	_g_.m.startingtrace = false
	trace.enabled = true

	// Register runtime goroutine labels.
	_, pid, bufp := traceAcquireBuffer()
	for i, label := range gcMarkWorkerModeStrings[:] {
		trace.markWorkerLabels[i], bufp = traceString(bufp, pid, label)
	}
	traceReleaseBuffer(pid)

	unlock(&trace.bufLock)

	unlock(&sched.sysmonlock)

	startTheWorldGC()
	return nil
}

// StopTrace stops tracing, if it was previously enabled.
// StopTrace only returns after all the reads for the trace have completed.
func StopTrace() {
	// Stop the world so that we can collect the trace buffers from all p's below,
	// and also to avoid races with traceEvent.
	stopTheWorldGC("stop tracing")

	// See the comment in StartTrace.
	lock(&sched.sysmonlock)

	// See the comment in StartTrace.
	lock(&trace.bufLock)

	if !trace.enabled {
		unlock(&trace.bufLock)
		unlock(&sched.sysmonlock)
		startTheWorldGC()
		return
	}

	traceGoSched()

	// Loop over all allocated Ps because dead Ps may still have
	// trace buffers.
	for _, p := range allp[:cap(allp)] {
		buf := p.tracebuf
		if buf != 0 {
			traceFullQueue(buf)
			p.tracebuf = 0
		}
	}
	if trace.buf != 0 {
		buf := trace.buf
		trace.buf = 0
		if buf.ptr().pos != 0 {
			traceFullQueue(buf)
		}
	}

	for {
		trace.ticksEnd = cputicks()
		trace.timeEnd = nanotime()
		// Windows time can tick only every 15ms, wait for at least one tick.
		if trace.timeEnd != trace.timeStart {
			break
		}
		osyield()
	}

	trace.enabled = false
	trace.shutdown = true
	unlock(&trace.bufLock)

	unlock(&sched.sysmonlock)

	startTheWorldGC()

	// The world is started but we've set trace.shutdown, so new tracing can't start.
	// Wait for the trace reader to flush pending buffers and stop.
	semacquire(&trace.shutdownSema)
	if raceenabled {
		raceacquire(unsafe.Pointer(&trace.shutdownSema))
	}

	// The lock protects us from races with StartTrace/StopTrace because they do stop-the-world.
	lock(&trace.lock)
	for _, p := range allp[:cap(allp)] {
		if p.tracebuf != 0 {
			throw("trace: non-empty trace buffer in proc")
		}
	}
	if trace.buf != 0 {
		throw("trace: non-empty global trace buffer")
	}
	if trace.fullHead != 0 || trace.fullTail != 0 {
		throw("trace: non-empty full trace buffer")
	}
	if trace.reading != 0 || trace.reader != 0 {
		throw("trace: reading after shutdown")
	}
	for trace.empty != 0 {
		buf := trace.empty
		trace.empty = buf.ptr().link
		sysFree(unsafe.Pointer(buf), unsafe.Sizeof(*buf.ptr()), &memstats.other_sys)
	}
	trace.strings = nil
	trace.shutdown = false
	unlock(&trace.lock)
}

// ReadTrace returns the next chunk of binary tracing data, blocking until data
// is available. If tracing is turned off and all the data accumulated while it
// was on has been returned, ReadTrace returns nil. The caller must copy the
// returned data before calling ReadTrace again.
// ReadTrace must be called from one goroutine at a time.
func ReadTrace() []byte {
	// This function may need to lock trace.lock recursively
	// (goparkunlock -> traceGoPark -> traceEvent -> traceFlush).
	// To allow this we use trace.lockOwner.
	// Also this function must not allocate while holding trace.lock:
	// allocation can call heap allocate, which will try to emit a trace
	// event while holding heap lock.
	lock(&trace.lock)
	trace.lockOwner = getg()

	if trace.reader != 0 {
		// More than one goroutine reads trace. This is bad.
		// But we rather do not crash the program because of tracing,
		// because tracing can be enabled at runtime on prod servers.
		trace.lockOwner = nil
		unlock(&trace.lock)
		println("runtime: ReadTrace called from multiple goroutines simultaneously")
		return nil
	}
	// Recycle the old buffer.
	if buf := trace.reading; buf != 0 {
		buf.ptr().link = trace.empty
		trace.empty = buf
		trace.reading = 0
	}
	// Write trace header.
	if !trace.headerWritten {
		trace.headerWritten = true
		trace.lockOwner = nil
		unlock(&trace.lock)
		return []byte("go 1.11 trace\x00\x00\x00")
	}
	// Wait for new data.
	if trace.fullHead == 0 && !trace.shutdown {
		trace.reader.set(getg())
		goparkunlock(&trace.lock, waitReasonTraceReaderBlocked, traceEvGoBlock, 2)
		lock(&trace.lock)
	}
	// Write a buffer.
	if trace.fullHead != 0 {
		buf := traceFullDequeue()
		trace.reading = buf
		trace.lockOwner = nil
		unlock(&trace.lock)
		return buf.ptr().arr[:buf.ptr().pos]
	}
	// Write footer with timer frequency.
	if !trace.footerWritten {
		trace.footerWritten = true
		// Use float64 because (trace.ticksEnd - trace.ticksStart) * 1e9 can overflow int64.
		freq := float64(trace.ticksEnd-trace.ticksStart) * 1e9 / float64(trace.timeEnd-trace.timeStart) / traceTickDiv
		if freq <= 0 {
			throw("trace: ReadTrace got invalid frequency")
		}
		trace.lockOwner = nil
		unlock(&trace.lock)
		var data []byte
		data = append(data, traceEvFrequency|0<<traceArgCountShift)
		data = traceAppend(data, uint64(freq))
		// This will emit a bunch of full buffers, we will pick them up
		// on the next iteration.
		trace.stackTab.dump()
		return data
	}
	// Done.
	if trace.shutdown {
		trace.lockOwner = nil
		unlock(&trace.lock)
		if raceenabled {
			// Model synchronization on trace.shutdownSema, which race
			// detector does not see. This is required to avoid false
			// race reports on writer passed to trace.Start.
			racerelease(unsafe.Pointer(&trace.shutdownSema))
		}
		// trace.enabled is already reset, so can call traceable functions.
		semrelease(&trace.shutdownSema)
		return nil
	}
	// Also bad, but see the comment above.
	trace.lockOwner = nil
	unlock(&trace.lock)
	println("runtime: spurious wakeup of trace reader")
	return nil
}

// traceReader returns the trace reader that should be woken up, if any.
func traceReader() *g {
	if trace.reader == 0 || (trace.fullHead == 0 && !trace.shutdown) {
		return nil
	}
	lock(&trace.lock)
	if trace.reader == 0 || (trace.fullHead == 0 && !trace.shutdown) {
		unlock(&trace.lock)
		return nil
	}
	gp := trace.reader.ptr()
	trace.reader.set(nil)
	unlock(&trace.lock)
	return gp
}

// traceProcFree frees trace buffer associated with pp.
func traceProcFree(pp *p) {
	buf := pp.tracebuf
	pp.tracebuf = 0
	if buf == 0 {
		return
	}
	lock(&trace.lock)
	traceFullQueue(buf)
	unlock(&trace.lock)
}

// traceFullQueue queues buf into queue of full buffers.
func traceFullQueue(buf traceBufPtr) {
	buf.ptr().link = 0
	if trace.fullHead == 0 {
		trace.fullHead = buf
	} else {
		trace.fullTail.ptr().link = buf
	}
	trace.fullTail = buf
}

// traceFullDequeue dequeues from queue of full buffers.
func traceFullDequeue() traceBufPtr {
	buf := trace.fullHead
	if buf == 0 {
		return 0
	}
	trace.fullHead = buf.ptr().link
	if trace.fullHead == 0 {
		trace.fullTail = 0
	}
	buf.ptr().link = 0
	return buf
}

// traceEvent writes a single event to trace buffer, flushing the buffer if necessary.
// ev is event type.
// If skip > 0, write current stack id as the last argument (skipping skip top frames).
// If skip = 0, this event type should contain a stack, but we don't want
// to collect and remember it for this particular call.
func traceEvent(ev byte, skip int, args ...uint64) {
	mp, pid, bufp := traceAcquireBuffer()
	// Double-check trace.enabled now that we've done m.locks++ and acquired bufLock.
	// This protects from races between traceEvent and StartTrace/StopTrace.

	// The caller checked that trace.enabled == true, but trace.enabled might have been
	// turned off between the check and now. Check again. traceLockBuffer did mp.locks++,
	// StopTrace does stopTheWorld, and stopTheWorld waits for mp.locks to go back to zero,
	// so if we see trace.enabled == true now, we know it's true for the rest of the function.
	// Exitsyscall can run even during stopTheWorld. The race with StartTrace/StopTrace
	// during tracing in exitsyscall is resolved by locking trace.bufLock in traceLockBuffer.
	//
	// Note trace_userTaskCreate runs the same check.
	if !trace.enabled && !mp.startingtrace {
		traceReleaseBuffer(pid)
		return
	}

	if skip > 0 {
		if getg() == mp.curg {
			skip++ // +1 because stack is captured in traceEventLocked.
		}
	}
	traceEventLocked(0, mp, pid, bufp, ev, skip, args...)
	traceReleaseBuffer(pid)
}

func traceEventLocked(extraBytes int, mp *m, pid int32, bufp *traceBufPtr, ev byte, skip int, args ...uint64) {
	buf := bufp.ptr()
	// TODO: test on non-zero extraBytes param.
	maxSize := 2 + 5*traceBytesPerNumber + extraBytes // event type, length, sequence, timestamp, stack id and two add params
	if buf == nil || len(buf.arr)-buf.pos < maxSize {
		buf = traceFlush(traceBufPtrOf(buf), pid).ptr()
		bufp.set(buf)
	}

	// NOTE: ticks might be same after tick division, although the real cputicks is
	// linear growth.
	ticks := uint64(cputicks()) / traceTickDiv
	tickDiff := ticks - buf.lastTicks
	if tickDiff == 0 {
		ticks = buf.lastTicks + 1
		tickDiff = 1
	}

	buf.lastTicks = ticks
	narg := byte(len(args))
	if skip >= 0 {
		narg++
	}
	// We have only 2 bits for number of arguments.
	// If number is >= 3, then the event type is followed by event length in bytes.
	if narg > 3 {
		narg = 3
	}
	startPos := buf.pos
	buf.byte(ev | narg<<traceArgCountShift)
	var lenp *byte
	if narg == 3 {
		// Reserve the byte for length assuming that length < 128.
		buf.varint(0)
		lenp = &buf.arr[buf.pos-1]
	}
	buf.varint(tickDiff)
	for _, a := range args {
		buf.varint(a)
	}
	if skip == 0 {
		buf.varint(0)
	} else if skip > 0 {
		buf.varint(traceStackID(mp, buf.stk[:], skip))
	}
	evSize := buf.pos - startPos
	if evSize > maxSize {
		throw("invalid length of trace event")
	}
	if lenp != nil {
		// Fill in actual length.
		*lenp = byte(evSize - 2)
	}
}

func traceStackID(mp *m, buf []uintptr, skip int) uint64 {
	_g_ := getg()
	gp := mp.curg
	var nstk int
	if gp == _g_ {
		nstk = callers(skip+1, buf)
	} else if gp != nil {
		gp = mp.curg
		nstk = gcallers(gp, skip, buf)
	}
	if nstk > 0 {
		nstk-- // skip runtime.goexit
	}
	if nstk > 0 && gp.goid == 1 {
		nstk-- // skip runtime.main
	}
	id := trace.stackTab.put(buf[:nstk])
	return uint64(id)
}

// traceAcquireBuffer returns trace buffer to use and, if necessary, locks it.
func traceAcquireBuffer() (mp *m, pid int32, bufp *traceBufPtr) {
	mp = acquirem()
	if p := mp.p.ptr(); p != nil {
		return mp, p.id, &p.tracebuf
	}
	lock(&trace.bufLock)
	return mp, traceGlobProc, &trace.buf
}

// traceReleaseBuffer releases a buffer previously acquired with traceAcquireBuffer.
func traceReleaseBuffer(pid int32) {
	if pid == traceGlobProc {
		unlock(&trace.bufLock)
	}
	releasem(getg().m)
}

// traceFlush puts buf onto stack of full buffers and returns an empty buffer.
func traceFlush(buf traceBufPtr, pid int32) traceBufPtr {
	owner := trace.lockOwner
	dolock := owner == nil || owner != getg().m.curg
	if dolock {
		lock(&trace.lock)
	}
	if buf != 0 {
		traceFullQueue(buf)
	}
	if trace.empty != 0 {
		buf = trace.empty
		trace.empty = buf.ptr().link
	} else {
		buf = traceBufPtr(sysAlloc(unsafe.Sizeof(traceBuf{}), &memstats.other_sys))
		if buf == 0 {
			throw("trace: out of memory")
		}
	}
	bufp := buf.ptr()
	bufp.link.set(nil)
	bufp.pos = 0

	// initialize the buffer for a new batch
	ticks := uint64(cputicks()) / traceTickDiv
	if ticks == bufp.lastTicks {
		ticks = bufp.lastTicks + 1
	}
	bufp.lastTicks = ticks
	bufp.byte(traceEvBatch | 1<<traceArgCountShift)
	bufp.varint(uint64(pid))
	bufp.varint(ticks)

	if dolock {
		unlock(&trace.lock)
	}
	return buf
}

// traceString adds a string to the trace.strings and returns the id.
func traceString(bufp *traceBufPtr, pid int32, s string) (uint64, *traceBufPtr) {
	if s == "" {
		return 0, bufp
	}

	lock(&trace.stringsLock)
	if raceenabled {
		// raceacquire is necessary because the map access
		// below is race annotated.
		raceacquire(unsafe.Pointer(&trace.stringsLock))
	}

	if id, ok := trace.strings[s]; ok {
		if raceenabled {
			racerelease(unsafe.Pointer(&trace.stringsLock))
		}
		unlock(&trace.stringsLock)

		return id, bufp
	}

	trace.stringSeq++
	id := trace.stringSeq
	trace.strings[s] = id

	if raceenabled {
		racerelease(unsafe.Pointer(&trace.stringsLock))
	}
	unlock(&trace.stringsLock)

	// memory allocation in above may trigger tracing and
	// cause *bufp changes. Following code now works with *bufp,
	// so there must be no memory allocation or any activities
	// that causes tracing after this point.

	buf := bufp.ptr()
	size := 1 + 2*traceBytesPerNumber + len(s)
	if buf == nil || len(buf.arr)-buf.pos < size {
		buf = traceFlush(traceBufPtrOf(buf), pid).ptr()
		bufp.set(buf)
	}
	buf.byte(traceEvString)
	buf.varint(id)

	// double-check the string and the length can fit.
	// Otherwise, truncate the string.
	slen := len(s)
	if room := len(buf.arr) - buf.pos; room < slen+traceBytesPerNumber {
		slen = room
	}

	buf.varint(uint64(slen))
	buf.pos += copy(buf.arr[buf.pos:], s[:slen])

	bufp.set(buf)
	return id, bufp
}

// traceAppend appends v to buf in little-endian-base-128 encoding.
func traceAppend(buf []byte, v uint64) []byte {
	for ; v >= 0x80; v >>= 7 {
		buf = append(buf, 0x80|byte(v))
	}
	buf = append(buf, byte(v))
	return buf
}

// varint appends v to buf in little-endian-base-128 encoding.
func (buf *traceBuf) varint(v uint64) {
	pos := buf.pos
	for ; v >= 0x80; v >>= 7 {
		buf.arr[pos] = 0x80 | byte(v)
		pos++
	}
	buf.arr[pos] = byte(v)
	pos++
	buf.pos = pos
}

// byte appends v to buf.
func (buf *traceBuf) byte(v byte) {
	buf.arr[buf.pos] = v
	buf.pos++
}

// traceStackTable maps stack traces (arrays of PC's) to unique uint32 ids.
// It is lock-free for reading.
type traceStackTable struct {
	lock mutex
	seq  uint32
	mem  traceAlloc
	tab  [1 << 13]traceStackPtr
}

// traceStack is a single stack in traceStackTable.
type traceStack struct {
	link traceStackPtr
	hash uintptr
	id   uint32
	n    int
	stk  [0]uintptr // real type [n]uintptr
}

type traceStackPtr uintptr

func (tp traceStackPtr) ptr() *traceStack { return (*traceStack)(unsafe.Pointer(tp)) }

// stack returns slice of PCs.
func (ts *traceStack) stack() []uintptr {
	return (*[traceStackSize]uintptr)(unsafe.Pointer(&ts.stk))[:ts.n]
}

// put returns a unique id for the stack trace pcs and caches it in the table,
// if it sees the trace for the first time.
func (tab *traceStackTable) put(pcs []uintptr) uint32 {
	if len(pcs) == 0 {
		return 0
	}
	hash := memhash(unsafe.Pointer(&pcs[0]), 0, uintptr(len(pcs))*unsafe.Sizeof(pcs[0]))
	// First, search the hashtable w/o the mutex.
	if id := tab.find(pcs, hash); id != 0 {
		return id
	}
	// Now, double check under the mutex.
	lock(&tab.lock)
	if id := tab.find(pcs, hash); id != 0 {
		unlock(&tab.lock)
		return id
	}
	// Create new record.
	tab.seq++
	stk := tab.newStack(len(pcs))
	stk.hash = hash
	stk.id = tab.seq
	stk.n = len(pcs)
	stkpc := stk.stack()
	for i, pc := range pcs {
		stkpc[i] = pc
	}
	part := int(hash % uintptr(len(tab.tab)))
	stk.link = tab.tab[part]
	atomicstorep(unsafe.Pointer(&tab.tab[part]), unsafe.Pointer(stk))
	unlock(&tab.lock)
	return stk.id
}

// find checks if the stack trace pcs is already present in the table.
func (tab *traceStackTable) find(pcs []uintptr, hash uintptr) uint32 {
	part := int(hash % uintptr(len(tab.tab)))
Search:
	for stk := tab.tab[part].ptr(); stk != nil; stk = stk.link.ptr() {
		if stk.hash == hash && stk.n == len(pcs) {
			for i, stkpc := range stk.stack() {
				if stkpc != pcs[i] {
					continue Search
				}
			}
			return stk.id
		}
	}
	return 0
}

// newStack allocates a new stack of size n.
func (tab *traceStackTable) newStack(n int) *traceStack {
	return (*traceStack)(tab.mem.alloc(unsafe.Sizeof(traceStack{}) + uintptr(n)*goarch.PtrSize))
}

// allFrames returns all of the Frames corresponding to pcs.
func allFrames(pcs []uintptr) []Frame {
	frames := make([]Frame, 0, len(pcs))
	ci := CallersFrames(pcs)
	for {
		f, more := ci.Next()
		frames = append(frames, f)
		if !more {
			return frames
		}
	}
}

// dump writes all previously cached stacks to trace buffers,
// releases all memory and resets state.
func (tab *traceStackTable) dump() {
	var tmp [(2 + 4*traceStackSize) * traceBytesPerNumber]byte
	bufp := traceFlush(0, 0)
	for _, stk := range tab.tab {
		stk := stk.ptr()
		for ; stk != nil; stk = stk.link.ptr() {
			tmpbuf := tmp[:0]
			tmpbuf = traceAppend(tmpbuf, uint64(stk.id))
			frames := allFrames(stk.stack())
			tmpbuf = traceAppend(tmpbuf, uint64(len(frames)))
			for _, f := range frames {
				var frame traceFrame
				frame, bufp = traceFrameForPC(bufp, 0, f)
				tmpbuf = traceAppend(tmpbuf, uint64(f.PC))
				tmpbuf = traceAppend(tmpbuf, uint64(frame.funcID))
				tmpbuf = traceAppend(tmpbuf, uint64(frame.fileID))
				tmpbuf = traceAppend(tmpbuf, uint64(frame.line))
			}
			// Now copy to the buffer.
			size := 1 + traceBytesPerNumber + len(tmpbuf)
			if buf := bufp.ptr(); len(buf.arr)-buf.pos < size {
				bufp = traceFlush(bufp, 0)
			}
			buf := bufp.ptr()
			buf.byte(traceEvStack | 3<<traceArgCountShift)
			buf.varint(uint64(len(tmpbuf)))
			buf.pos += copy(buf.arr[buf.pos:], tmpbuf)
		}
	}

	lock(&trace.lock)
	traceFullQueue(bufp)
	unlock(&trace.lock)

	tab.mem.drop()
	*tab = traceStackTable{}
	lockInit(&((*tab).lock), lockRankTraceStackTab)
}

type traceFrame struct {
	funcID uint64
	fileID uint64
	line   uint64
}

// traceFrameForPC records the frame information.
// It may allocate memory.
func traceFrameForPC(buf traceBufPtr, pid int32, f Frame) (traceFrame, traceBufPtr) {
	bufp := &buf
	var frame traceFrame

	fn := f.Function
	const maxLen = 1 << 10
	if len(fn) > maxLen {
		fn = fn[len(fn)-maxLen:]
	}
	frame.funcID, bufp = traceString(bufp, pid, fn)
	frame.line = uint64(f.Line)
	file := f.File
	if len(file) > maxLen {
		file = file[len(file)-maxLen:]
	}
	frame.fileID, bufp = traceString(bufp, pid, file)
	return frame, (*bufp)
}

// traceAlloc is a non-thread-safe region allocator.
// It holds a linked list of traceAllocBlock.
type traceAlloc struct {
	head traceAllocBlockPtr
	off  uintptr
}

// traceAllocBlock is a block in traceAlloc.
//
// traceAllocBlock is allocated from non-GC'd memory, so it must not
// contain heap pointers. Writes to pointers to traceAllocBlocks do
// not need write barriers.
//
//go:notinheap
type traceAllocBlock struct {
	next traceAllocBlockPtr
	data [64<<10 - goarch.PtrSize]byte
}

// TODO: Since traceAllocBlock is now go:notinheap, this isn't necessary.
type traceAllocBlockPtr uintptr

func (p traceAllocBlockPtr) ptr() *traceAllocBlock   { return (*traceAllocBlock)(unsafe.Pointer(p)) }
func (p *traceAllocBlockPtr) set(x *traceAllocBlock) { *p = traceAllocBlockPtr(unsafe.Pointer(x)) }

// alloc allocates n-byte block.
func (a *traceAlloc) alloc(n uintptr) unsafe.Pointer {
	n = alignUp(n, goarch.PtrSize)
	if a.head == 0 || a.off+n > uintptr(len(a.head.ptr().data)) {
		if n > uintptr(len(a.head.ptr().data)) {
			throw("trace: alloc too large")
		}
		block := (*traceAllocBlock)(sysAlloc(unsafe.Sizeof(traceAllocBlock{}), &memstats.other_sys))
		if block == nil {
			throw("trace: out of memory")
		}
		block.next.set(a.head.ptr())
		a.head.set(block)
		a.off = 0
	}
	p := &a.head.ptr().data[a.off]
	a.off += n
	return unsafe.Pointer(p)
}

// drop frees all previously allocated memory and resets the allocator.
func (a *traceAlloc) drop() {
	for a.head != 0 {
		block := a.head.ptr()
		a.head.set(block.next.ptr())
		sysFree(unsafe.Pointer(block), unsafe.Sizeof(traceAllocBlock{}), &memstats.other_sys)
	}
}

// The following functions write specific events to trace.

func traceGomaxprocs(procs int32) {
	traceEvent(traceEvGomaxprocs, 1, uint64(procs))
}

func traceProcStart() {
	traceEvent(traceEvProcStart, -1, uint64(getg().m.id))
}

func traceProcStop(pp *p) {
	// Sysmon and stopTheWorld can stop Ps blocked in syscalls,
	// to handle this we temporary employ the P.
	mp := acquirem()
	oldp := mp.p
	mp.p.set(pp)
	traceEvent(traceEvProcStop, -1)
	mp.p = oldp
	releasem(mp)
}

func traceGCStart() {
	traceEvent(traceEvGCStart, 3, trace.seqGC)
	trace.seqGC++
}

func traceGCDone() {
	traceEvent(traceEvGCDone, -1)
}

func traceGCSTWStart(kind int) {
	traceEvent(traceEvGCSTWStart, -1, uint64(kind))
}

func traceGCSTWDone() {
	traceEvent(traceEvGCSTWDone, -1)
}

// traceGCSweepStart prepares to trace a sweep loop. This does not
// emit any events until traceGCSweepSpan is called.
//
// traceGCSweepStart must be paired with traceGCSweepDone and there
// must be no preemption points between these two calls.
func traceGCSweepStart() {
	// Delay the actual GCSweepStart event until the first span
	// sweep. If we don't sweep anything, don't emit any events.
	_p_ := getg().m.p.ptr()
	if _p_.traceSweep {
		throw("double traceGCSweepStart")
	}
	_p_.traceSweep, _p_.traceSwept, _p_.traceReclaimed = true, 0, 0
}

// traceGCSweepSpan traces the sweep of a single page.
//
// This may be called outside a traceGCSweepStart/traceGCSweepDone
// pair; however, it will not emit any trace events in this case.
func traceGCSweepSpan(bytesSwept uintptr) {
	_p_ := getg().m.p.ptr()
	if _p_.traceSweep {
		if _p_.traceSwept == 0 {
			traceEvent(traceEvGCSweepStart, 1)
		}
		_p_.traceSwept += bytesSwept
	}
}

func traceGCSweepDone() {
	_p_ := getg().m.p.ptr()
	if !_p_.traceSweep {
		throw("missing traceGCSweepStart")
	}
	if _p_.traceSwept != 0 {
		traceEvent(traceEvGCSweepDone, -1, uint64(_p_.traceSwept), uint64(_p_.traceReclaimed))
	}
	_p_.traceSweep = false
}

func traceGCMarkAssistStart() {
	traceEvent(traceEvGCMarkAssistStart, 1)
}

func traceGCMarkAssistDone() {
	traceEvent(traceEvGCMarkAssistDone, -1)
}

func traceGoCreate(newg *g, pc uintptr) {
	newg.traceseq = 0
	newg.tracelastp = getg().m.p
	// +PCQuantum because traceFrameForPC expects return PCs and subtracts PCQuantum.
	id := trace.stackTab.put([]uintptr{startPCforTrace(pc) + sys.PCQuantum})
	traceEvent(traceEvGoCreate, 2, uint64(newg.goid), uint64(id))
}

func traceGoStart() {
	_g_ := getg().m.curg
	_p_ := _g_.m.p
	_g_.traceseq++
	if _p_.ptr().gcMarkWorkerMode != gcMarkWorkerNotWorker {
		traceEvent(traceEvGoStartLabel, -1, uint64(_g_.goid), _g_.traceseq, trace.markWorkerLabels[_p_.ptr().gcMarkWorkerMode])
	} else if _g_.tracelastp == _p_ {
		traceEvent(traceEvGoStartLocal, -1, uint64(_g_.goid))
	} else {
		_g_.tracelastp = _p_
		traceEvent(traceEvGoStart, -1, uint64(_g_.goid), _g_.traceseq)
	}
}

func traceGoEnd() {
	traceEvent(traceEvGoEnd, -1)
}

func traceGoSched() {
	_g_ := getg()
	_g_.tracelastp = _g_.m.p
	traceEvent(traceEvGoSched, 1)
}

func traceGoPreempt() {
	_g_ := getg()
	_g_.tracelastp = _g_.m.p
	traceEvent(traceEvGoPreempt, 1)
}

func traceGoPark(traceEv byte, skip int) {
	if traceEv&traceFutileWakeup != 0 {
		traceEvent(traceEvFutileWakeup, -1)
	}
	traceEvent(traceEv & ^traceFutileWakeup, skip)
}

func traceGoUnpark(gp *g, skip int) {
	_p_ := getg().m.p
	gp.traceseq++
	if gp.tracelastp == _p_ {
		traceEvent(traceEvGoUnblockLocal, skip, uint64(gp.goid))
	} else {
		gp.tracelastp = _p_
		traceEvent(traceEvGoUnblock, skip, uint64(gp.goid), gp.traceseq)
	}
}

func traceGoSysCall() {
	traceEvent(traceEvGoSysCall, 1)
}

func traceGoSysExit(ts int64) {
	if ts != 0 && ts < trace.ticksStart {
		// There is a race between the code that initializes sysexitticks
		// (in exitsyscall, which runs without a P, and therefore is not
		// stopped with the rest of the world) and the code that initializes
		// a new trace. The recorded sysexitticks must therefore be treated
		// as "best effort". If they are valid for this trace, then great,
		// use them for greater accuracy. But if they're not valid for this
		// trace, assume that the trace was started after the actual syscall
		// exit (but before we actually managed to start the goroutine,
		// aka right now), and assign a fresh time stamp to keep the log consistent.
		ts = 0
	}
	_g_ := getg().m.curg
	_g_.traceseq++
	_g_.tracelastp = _g_.m.p
	traceEvent(traceEvGoSysExit, -1, uint64(_g_.goid), _g_.traceseq, uint64(ts)/traceTickDiv)
}

func traceGoSysBlock(pp *p) {
	// Sysmon and stopTheWorld can declare syscalls running on remote Ps as blocked,
	// to handle this we temporary employ the P.
	mp := acquirem()
	oldp := mp.p
	mp.p.set(pp)
	traceEvent(traceEvGoSysBlock, -1)
	mp.p = oldp
	releasem(mp)
}

func traceHeapAlloc() {
	traceEvent(traceEvHeapAlloc, -1, gcController.heapLive)
}

func traceHeapGoal() {
	if heapGoal := atomic.Load64(&gcController.heapGoal); heapGoal == ^uint64(0) {
		// Heap-based triggering is disabled.
		traceEvent(traceEvHeapGoal, -1, 0)
	} else {
		traceEvent(traceEvHeapGoal, -1, heapGoal)
	}
}

// To access runtime functions from runtime/trace.
// See runtime/trace/annotation.go

//go:linkname trace_userTaskCreate runtime/trace.userTaskCreate
func trace_userTaskCreate(id, parentID uint64, taskType string) {
	if !trace.enabled {
		return
	}

	// Same as in traceEvent.
	mp, pid, bufp := traceAcquireBuffer()
	if !trace.enabled && !mp.startingtrace {
		traceReleaseBuffer(pid)
		return
	}

	typeStringID, bufp := traceString(bufp, pid, taskType)
	traceEventLocked(0, mp, pid, bufp, traceEvUserTaskCreate, 3, id, parentID, typeStringID)
	traceReleaseBuffer(pid)
}

//go:linkname trace_userTaskEnd runtime/trace.userTaskEnd
func trace_userTaskEnd(id uint64) {
	traceEvent(traceEvUserTaskEnd, 2, id)
}

//go:linkname trace_userRegion runtime/trace.userRegion
func trace_userRegion(id, mode uint64, name string) {
	if !trace.enabled {
		return
	}

	mp, pid, bufp := traceAcquireBuffer()
	if !trace.enabled && !mp.startingtrace {
		traceReleaseBuffer(pid)
		return
	}

	nameStringID, bufp := traceString(bufp, pid, name)
	traceEventLocked(0, mp, pid, bufp, traceEvUserRegion, 3, id, mode, nameStringID)
	traceReleaseBuffer(pid)
}

//go:linkname trace_userLog runtime/trace.userLog
func trace_userLog(id uint64, category, message string) {
	if !trace.enabled {
		return
	}

	mp, pid, bufp := traceAcquireBuffer()
	if !trace.enabled && !mp.startingtrace {
		traceReleaseBuffer(pid)
		return
	}

	categoryID, bufp := traceString(bufp, pid, category)

	extraSpace := traceBytesPerNumber + len(message) // extraSpace for the value string
	traceEventLocked(extraSpace, mp, pid, bufp, traceEvUserLog, 3, id, categoryID)
	// traceEventLocked reserved extra space for val and len(val)
	// in buf, so buf now has room for the following.
	buf := bufp.ptr()

	// double-check the message and its length can fit.
	// Otherwise, truncate the message.
	slen := len(message)
	if room := len(buf.arr) - buf.pos; room < slen+traceBytesPerNumber {
		slen = room
	}
	buf.varint(uint64(slen))
	buf.pos += copy(buf.arr[buf.pos:], message[:slen])

	traceReleaseBuffer(pid)
}

// the start PC of a goroutine for tracing purposes. If pc is a wrapper,
// it returns the PC of the wrapped function. Otherwise it returns pc.
func startPCforTrace(pc uintptr) uintptr {
	f := findfunc(pc)
	if !f.valid() {
		return pc // should not happen, but don't care
	}
	w := funcdata(f, _FUNCDATA_WrapInfo)
	if w == nil {
		return pc // not a wrapper
	}
	return f.datap.textAddr(*(*uint32)(w))
}
